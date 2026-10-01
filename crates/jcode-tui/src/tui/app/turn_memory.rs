use super::*;

/// Own the local session binding so close/drop also rejects late retrievals.
pub(super) struct LocalTurnMemory {
    session_id: String,
    external: Option<crate::memory::ExternalTurnMemory>,
}

impl Drop for LocalTurnMemory {
    fn drop(&mut self) {
        crate::memory::clear_pending_memory(&self.session_id);
    }
}

impl App {
    /// Local mode has no daemon Agent to report its outcome. Emit raw, anchored
    /// records here even when recall is disabled. Remote clients never duplicate it.
    pub(super) fn fire_local_turn_end_hook(
        &self,
        status: &str,
        started: Instant,
        error: Option<String>,
    ) {
        if self.is_remote || !crate::hooks::hook_configured("turn_end") {
            return;
        }
        let records = crate::hooks::turn_records(&self.session);
        let mut event = crate::hooks::HookEvent::new("turn_end")
            .session_id(self.session.id.clone())
            .field("STATUS", status)
            .field("DURATION_MS", started.elapsed().as_millis().to_string())
            .field("MODEL", self.provider.model())
            .field("TURN_RECORDS_JSON", records.to_string());
        if let Some(turn_id) = &self.session.model_usage_turn_id {
            event = event.field("TURN_ID", turn_id);
        }
        if let Some(cwd) = &self.session.working_dir {
            event = event.cwd(cwd);
        }
        if let Some(text) = records["records"].as_array().and_then(|rows| {
            rows.iter()
                .rev()
                .find(|row| row["kind"] == "assistant_claim")
                .and_then(|row| row["text"].as_str())
        }) {
            event = event.field(
                "LAST_ASSISTANT_TEXT",
                text.chars().take(4000).collect::<String>(),
            );
        }
        if let Some(error) = error {
            event = event.field("ERROR", error);
        }
        crate::hooks::dispatch_observer(event);
    }

    pub(super) fn clear_local_memory(&mut self) {
        self.local_turn_memory = None;
        if !self.is_remote {
            crate::memory::clear_pending_memory(&self.session.id);
        }
        self.last_injected_memory_signature = None;
    }

    /// Only durable human input begins a turn, never tool results or reminders.
    pub(super) fn begin_local_memory_turn(&mut self, message_id: &str) {
        self.clear_local_memory();
        self.session.model_usage_turn_id = Some(format!("{}:{message_id}", self.session.id));
    }

    /// Build split system prompt for better caching
    pub(super) fn build_system_prompt_split(
        &mut self,
        memory_prompt: Option<&str>,
    ) -> crate::prompt::SplitSystemPrompt {
        // Ambient mode: use the full override prompt directly
        if let Some(ref prompt) = self.ambient_system_prompt {
            return crate::prompt::SplitSystemPrompt {
                static_part: prompt.clone(),
                dynamic_part: String::new(),
            };
        }

        let skills = self.current_skills_snapshot();
        let skill_prompt = self
            .active_skill
            .as_ref()
            .and_then(|name| skills.get(name).map(|s| s.get_prompt().to_string()));
        let available_skills: Vec<crate::prompt::SkillInfo> = skills
            .list()
            .iter()
            .map(|s| crate::prompt::SkillInfo {
                name: s.name.clone(),
                description: s.description.clone(),
            })
            .collect();
        let (mut split, context_info) = crate::prompt::build_system_prompt_split(
            skill_prompt.as_deref(),
            &available_skills,
            self.session.is_canary,
            memory_prompt,
            None,
        );
        self.append_current_turn_system_reminder(&mut split);
        crate::prompt::append_swarm_effort_directive(
            &mut split,
            self.provider.reasoning_effort().as_deref(),
        );
        self.context_info = context_info;
        split
    }

    pub(in crate::tui::app) fn show_injected_memory_context(
        &mut self,
        prompt: &str,
        display_prompt: Option<&str>,
        count: usize,
        age_ms: u64,
        memory_ids: Vec<String>,
    ) {
        let count = count.max(1);
        let plural = if count == 1 { "memory" } else { "memories" };
        let display_prompt = if let Some(display_prompt) = display_prompt {
            display_prompt.to_string()
        } else if prompt.trim().is_empty() {
            "# Memory\n\n## Notes\n1. (empty injection payload)".to_string()
        } else {
            prompt.to_string()
        };
        if !self.should_inject_memory_context(prompt) {
            return;
        }
        crate::memory::record_injected_prompt(prompt, count, age_ms);
        let summary = if count == 1 {
            "🧠 auto-recalled 1 memory".to_string()
        } else {
            format!("🧠 auto-recalled {} memories", count)
        };
        // Record to session for replay visualization
        self.session.record_memory_injection(
            summary.clone(),
            display_prompt.clone(),
            count as u32,
            age_ms,
            memory_ids,
        );
        if let Err(err) = self.session.save() {
            crate::logging::warn(&format!(
                "Failed to persist memory injection for session {}: {}",
                self.session.id, err
            ));
        }
        self.push_display_message(DisplayMessage::memory(summary, display_prompt));
        let notice = if let Some(experimental_notice) =
            self.note_experimental_feature_use("memory_injection")
        {
            format!(
                "🧠 {} {} injected · ⚠ {}",
                count, plural, experimental_notice
            )
        } else {
            format!("🧠 {} {} injected", count, plural)
        };
        self.set_status_notice(notice);
    }

    /// Assemble the ephemeral request separately from the durable source transcript.
    pub(super) async fn prepare_local_memory_request(
        &mut self,
        messages: Vec<Message>,
    ) -> (Vec<Message>, crate::prompt::SplitSystemPrompt) {
        let split = self.build_system_prompt_split(None);
        let mut messages = if crate::config::config().features.message_timestamps {
            Message::with_timestamps(&messages)
        } else {
            messages
        };
        let turn_id = self.session.model_usage_turn_id.clone().filter(|turn_id| {
            self.session
                .visible_conversation_messages()
                .iter()
                .any(|message| {
                    message.role == Role::User
                        && format!("{}:{}", self.session.id, message.id) == *turn_id
                })
        });
        if self.is_remote || !self.memory_enabled || turn_id.is_none() {
            self.clear_local_memory();
            return (messages, split);
        }
        if self
            .local_turn_memory
            .as_ref()
            .is_some_and(|state| state.session_id != self.session.id)
        {
            self.local_turn_memory = None;
        }
        let state = self
            .local_turn_memory
            .get_or_insert_with(|| LocalTurnMemory {
                session_id: self.session.id.clone(),
                external: None,
            });
        // Read the original transcript, not a compacted summary or a tool-result tail.
        let raw: std::sync::Arc<[Message]> = self
            .session
            .visible_conversation_messages()
            .into_iter()
            .map(|m| m.to_message())
            .collect::<Vec<_>>()
            .into();
        let (native, external) = tokio::join!(
            crate::memory::await_turn_memory(
                &self.session.id,
                turn_id.as_deref().unwrap(),
                raw,
                self.session.working_dir.clone()
            ),
            crate::memory::prepare_external_turn_memory(&mut state.external, &self.session, true),
        );
        if let Some(pending) = native.memory {
            if native.first_delivery {
                self.show_injected_memory_context(
                    &pending.prompt,
                    pending.display_prompt.as_deref(),
                    pending.count,
                    pending.computed_at.elapsed().as_millis() as u64,
                    pending.memory_ids,
                );
            }
            let text = format!("<system-reminder>\n{}\n</system-reminder>", pending.prompt);
            if !messages.iter().any(|m| {
                m.role == Role::User
                    && matches!(m.content.as_slice(),
                [ContentBlock::Text { text: existing, .. }] if existing == &text)
            }) {
                messages.push(Message::user(&text));
            }
        }
        if let Some((message, _, _)) = external {
            messages.push(message);
        }
        (messages, split)
    }

    pub(super) fn extraction_transcript(&self) -> String {
        let messages: Vec<_> = self
            .session
            .messages
            .iter()
            .filter(|m| m.display_role.is_none())
            .map(|m| m.to_message())
            .collect();
        crate::memory_agent::build_transcript_for_extraction(&messages)
    }

    /// Extract raw session evidence, not the potentially compacted provider history.
    pub(super) async fn extract_session_memories(&self) {
        if self.is_remote {
            return;
        }
        crate::memory_agent::extract_and_store(
            &self.extraction_transcript(),
            &self.session.id,
            self.session.working_dir.as_deref(),
            Some(self.provider.fork()),
        )
        .await;
    }
}

#[cfg(test)]
#[path = "turn_memory_tests.rs"]
mod tests;
