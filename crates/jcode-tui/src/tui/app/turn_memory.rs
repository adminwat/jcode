use super::*;

impl App {
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

    /// Get memory prompt using async non-blocking approach
    /// Takes any pending memory from background check and sends context to memory agent for next turn
    pub(in crate::tui::app) fn build_memory_prompt_nonblocking(
        &self,
        messages: &[Message],
    ) -> Option<crate::memory::PendingMemory> {
        if self.is_remote || !self.memory_enabled {
            return None;
        }

        // Take pending memory if available (computed in background during last turn)
        let fresh_user_turn = crate::message::ends_with_fresh_user_turn(messages);
        let pending = if fresh_user_turn {
            crate::memory::take_pending_memory(&self.session.id)
        } else {
            None
        };

        // Send context to memory agent for the NEXT turn (doesn't block current send)
        // Relevance results are consumed only at the start of a fresh user turn.
        // Tool continuations do not provide another injection opportunity, so
        // avoid re-running the local embedding model after every tool result.
        if fresh_user_turn {
            let shared_messages: std::sync::Arc<[crate::message::Message]> =
                messages.to_vec().into();
            crate::memory_agent::update_context_sync_with_dir(
                &self.session.id,
                shared_messages,
                self.session.working_dir.clone(),
            );
        }

        // Return pending memory from previous turn
        pending
    }

    pub(super) fn extraction_transcript(&self) -> String {
        let messages: Vec<_> = self.session.messages.iter().map(|m| m.to_message()).collect();
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
        ).await;
    }
}
