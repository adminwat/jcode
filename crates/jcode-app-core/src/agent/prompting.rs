use super::Agent;
use crate::logging;
use crate::message::{Message, ToolDefinition};

impl Agent {
    pub(super) fn log_prompt_prefix_accounting(
        &self,
        split: &crate::prompt::SplitSystemPrompt,
        tools: &[ToolDefinition],
    ) {
        let system_tokens = split.estimated_tokens();
        let tool_tokens = ToolDefinition::aggregate_prompt_token_estimate(tools);
        let prefix_tokens = system_tokens + tool_tokens;
        logging::info(&format!(
            "Prompt prefix estimate: total={} tokens (system={} tools={})",
            prefix_tokens, system_tokens, tool_tokens
        ));
    }

    pub(super) async fn build_turn_memory_prompt(
        &mut self,
        messages: std::sync::Arc<[Message]>,
    ) -> crate::memory::TurnMemoryResult {
        if !self.memory_enabled {
            return Default::default();
        }
        let turn_id = self.model_usage_turn_id();
        let session_id = &self.session.id;
        let started = crate::memory::begin_turn_memory(session_id, &turn_id);
        if started {
            crate::memory_agent::update_context_sync_with_dir(
                session_id,
                messages,
                self.session.working_dir.clone(),
            );
        }

        // Only the initial request waits. Slow retrieval can still reach a later
        // continuation, but can never be borrowed by the next user turn.
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(750);
        loop {
            let result = crate::memory::read_turn_memory(session_id, &turn_id);
            if result.ready || !started {
                return result;
            }
            if tokio::time::Instant::now() >= deadline {
                logging::info(&format!(
                    "MEMORY_DELIVERY_DEADLINE session={} turn={} wait_ms=750",
                    session_id, turn_id
                ));
                return result;
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
    }

    pub(super) fn prepare_turn_memory_injection(
        &mut self,
        result: &crate::memory::TurnMemoryResult,
        messages: &[Message],
    ) -> Option<(Message, bool)> {
        let memory = result.memory.as_ref()?;
        if result.first_delivery {
            let count = memory.count.max(1);
            let age_ms = memory.computed_at.elapsed().as_millis() as u64;
            crate::memory::record_injected_prompt(&memory.prompt, count, age_ms);
            crate::memory_log::log_pending_consumed(
                &self.session.id,
                count,
                age_ms,
                memory.prompt.len(),
            );
            self.record_memory_injection_in_session(memory);
        }
        let expected = format!("<system-reminder>\n{}\n</system-reminder>", memory.prompt);
        if messages.iter().any(|message| {
            message.role == crate::message::Role::User
                && matches!(message.content.as_slice(),
                    [crate::message::ContentBlock::Text { text, .. }] if text == &expected)
        }) {
            // Persisted memory is already present. If compaction removes it,
            // the next request will reinsert it instead of losing the context.
            return None;
        }
        Some(self.prepare_memory_injection_message(memory))
    }

    fn append_current_turn_system_reminder(&self, split: &mut crate::prompt::SplitSystemPrompt) {
        let Some(reminder) = self
            .current_turn_system_reminder
            .as_ref()
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
        else {
            return;
        };

        if !split.dynamic_part.is_empty() {
            split.dynamic_part.push_str("\n\n");
        }
        split.dynamic_part.push_str("# System Reminder\n\n");
        split.dynamic_part.push_str(reminder);
    }

    /// Build split system prompt for better caching
    /// Returns static (cacheable) and dynamic (not cached) parts separately
    pub(super) fn build_system_prompt_split(
        &self,
        memory_prompt: Option<&str>,
    ) -> crate::prompt::SplitSystemPrompt {
        if let Some(ref override_prompt) = self.system_prompt_override {
            return crate::prompt::SplitSystemPrompt {
                static_part: override_prompt.clone(),
                dynamic_part: String::new(),
            };
        }

        let skills = self.current_skills_snapshot();
        let skill_prompt = self
            .active_skill
            .as_ref()
            .and_then(|name| skills.get(name).map(|skill| skill.get_prompt().to_string()));

        let available_skills: Vec<crate::prompt::SkillInfo> = self
            .current_skills_snapshot()
            .list()
            .iter()
            .map(|skill| crate::prompt::SkillInfo {
                name: skill.name.clone(),
                description: skill.description.clone(),
            })
            .collect();

        let working_dir = self
            .session
            .working_dir
            .as_ref()
            .map(std::path::PathBuf::from);

        let (mut split, _context_info) = crate::prompt::build_system_prompt_split_with_agents_md(
            skill_prompt.as_deref(),
            &available_skills,
            self.session.is_canary,
            memory_prompt,
            working_dir.as_deref(),
            self.agents_md_snapshot.clone(),
        );

        self.append_current_turn_system_reminder(&mut split);
        crate::prompt::append_swarm_effort_directive(
            &mut split,
            self.provider.reasoning_effort().as_deref(),
        );

        split
    }

    /// Test wrapper around the same bounded delivery path used by both loops.
    #[cfg(test)]
    pub(super) async fn build_memory_prompt_nonblocking(
        &mut self,
        messages: &[Message],
        _memory_event_tx: Option<crate::memory::MemoryEventSink>,
    ) -> Option<crate::memory::PendingMemory> {
        self.build_turn_memory_prompt(messages.to_vec().into())
            .await
            .memory
    }
}
