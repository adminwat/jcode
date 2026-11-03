use jcode_message_types::{ContentBlock, Message, Role};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

/// Default token budget (200k tokens - matches Claude's actual context limit)
pub const DEFAULT_TOKEN_BUDGET: usize = 200_000;

/// Trigger compaction at this percentage of budget
pub const COMPACTION_THRESHOLD: f32 = 0.80;

/// If context is above this threshold when compaction starts, do a synchronous
/// hard-compact (drop old messages) so the API call doesn't fail.
pub const CRITICAL_THRESHOLD: f32 = 0.95;

/// Minimum threshold for manual compaction (can compact at any time above this)
pub const MANUAL_COMPACT_MIN_THRESHOLD: f32 = 0.10;

/// Keep this many recent turns verbatim (not summarized)
pub const RECENT_TURNS_TO_KEEP: usize = 10;

/// Absolute minimum turns to keep during emergency compaction
pub const MIN_TURNS_TO_KEEP: usize = 2;

/// Max chars for a single tool result during emergency truncation
pub const EMERGENCY_TOOL_RESULT_MAX_CHARS: usize = 4000;

/// Max chars to keep for an inline image payload during emergency recovery.
/// Images are usually base64 screenshots; at hard-threshold time the useful
/// state should be represented by nearby tool text/summary, not by replaying a
/// huge raw image in the recent tail.
pub const EMERGENCY_IMAGE_MAX_CHARS: usize = 1024;

/// Approximate maximum request body size (in base64 characters) we aim to keep
/// the transcript under when recovering from a provider "request too large" /
/// 413 payload error. Anthropic rejects requests whose serialized body exceeds
/// roughly 32 MB; this distinct failure mode is driven almost entirely by inline
/// base64 images, which the normal token-budget accounting deliberately
/// undercounts (see `IMAGE_TOKEN_COST`). We target a conservative budget well
/// under the hard provider cap so a single retry reliably fits.
pub const PAYLOAD_IMAGE_CHAR_BUDGET: usize = 12 * 1024 * 1024;

/// Approximate chars per token for estimation
pub const CHARS_PER_TOKEN: usize = 4;

/// Approximate token cost charged for a single inline image.
///
/// Image content blocks carry base64-encoded payloads that are often hundreds
/// of kilobytes. Counting that raw base64 length as message text (len / 4)
/// massively overestimates the real context cost: providers tokenize images by
/// resolution, not by transport-encoded byte length, and a typical screenshot
/// costs on the order of ~1-2k tokens regardless of base64 size. Using the raw
/// length caused the token estimate to balloon far above the real
/// provider-observed input, spuriously tripping the compaction threshold and
/// driving repeated back-to-back ("triple") compactions that could not bring
/// the estimate down because the images stayed in the recent kept turns.
///
/// We charge a flat, slightly conservative per-image token budget instead.
pub const IMAGE_TOKEN_COST: usize = 1_600;

/// Fixed token overhead for system prompt + tool definitions.
/// These are not counted in message content but do count toward the context limit.
/// Estimated conservatively: ~8k tokens for system prompt + ~10k for 50+ tools.
pub const SYSTEM_OVERHEAD_TOKENS: usize = 18_000;

/// Rolling window size for token history (proactive/semantic modes)
pub const TOKEN_HISTORY_WINDOW: usize = 20;

/// Maximum characters to embed per message (first N chars capture semantic content)
pub const EMBED_MAX_CHARS_PER_MSG: usize = 512;

/// Rolling window of per-turn embeddings used for topic-shift detection
pub const EMBEDDING_HISTORY_WINDOW: usize = 10;

/// Per-manager semantic embedding cache capacity.
pub const SEMANTIC_EMBED_CACHE_CAPACITY: usize = 256;

pub const SUMMARY_PROMPT: &str = r#"Summarize our conversation so you can continue this work later.

Write in natural language with these sections:
- **Context:** What we're working on and why (1-2 sentences)
- **What we did:** Key actions taken, files changed, problems solved
- **Current state:** What works, what's broken, what's next
- **User preferences:** Specific requirements or decisions they made

Be concise but preserve important details. You can search the full conversation later if you need exact error messages or code snippets."#;

/// A completed summary covering turns up to a certain point
#[derive(Debug, Clone)]
pub struct Summary {
    pub text: String,
    pub openai_encrypted_content: Option<String>,
    pub covers_up_to_turn: usize,
    pub original_turn_count: usize,
}

/// Event emitted when compaction is applied
#[derive(Debug, Clone)]
pub struct CompactionEvent {
    pub trigger: String,
    pub pre_tokens: Option<u64>,
    pub post_tokens: Option<u64>,
    pub tokens_saved: Option<u64>,
    pub duration_ms: Option<u64>,
    pub messages_dropped: Option<usize>,
    pub messages_compacted: Option<usize>,
    pub summary_chars: Option<usize>,
    pub active_messages: Option<usize>,
}

/// What happened when ensure_context_fits was called
#[derive(Debug, Clone, PartialEq)]
pub enum CompactionAction {
    /// Nothing needed, context is fine.
    None,
    /// Background summarization started.
    BackgroundStarted { trigger: String },
    /// Emergency hard compact performed. Contains number of messages dropped.
    HardCompacted(usize),
}

/// Stats about compaction state
#[derive(Debug, Clone)]
pub struct CompactionStats {
    pub total_turns: usize,
    pub active_messages: usize,
    pub has_summary: bool,
    pub is_compacting: bool,
    pub token_estimate: usize,
    pub effective_tokens: usize,
    pub observed_input_tokens: Option<u64>,
    pub context_usage: f32,
}

pub fn compacted_summary_text_block(summary: &str) -> String {
    format!("## Previous Conversation Summary\n\n{}\n\n---\n\n", summary)
}

pub fn build_compaction_prompt(
    messages: &[Message],
    existing_summary: Option<&Summary>,
    max_prompt_chars: usize,
) -> String {
    let mut conversation_text = build_compaction_conversation_text(messages, existing_summary);
    let overhead = SUMMARY_PROMPT.len() + 50;
    if conversation_text.len() + overhead > max_prompt_chars && max_prompt_chars > overhead {
        let budget = max_prompt_chars - overhead;
        conversation_text = truncate_str_boundary(&conversation_text, budget).to_string();
        conversation_text
            .push_str("\n\n... [earlier conversation truncated to fit context window]\n");
    }
    format!("{}\n\n---\n\n{}", conversation_text, SUMMARY_PROMPT)
}

pub fn build_compaction_conversation_text(
    messages: &[Message],
    existing_summary: Option<&Summary>,
) -> String {
    let mut conversation_text = String::new();
    if let Some(summary) = existing_summary {
        conversation_text.push_str("## Previous Summary\n\n");
        conversation_text.push_str(&summary.text);
        conversation_text.push_str("\n\n## New Conversation\n\n");
    }

    for msg in messages {
        let role_str = match msg.role {
            Role::User => "User",
            Role::Assistant => "Assistant",
        };
        conversation_text.push_str(&format!("**{}:**\n", role_str));
        for block in &msg.content {
            match block {
                ContentBlock::Text { text, .. } => {
                    conversation_text.push_str(text);
                    conversation_text.push('\n');
                }
                ContentBlock::ToolUse { name, input, .. } => {
                    conversation_text.push_str(&format!("[Tool: {} - {}]\n", name, input));
                }
                ContentBlock::ToolResult { content, .. } => {
                    let truncated = if content.len() > 500 {
                        format!("{}... (truncated)", truncate_str_boundary(content, 500))
                    } else {
                        content.clone()
                    };
                    conversation_text.push_str(&format!("[Result: {}]\n", truncated));
                }
                ContentBlock::Reasoning { .. }
                | ContentBlock::ReasoningTrace { .. }
                | ContentBlock::AnthropicThinking { .. }
                | ContentBlock::OpenAIReasoning { .. } => {}
                ContentBlock::Image { .. } => conversation_text.push_str("[Image]\n"),
                ContentBlock::OpenAICompaction { .. } => {
                    conversation_text.push_str("[OpenAI native compaction]\n")
                }
            }
        }
        conversation_text.push('\n');
    }
    conversation_text
}

pub fn truncate_str_boundary(value: &str, max_bytes: usize) -> &str {
    if value.len() <= max_bytes {
        return value;
    }
    let mut end = max_bytes.min(value.len());
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

pub fn mean_embedding(embeddings: &[&Vec<f32>], dim: usize) -> Vec<f32> {
    let mut mean = vec![0f32; dim];
    for emb in embeddings {
        for (i, v) in emb.iter().enumerate() {
            if i < dim {
                mean[i] += v;
            }
        }
    }
    let n = embeddings.len().max(1) as f32;
    for v in &mut mean {
        *v /= n;
    }
    let norm: f32 = mean.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut mean {
            *v /= norm;
        }
    }
    mean
}

/// Find a safe compaction cutoff that does not leave kept tool results without
/// their corresponding tool calls.
pub fn safe_compaction_cutoff(messages: &[Message], initial_cutoff: usize) -> usize {
    let mut cutoff = initial_cutoff.min(messages.len());

    // Track tool call/result ids in the kept portion.
    let mut available_tool_ids = HashSet::new();
    let mut missing_tool_ids = HashSet::new();

    for msg in &messages[cutoff..] {
        for block in &msg.content {
            match block {
                ContentBlock::ToolUse { id, .. } => {
                    available_tool_ids.insert(id.clone());
                    missing_tool_ids.remove(id);
                }
                ContentBlock::ToolResult { tool_use_id, .. }
                    if !available_tool_ids.contains(tool_use_id) =>
                {
                    missing_tool_ids.insert(tool_use_id.clone());
                }
                _ => {}
            }
        }
    }

    if missing_tool_ids.is_empty() {
        return cutoff;
    }

    // Walk backward once, progressively growing the kept suffix until every
    // kept tool result has its matching tool use in the same suffix.
    for (idx, msg) in messages[..cutoff].iter().enumerate().rev() {
        for block in &msg.content {
            match block {
                ContentBlock::ToolUse { id, .. } => {
                    available_tool_ids.insert(id.clone());
                    missing_tool_ids.remove(id);
                }
                ContentBlock::ToolResult { tool_use_id, .. }
                    if !available_tool_ids.contains(tool_use_id) =>
                {
                    missing_tool_ids.insert(tool_use_id.clone());
                }
                _ => {}
            }
        }
        if missing_tool_ids.is_empty() {
            cutoff = idx;
            return cutoff;
        }
    }

    // If we couldn't find every matching tool call, don't compact at all.
    0
}

pub fn message_char_count(msg: &Message) -> usize {
    content_char_count(&msg.content)
}

pub fn content_char_count(content: &[ContentBlock]) -> usize {
    content
        .iter()
        .map(|block| match block {
            ContentBlock::Text { text, .. } => text.len(),
            ContentBlock::Reasoning { text } => text.len(),
            ContentBlock::ReasoningTrace { text } => text.len(),
            ContentBlock::AnthropicThinking {
                thinking,
                signature,
            } => thinking.len() + signature.len(),
            ContentBlock::OpenAIReasoning {
                id,
                summary,
                encrypted_content,
                status,
            } => {
                id.len()
                    + summary.iter().map(String::len).sum::<usize>()
                    + encrypted_content.as_ref().map(String::len).unwrap_or(0)
                    + status.as_ref().map(String::len).unwrap_or(0)
            }
            ContentBlock::ToolUse { input, .. } => input.to_string().len() + 50,
            ContentBlock::ToolResult { content, .. } => content.len() + 20,
            // Charge a flat token cost for images instead of the raw base64
            // payload length. See IMAGE_TOKEN_COST: counting base64 length here
            // overestimates context by ~100x and triggers spurious repeated
            // compactions.
            ContentBlock::Image { .. } => IMAGE_TOKEN_COST * CHARS_PER_TOKEN,
            ContentBlock::OpenAICompaction { encrypted_content } => encrypted_content.len(),
        })
        .sum()
}

pub fn summary_payload_char_count(summary: &Summary) -> usize {
    summary
        .openai_encrypted_content
        .as_ref()
        .map(|value| value.len())
        .unwrap_or_else(|| summary.text.len())
}

pub fn estimate_compaction_tokens(
    summary: Option<&Summary>,
    active_message_chars: usize,
    token_budget: usize,
) -> usize {
    let summary_chars = summary.map(summary_payload_char_count).unwrap_or(0);
    estimate_compaction_tokens_from_chars(summary_chars + active_message_chars, token_budget)
}

/// Best-effort context size (tokens) from a provider usage report.
///
/// Providers disagree on what `input_tokens` means:
/// - **Split accounting** (Anthropic-style): `input_tokens` is only the
///   *uncached* remainder; cache reads/writes are separate counters, so the
///   real context size is `input + cache_read + cache_creation`.
/// - **Subset accounting** (OpenAI-style): `input_tokens` (`prompt_tokens`)
///   already includes cached tokens; `cached_tokens` is a subset and must NOT
///   be added again.
///
/// This is the single source of truth for that heuristic. Both the sidebar
/// context figure and the compaction manager's observed-token feed must use it
/// so the two never disagree (issue #441). When in doubt, avoid over-counting
/// unless there is strong evidence of split accounting.
pub fn effective_context_tokens_from_usage(
    provider_name: &str,
    input_tokens: u64,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
) -> u64 {
    if input_tokens == 0 {
        return 0;
    }
    let cache_read = cache_read_input_tokens.unwrap_or(0);
    let cache_creation = cache_creation_input_tokens.unwrap_or(0);
    let provider_name = provider_name.to_lowercase();

    let split_cache_accounting = provider_name.contains("anthropic")
        || provider_name.contains("claude")
        || cache_creation > 0
        || cache_read > input_tokens;

    if split_cache_accounting {
        input_tokens
            .saturating_add(cache_read)
            .saturating_add(cache_creation)
    } else {
        input_tokens
    }
}

pub fn estimate_compaction_tokens_from_chars(total_chars: usize, token_budget: usize) -> usize {
    let msg_tokens = total_chars / CHARS_PER_TOKEN;
    // Add overhead for system prompt + tool definitions, which are not in the
    // message list but do count toward the context limit. Scale the overhead to
    // the budget so tests with tiny budgets aren't affected.
    let overhead = if token_budget >= DEFAULT_TOKEN_BUDGET / 2 {
        SYSTEM_OVERHEAD_TOKENS
    } else {
        0
    };
    msg_tokens + overhead
}

pub fn semantic_goal_text(messages: &[Message]) -> String {
    let mut text = String::new();
    for msg in messages {
        for block in &msg.content {
            match block {
                ContentBlock::Text {
                    text: block_text, ..
                } => push_semantic_excerpt(&mut text, block_text, 200),
                ContentBlock::ToolResult { content, .. } => {
                    push_semantic_excerpt(&mut text, content, 100)
                }
                _ => {}
            }
        }
    }
    text
}

pub fn semantic_message_text(msg: &Message) -> String {
    let mut text = String::new();
    for block in &msg.content {
        if let ContentBlock::Text {
            text: block_text, ..
        } = block
        {
            push_semantic_excerpt(&mut text, block_text, EMBED_MAX_CHARS_PER_MSG);
        }
    }
    text
}

pub fn push_semantic_excerpt(target: &mut String, source: &str, max_chars: usize) {
    if source.is_empty() {
        return;
    }
    if !target.is_empty() {
        target.push(' ');
    }
    target.extend(source.chars().take(max_chars));
}

pub fn semantic_cache_key(text: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

pub fn build_emergency_summary_text(
    existing_summary: Option<&str>,
    dropped_count: usize,
    pre_tokens: u64,
    token_budget: usize,
    dropped_messages: &[Message],
) -> String {
    let mut summary_parts: Vec<String> = Vec::new();

    if let Some(existing) = existing_summary
        && !existing.is_empty()
    {
        summary_parts.push(existing.to_string());
    }

    summary_parts.push(format!(
        "**[Emergency compaction]**: {} messages were dropped to recover from context overflow. \
         The conversation had ~{}k tokens which exceeded the {}k limit.",
        dropped_count,
        pre_tokens / 1000,
        token_budget / 1000,
    ));

    let mut file_mentions = Vec::new();
    let mut tool_names = HashSet::new();
    for msg in dropped_messages {
        collect_emergency_summary_hints(msg, &mut tool_names, &mut file_mentions);
    }

    if !tool_names.is_empty() {
        let mut tools: Vec<_> = tool_names.into_iter().collect();
        tools.sort();
        summary_parts.push(format!("Tools used: {}", tools.join(", ")));
    }

    file_mentions.sort();
    file_mentions.dedup();
    if !file_mentions.is_empty() {
        file_mentions.truncate(30);
        summary_parts.push(format!("Files referenced: {}", file_mentions.join(", ")));
    }

    summary_parts.join("\n\n")
}

fn collect_emergency_summary_hints(
    msg: &Message,
    tool_names: &mut HashSet<String>,
    file_mentions: &mut Vec<String>,
) {
    for block in &msg.content {
        match block {
            ContentBlock::ToolUse { name, .. } => {
                tool_names.insert(name.clone());
            }
            ContentBlock::Text { text, .. } => {
                extract_file_mentions(text, file_mentions);
            }
            _ => {}
        }
    }
}

pub fn extract_file_mentions(text: &str, file_mentions: &mut Vec<String>) {
    for word in text.split_whitespace() {
        if looks_like_file_reference(word) {
            let cleaned = clean_file_reference(word);
            if !cleaned.is_empty() {
                file_mentions.push(cleaned.to_string());
            }
        }
    }
}

pub fn looks_like_file_reference(word: &str) -> bool {
    (word.contains('/') || word.contains('.'))
        && word.len() > 3
        && word.len() < 120
        && !word.starts_with("http")
        && (word.contains(".rs")
            || word.contains(".ts")
            || word.contains(".py")
            || word.contains(".toml")
            || word.contains(".json")
            || word.starts_with("src/")
            || word.starts_with("./"))
}

pub fn clean_file_reference(word: &str) -> &str {
    word.trim_matches(|c: char| {
        !c.is_alphanumeric() && c != '/' && c != '.' && c != '_' && c != '-'
    })
}

pub fn emergency_truncate_tool_results(messages: &mut [Message], max_chars: usize) -> usize {
    let mut truncated = 0;

    for msg in messages.iter_mut() {
        for block in msg.content.iter_mut() {
            match block {
                ContentBlock::ToolResult { content, .. } if content.len() > max_chars => {
                    *content = emergency_truncated_tool_result(content, max_chars);
                    truncated += 1;
                }
                _ => {}
            }
        }
    }

    truncated
}

pub fn emergency_truncate_large_payloads(
    messages: &mut [Message],
    max_tool_result_chars: usize,
    max_image_chars: usize,
) -> usize {
    let mut truncated = 0;

    for msg in messages.iter_mut() {
        for block in msg.content.iter_mut() {
            match block {
                ContentBlock::ToolResult { content, .. }
                    if content.len() > max_tool_result_chars =>
                {
                    *content = emergency_truncated_tool_result(content, max_tool_result_chars);
                    truncated += 1;
                }
                ContentBlock::Image { media_type, data } if data.len() > max_image_chars => {
                    let original_len = data.len();
                    let media_type = media_type.clone();
                    *block = ContentBlock::Text {
                        text: format!(
                            "[Image omitted during emergency context recovery: media_type={media_type}, original_base64_chars={original_len}. Rely on adjacent browser/tool text, screenshots saved to disk, or re-open/re-screenshot if visual details are needed.]"
                        ),
                        cache_control: None,
                    };
                    truncated += 1;
                }
                _ => {}
            }
        }
    }

    truncated
}

/// Files whose tool results are never folded, matched on any string in the
/// originating `ToolUse` input. These files already reach the model through
/// the system prompt every request; this keeps an explicit read of them
/// verbatim in history as well.
pub const FOLD_PROTECTED_FILES: [&str; 2] = ["prompt-overlay.md", "RUNTIME-CONTRACT.md"];

/// Max chars kept from the head, signal and tail lines of a folded result.
const FOLD_LINE_CHARS: usize = 120;

/// Fold tool results older than the last `keep_recent` messages into one-line
/// snapshots in the request view. Only `ToolResult` text is rewritten: user and
/// assistant text, tool calls, reasoning, images and the system prompt (which
/// carries the prompt overlays) are outside this function's reach by type.
/// Error results and results of calls naming a `FOLD_PROTECTED_FILES` entry
/// stay verbatim. Deterministic, so a folded prefix is byte-stable across
/// turns and stays cache-warm. Returns the number of folded blocks.
pub fn fold_old_tool_results(
    messages: &mut [Message],
    keep_recent: usize,
    max_chars: usize,
) -> usize {
    if max_chars == 0 || messages.len() <= keep_recent {
        return 0;
    }
    let protected: HashSet<String> = messages
        .iter()
        .flat_map(|msg| msg.content.iter())
        .filter_map(|block| match block {
            ContentBlock::ToolUse { id, input, .. }
                if FOLD_PROTECTED_FILES
                    .iter()
                    .any(|file| input.to_string().contains(file)) =>
            {
                Some(id.clone())
            }
            _ => None,
        })
        .collect();

    // Pin the boundary to the start of a user turn. A boundary that slides one
    // message per call folds a result the provider cached verbatim on the
    // previous call, invalidating the whole prefix after it every turn
    // (measured: 70k cache_write per call). Turn starts move only when the
    // user speaks, so between those points the folded view is byte-stable.
    let fold_end = messages[..messages.len() - keep_recent]
        .iter()
        .rposition(|msg| {
            msg.role == Role::User
                && msg
                    .content
                    .iter()
                    .all(|block| !matches!(block, ContentBlock::ToolResult { .. }))
        })
        .unwrap_or(0);
    let mut folded = 0;
    for msg in &mut messages[..fold_end] {
        for block in &mut msg.content {
            if let ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } = block
                && !matches!(is_error, Some(true))
                && content.len() > max_chars
                && !protected.contains(tool_use_id)
            {
                *content = fold_tool_result(content);
                folded += 1;
            }
        }
    }
    folded
}

/// One-line snapshot: size, first line, first error/exit/failure line, last line.
fn fold_tool_result(content: &str) -> String {
    let lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
    let clip = |line: &str| -> String {
        let line = line.trim();
        if line.chars().count() > FOLD_LINE_CHARS {
            format!(
                "{}…",
                line.chars().take(FOLD_LINE_CHARS).collect::<String>()
            )
        } else {
            line.to_string()
        }
    };
    let head = lines.first().map(|l| clip(l)).unwrap_or_default();
    let tail = lines.last().filter(|_| lines.len() > 1).map(|l| clip(l));
    let signal = lines
        .iter()
        .find(|l| {
            let low = l.to_ascii_lowercase();
            low.contains("exit code")
                || low.contains("error")
                || low.contains("panicked")
                || low.contains("failed")
        })
        .map(|l| clip(l))
        .filter(|s| *s != head && tail.as_ref() != Some(s));
    let mut out = format!(
        "[folded tool result: {} chars, {} lines] {}",
        content.len(),
        lines.len(),
        head
    );
    if let Some(signal) = signal {
        out.push_str(" | ");
        out.push_str(&signal);
    }
    if let Some(tail) = tail {
        out.push_str(" … ");
        out.push_str(&tail);
    }
    out
}

/// Whether a provider error indicates the *serialized request body* was too
/// large (HTTP 413), as distinct from exceeding the model's token context
/// window. Anthropic surfaces this as `request_too_large` / "Request exceeds the
/// maximum size" / "413 Payload Too Large"; OpenAI and gateways use similar
/// wording. This failure mode is dominated by inline base64 images, which the
/// token-budget accounting deliberately undercounts, so it needs a dedicated
/// byte-size recovery rather than ordinary context compaction.
pub fn is_request_payload_too_large_error(error: &str) -> bool {
    let lower = error.to_ascii_lowercase();
    lower.contains("request_too_large")
        || lower.contains("request too large")
        || lower.contains("payload too large")
        || lower.contains("request entity too large")
        || lower.contains("request exceeds the maximum size")
        || lower.contains("exceeds the maximum size")
        || contains_independent_status_code(&lower, "413")
}

/// Whether `haystack` contains `code` as a standalone status code rather than as
/// a fragment of a longer number (so "413" matches but "4130"/"version 4131"
/// does not). Mirrors the failover classifier's guard.
fn contains_independent_status_code(haystack: &str, code: &str) -> bool {
    let bytes = haystack.as_bytes();
    haystack.match_indices(code).any(|(start, _)| {
        let before_ok = start == 0 || !bytes[start - 1].is_ascii_digit();
        let end = start + code.len();
        let after_ok = end == bytes.len() || !bytes[end].is_ascii_digit();
        before_ok && after_ok
    })
}

/// Strip oversized inline images from `messages`, oldest-first, until the total
/// remaining base64 image payload fits within `target_total_chars`.
///
/// Unlike [`emergency_truncate_large_payloads`] (which is driven by the token
/// budget and replaces *every* image past a tiny per-image cap), this is the
/// byte-size recovery path for HTTP 413 "request too large" errors: it preserves
/// as many of the most recent images as the request size budget allows and only
/// drops the older ones. Each stripped image is replaced with a text marker so
/// the model still knows an image existed and where to recover it from.
///
/// Returns the number of images that were replaced with text markers.
pub fn emergency_strip_large_images(messages: &mut [Message], target_total_chars: usize) -> usize {
    let mut contents: Vec<&mut Vec<ContentBlock>> =
        messages.iter_mut().map(|m| &mut m.content).collect();
    strip_large_images_in_contents(&mut contents, target_total_chars)
}

/// Core of [`emergency_strip_large_images`], operating directly on a slice of
/// content-block vectors so it can be reused for both provider `Message`s and
/// the session's stored-message representation (which share `ContentBlock`).
pub fn strip_large_images_in_contents(
    contents: &mut [&mut Vec<ContentBlock>],
    target_total_chars: usize,
) -> usize {
    // Collect (content_index, block_index, payload_len) for every inline image,
    // in transcript order (oldest first).
    let mut images: Vec<(usize, usize, usize)> = Vec::new();
    let mut total: usize = 0;
    for (ci, content) in contents.iter().enumerate() {
        for (bi, block) in content.iter().enumerate() {
            if let ContentBlock::Image { data, .. } = block {
                images.push((ci, bi, data.len()));
                total = total.saturating_add(data.len());
            }
        }
    }

    if total <= target_total_chars {
        return 0;
    }

    let mut stripped = 0;
    // Drop oldest images first until we're under budget (always keep trying even
    // if a single huge recent image alone exceeds the budget — better to ship a
    // request the provider might still trim than to give up entirely).
    for (ci, bi, payload_len) in images {
        if total <= target_total_chars {
            break;
        }
        let block = &mut contents[ci][bi];
        if let ContentBlock::Image { media_type, data } = block {
            let original_len = data.len();
            let media_type = media_type.clone();
            *block = ContentBlock::Text {
                text: format!(
                    "[Image omitted during request-size recovery: media_type={media_type}, original_base64_chars={original_len}. The request body exceeded the provider size limit; older images were dropped. Rely on adjacent browser/tool text, screenshots saved to disk, or re-open/re-screenshot if visual details are needed.]"
                ),
                cache_control: None,
            };
            total = total.saturating_sub(payload_len);
            stripped += 1;
        }
    }

    stripped
}

pub fn emergency_truncated_tool_result(content: &str, max_chars: usize) -> String {
    let original_len = content.len();
    let keep_head = max_chars / 2;
    let keep_tail = max_chars / 4;
    let head = truncate_str_boundary(content, keep_head);
    let tail = tail_str_boundary(content, keep_tail);
    let truncated_len = original_len.saturating_sub(head.len() + tail.len());
    format!(
        "{}\n\n... [{} chars truncated for context recovery] ...\n\n{}",
        head, truncated_len, tail,
    )
}

pub fn tail_str_boundary(value: &str, max_bytes: usize) -> &str {
    if value.len() <= max_bytes {
        return value;
    }
    let mut start = value.len().saturating_sub(max_bytes);
    while start < value.len() && !value.is_char_boundary(start) {
        start += 1;
    }
    &value[start..]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_pair(id: &str, input: &str, result: &str, is_error: bool) -> [Message; 2] {
        [
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: id.to_string(),
                    name: "bash".to_string(),
                    input: serde_json::json!({ "command": input }),
                    thought_signature: None,
                }],
                timestamp: None,
                tool_duration_ms: None,
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: id.to_string(),
                    content: result.to_string(),
                    is_error: if is_error { Some(true) } else { None },
                }],
                timestamp: None,
                tool_duration_ms: None,
            },
        ]
    }

    fn result_text(msg: &Message) -> &str {
        match &msg.content[0] {
            ContentBlock::ToolResult { content, .. } => content,
            _ => panic!("not a tool result"),
        }
    }

    fn user_turn(text: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: text.to_string(),
                cache_control: None,
            }],
            timestamp: None,
            tool_duration_ms: None,
        }
    }

    #[test]
    fn fold_rewrites_only_old_large_ok_tool_results() {
        let big = (0..50)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let mut msgs: Vec<Message> = Vec::new();
        msgs.extend(tool_pair("old", "cargo build", &big, false)); // 0,1
        msgs.extend(tool_pair("err", "cargo test", &big, true)); // 2,3
        msgs.push(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Text {
                text: big.clone(),
                cache_control: None,
            }],
            timestamp: None,
            tool_duration_ms: None,
        }); // 4
        msgs.push(user_turn("next")); // 5: fold boundary
        msgs.extend(tool_pair("recent", "ls", &big, false)); // 6,7

        let folded = fold_old_tool_results(&mut msgs, 2, 100);
        assert_eq!(folded, 1);
        let snap = result_text(&msgs[1]);
        assert!(snap.starts_with("[folded tool result: "), "{snap}");
        assert!(
            snap.contains("line 0") && snap.contains("line 49"),
            "{snap}"
        );
        assert!(snap.len() < 200);
        assert_eq!(result_text(&msgs[3]), big, "error results stay verbatim");
        assert!(
            matches!(&msgs[4].content[0], ContentBlock::Text { text, .. } if *text == big),
            "assistant text is never folded"
        );
        assert_eq!(result_text(&msgs[7]), big, "recent results stay verbatim");
    }

    #[test]
    fn fold_keeps_protected_overlay_files_verbatim() {
        let big = "x".repeat(5_000);
        let mut msgs: Vec<Message> = Vec::new();
        for file in FOLD_PROTECTED_FILES {
            msgs.extend(tool_pair(file, &format!("cat .jcode/{file}"), &big, false));
        }
        msgs.extend(tool_pair("plain", "cat README.md", &big, false));
        msgs.push(user_turn("next"));
        msgs.extend(tool_pair("tail", "pwd", "/", false));

        let folded = fold_old_tool_results(&mut msgs, 2, 100);
        assert_eq!(folded, 1, "only the unprotected read folds");
        assert_eq!(result_text(&msgs[1]), big);
        assert_eq!(result_text(&msgs[3]), big);
        assert!(result_text(&msgs[5]).starts_with("[folded tool result: "));
    }

    #[test]
    fn fold_boundary_is_pinned_to_a_user_turn_while_the_tail_grows() {
        let big = "y".repeat(5_000);
        let mut msgs: Vec<Message> = tool_pair("a", "x", &big, false).into();
        msgs.push(user_turn("turn 2")); // 2
        msgs.extend(tool_pair("b", "x", &big, false)); // 3,4
        msgs.push(user_turn("turn 3")); // 5: open turn
        let keep = 3;
        // The open turn keeps producing tool calls. A sliding window would
        // fold one more result on every call; the pinned boundary moves once
        // (when "turn 3" is `keep` messages back) and the open turn's own
        // results never fold.
        let mut b_flips = 0;
        let mut b_folded = false;
        for i in 0..6 {
            msgs.extend(tool_pair(&format!("c{i}"), "x", &big, false));
            let mut view = msgs.clone();
            fold_old_tool_results(&mut view, keep, 100);
            let now = result_text(&view[4]).starts_with("[folded tool result: ");
            if now != b_folded {
                b_flips += 1;
                b_folded = now;
            }
            for j in 0..=i {
                assert_eq!(
                    result_text(&view[7 + 2 * j]),
                    big,
                    "c{j} verbatim on call {i}"
                );
            }
        }
        assert_eq!(
            b_flips, 1,
            "boundary moves once per turn, not once per call"
        );
        assert!(
            b_folded,
            "previous turn folds once the open turn is `keep` deep"
        );
    }

    #[test]
    fn fold_is_deterministic_and_disabled_at_zero() {
        let big = "a\nerror: boom\nb\n".repeat(100);
        let mut a: Vec<Message> = tool_pair("t", "x", &big, false).into();
        a.push(user_turn("next"));
        a.extend(tool_pair("tail", "pwd", "/", false));
        let mut b = a.clone();
        assert_eq!(fold_old_tool_results(&mut a, 2, 100), 1);
        assert_eq!(fold_old_tool_results(&mut b, 2, 100), 1);
        assert_eq!(result_text(&a[1]), result_text(&b[1]));
        assert!(result_text(&a[1]).contains("error: boom"));
        let mut c: Vec<Message> = tool_pair("t", "x", &big, false).into();
        c.push(user_turn("next"));
        c.extend(tool_pair("tail", "pwd", "/", false));
        assert_eq!(fold_old_tool_results(&mut c, 2, 0), 0);
        assert_eq!(result_text(&c[1]), big);
    }

    #[test]
    fn effective_context_split_accounting_adds_cache_counters() {
        // Anthropic-style: input is the uncached remainder.
        assert_eq!(
            effective_context_tokens_from_usage("anthropic", 10_000, Some(300_000), Some(5_000)),
            315_000
        );
        assert_eq!(
            effective_context_tokens_from_usage("Claude", 10_000, Some(300_000), None),
            310_000
        );
    }

    #[test]
    fn effective_context_subset_accounting_does_not_double_count() {
        // OpenAI-style: prompt_tokens already includes cached tokens.
        assert_eq!(
            effective_context_tokens_from_usage("openai", 400_000, Some(390_000), None),
            400_000
        );
        // No cache info at all: pass through.
        assert_eq!(
            effective_context_tokens_from_usage("opencode-go", 396_000, None, None),
            396_000
        );
    }

    #[test]
    fn effective_context_infers_split_accounting_from_counter_shape() {
        // cache_read > input implies input can't already contain it.
        assert_eq!(
            effective_context_tokens_from_usage("unknown", 10_000, Some(500_000), None),
            510_000
        );
        // Any cache_creation implies split accounting.
        assert_eq!(
            effective_context_tokens_from_usage("unknown", 10_000, Some(2_000), Some(1_000)),
            13_000
        );
    }

    #[test]
    fn effective_context_zero_input_reports_zero() {
        assert_eq!(
            effective_context_tokens_from_usage("anthropic", 0, Some(300_000), Some(5_000)),
            0
        );
    }

    #[test]
    fn builds_compaction_prompt_with_summary_and_truncated_tool_result() {
        let summary = Summary {
            text: "prior work".to_string(),
            openai_encrypted_content: None,
            covers_up_to_turn: 1,
            original_turn_count: 1,
        };
        let message = Message::user("hello");
        let prompt = build_compaction_prompt(&[message], Some(&summary), 10_000);
        assert!(prompt.contains("## Previous Summary"));
        assert!(prompt.contains("prior work"));
        assert!(prompt.contains("**User:**"));
        assert!(prompt.contains(SUMMARY_PROMPT));
    }

    #[test]
    fn truncates_on_utf8_boundary() {
        assert_eq!(truncate_str_boundary("éabc", 1), "");
        assert_eq!(truncate_str_boundary("éabc", 2), "é");
    }

    #[test]
    fn mean_embedding_is_normalized() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let mean = mean_embedding(&[&a, &b], 2);
        let norm = (mean[0] * mean[0] + mean[1] * mean[1]).sqrt();
        assert!((norm - 1.0).abs() < 0.0001);
    }

    #[test]
    fn safe_cutoff_keeps_tool_use_with_tool_result() {
        let tool_use = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: "read".to_string(),
                input: serde_json::json!({"file":"src/lib.rs"}),
                thought_signature: None,
            }],
            timestamp: None,
            tool_duration_ms: None,
        };
        let tool_result = Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "call_1".to_string(),
                content: "ok".to_string(),
                is_error: None,
            }],
            timestamp: None,
            tool_duration_ms: None,
        };
        let messages = vec![
            Message::user("old"),
            tool_use,
            tool_result,
            Message::user("new"),
        ];

        assert_eq!(safe_compaction_cutoff(&messages, 2), 1);
    }

    #[test]
    fn estimates_tokens_with_large_budget_overhead() {
        let summary = Summary {
            text: "abcd".repeat(100),
            openai_encrypted_content: None,
            covers_up_to_turn: 1,
            original_turn_count: 1,
        };

        assert_eq!(estimate_compaction_tokens(Some(&summary), 0, 1000), 100);
        assert_eq!(
            estimate_compaction_tokens(Some(&summary), 0, DEFAULT_TOKEN_BUDGET),
            100 + SYSTEM_OVERHEAD_TOKENS
        );
    }

    #[test]
    fn image_token_cost_is_bounded_not_base64_length() {
        // Regression: a large base64 image payload must not be counted as ~len/4
        // tokens. Doing so inflated the estimate ~100x and caused repeated
        // back-to-back compactions.
        let huge_base64 = "A".repeat(1_400_000);
        let mut image_msg = Message::user("");
        image_msg.content = vec![ContentBlock::Image {
            media_type: "image/png".to_string(),
            data: huge_base64.clone(),
        }];

        let chars = message_char_count(&image_msg);
        // Flat per-image cost in char-equivalents, far below the raw payload.
        assert_eq!(chars, IMAGE_TOKEN_COST * CHARS_PER_TOKEN);
        assert!(
            chars < huge_base64.len() / 10,
            "image should not be charged anywhere near its base64 length"
        );

        // The token estimate for four such images stays small.
        let tokens = estimate_compaction_tokens_from_chars(chars * 4, DEFAULT_TOKEN_BUDGET);
        assert!(
            tokens < SYSTEM_OVERHEAD_TOKENS + 4 * IMAGE_TOKEN_COST + 10,
            "four images should cost ~{} tokens, got {}",
            SYSTEM_OVERHEAD_TOKENS + 4 * IMAGE_TOKEN_COST,
            tokens
        );
    }

    #[test]
    fn builds_semantic_text_from_relevant_content() {
        let message = Message {
            role: Role::User,
            content: vec![
                ContentBlock::Text {
                    text: "hello world".to_string(),
                    cache_control: None,
                },
                ContentBlock::ToolResult {
                    tool_use_id: "call_1".to_string(),
                    content: "tool output".to_string(),
                    is_error: None,
                },
            ],
            timestamp: None,
            tool_duration_ms: None,
        };

        assert_eq!(semantic_message_text(&message), "hello world");
        assert_eq!(semantic_goal_text(&[message]), "hello world tool output");
        assert_eq!(semantic_cache_key("stable"), semantic_cache_key("stable"));
    }

    #[test]
    fn builds_emergency_summary_with_tools_and_files() {
        let messages = vec![
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "read".to_string(),
                    input: serde_json::json!({"file":"src/lib.rs"}),
                    thought_signature: None,
                }],
                timestamp: None,
                tool_duration_ms: None,
            },
            Message::user("Edited src/compaction.rs and Cargo.toml, ignored https://example.com"),
        ];

        let summary =
            build_emergency_summary_text(Some("previous"), 2, 201_000, 200_000, &messages);
        assert!(summary.contains("previous"));
        assert!(summary.contains("2 messages were dropped"));
        assert!(summary.contains("Tools used: read"));
        assert!(summary.contains("Files referenced: Cargo.toml, src/compaction.rs"));
        assert!(!summary.contains("https://example.com"));
    }

    #[test]
    fn emergency_truncation_is_utf8_safe() {
        let original = format!("{}middle{}", "é".repeat(20), "尾".repeat(20));
        let truncated = emergency_truncated_tool_result(&original, 25);
        assert!(truncated.contains("chars truncated for context recovery"));
        assert!(truncated.is_char_boundary(truncated.len()));
    }

    #[test]
    fn emergency_truncation_replaces_large_images_with_text_marker() {
        let mut messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Image {
                media_type: "image/png".to_string(),
                data: "a".repeat(2048),
            }],
            timestamp: None,
            tool_duration_ms: None,
        }];

        let truncated = emergency_truncate_large_payloads(&mut messages, 4000, 1024);
        assert_eq!(truncated, 1);
        match &messages[0].content[0] {
            ContentBlock::Text { text, .. } => {
                assert!(text.contains("Image omitted during emergency context recovery"));
                assert!(text.contains("original_base64_chars=2048"));
            }
            other => panic!("expected image to be replaced with text marker, got {other:?}"),
        }
    }

    #[test]
    fn detects_request_payload_too_large_errors() {
        assert!(is_request_payload_too_large_error(
            "Anthropic API error (413 Payload Too Large): {\"error\":{\"type\":\"request_too_large\",\"message\":\"Request exceeds the maximum size\"}}"
        ));
        assert!(is_request_payload_too_large_error(
            "413 Request Entity Too Large"
        ));
        assert!(is_request_payload_too_large_error("request too large"));
        // Not a payload error — should not match.
        assert!(!is_request_payload_too_large_error(
            "rate limit exceeded, retry after 20s"
        ));
        // Embedded digits must not trip the standalone 413 check.
        assert!(!is_request_payload_too_large_error(
            "model version 4130 is unavailable"
        ));
    }

    fn image_msg(data_len: usize) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::Image {
                media_type: "image/png".to_string(),
                data: "a".repeat(data_len),
            }],
            timestamp: None,
            tool_duration_ms: None,
        }
    }

    #[test]
    fn strip_large_images_drops_oldest_until_under_budget() {
        // Four 1000-char images = 4000 total; budget 2500 should drop the two
        // oldest (leaving 2000 <= 2500), keeping the two most recent.
        let mut messages = vec![
            image_msg(1000),
            image_msg(1000),
            image_msg(1000),
            image_msg(1000),
        ];

        let stripped = emergency_strip_large_images(&mut messages, 2500);
        assert_eq!(stripped, 2);
        // Oldest two replaced with text markers.
        assert!(matches!(messages[0].content[0], ContentBlock::Text { .. }));
        assert!(matches!(messages[1].content[0], ContentBlock::Text { .. }));
        // Most recent two preserved as images.
        assert!(matches!(messages[2].content[0], ContentBlock::Image { .. }));
        assert!(matches!(messages[3].content[0], ContentBlock::Image { .. }));
        if let ContentBlock::Text { text, .. } = &messages[0].content[0] {
            assert!(text.contains("Image omitted during request-size recovery"));
            assert!(text.contains("original_base64_chars=1000"));
        }
    }

    #[test]
    fn strip_large_images_noop_when_under_budget() {
        let mut messages = vec![image_msg(500), image_msg(500)];
        let stripped = emergency_strip_large_images(&mut messages, 4000);
        assert_eq!(stripped, 0);
        assert!(matches!(messages[0].content[0], ContentBlock::Image { .. }));
        assert!(matches!(messages[1].content[0], ContentBlock::Image { .. }));
    }

    #[test]
    fn strip_large_images_strips_all_when_single_image_exceeds_budget() {
        // Even a lone over-budget image is stripped (better than re-sending an
        // oversized request that the provider will reject again).
        let mut messages = vec![image_msg(8000)];
        let stripped = emergency_strip_large_images(&mut messages, 2000);
        assert_eq!(stripped, 1);
        assert!(matches!(messages[0].content[0], ContentBlock::Text { .. }));
    }
}
