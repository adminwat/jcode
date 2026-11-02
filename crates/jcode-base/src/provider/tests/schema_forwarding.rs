//! Response-schema forwarding through the provider facades.
//!
//! `Provider::set_response_schema` has a no-op default, so a facade that does
//! not forward it silently drops the schema: the turn asks for native
//! structured output, the request goes out unconstrained, and the assistant
//! answers with free text. These tests pin the forwarding chain
//! `JcodeProvider` -> `MultiProvider` -> active runtime.

use super::*;
use crate::provider::ActiveProvider;
use std::sync::atomic::{AtomicUsize, Ordering};

/// `MultiProvider::new` boots a runtime, so the test needs a Tokio reactor.
fn provider() -> MultiProvider {
    super::enter_test_runtime().block_on(async { MultiProvider::new() })
}

/// A runtime that records the schema it was handed and whether it was
/// cleared. It accepts only one schema so a test can prove the schema that
/// reached the runtime is the schema the caller set.
#[derive(Default)]
struct SchemaRecordingProvider {
    set_calls: AtomicUsize,
    clear_calls: AtomicUsize,
    schema: std::sync::Mutex<Option<serde_json::Value>>,
}

impl SchemaRecordingProvider {
    fn recorded(&self) -> Option<serde_json::Value> {
        self.schema.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl Provider for SchemaRecordingProvider {
    async fn complete(
        &self,
        _messages: &[Message],
        _tools: &[ToolDefinition],
        _system: &str,
        _resume_session_id: Option<&str>,
    ) -> Result<EventStream> {
        Err(anyhow!("the schema forwarding test never completes a call"))
    }

    fn name(&self) -> &str {
        "OpenAI"
    }

    fn model(&self) -> String {
        "gpt-5.4".into()
    }

    fn set_response_schema(&self, schema: &serde_json::Value) -> bool {
        self.set_calls.fetch_add(1, Ordering::SeqCst);
        *self.schema.lock().unwrap() = Some(schema.clone());
        true
    }

    fn clear_response_schema(&self) {
        self.clear_calls.fetch_add(1, Ordering::SeqCst);
        *self.schema.lock().unwrap() = None;
    }

    fn fork(&self) -> Arc<dyn Provider> {
        Arc::new(self.clone_arc())
    }
}

impl SchemaRecordingProvider {
    fn clone_arc(&self) -> SchemaRecordingProvider {
        SchemaRecordingProvider {
            set_calls: AtomicUsize::new(self.set_calls.load(Ordering::SeqCst)),
            clear_calls: AtomicUsize::new(self.clear_calls.load(Ordering::SeqCst)),
            schema: std::sync::Mutex::new(self.schema.lock().unwrap().clone()),
        }
    }
}

fn test_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "answer": { "type": "string" }
        },
        "required": ["answer"],
        "additionalProperties": false
    })
}

#[test]
fn multi_provider_forwards_response_schema_to_the_active_openai_runtime() {
    let provider = provider();
    let runtime = Arc::new(SchemaRecordingProvider::default());
    *provider.openai.write().unwrap() = Some(runtime.clone());
    provider.set_active_provider(ActiveProvider::OpenAI);

    assert!(
        provider.set_response_schema(&test_schema()),
        "the active OpenAI runtime accepts the schema, so the facade must report it"
    );

    assert_eq!(
        runtime.set_calls.load(Ordering::SeqCst),
        1,
        "the schema must reach the runtime that sends the request"
    );
    assert_eq!(
        runtime.recorded(),
        Some(test_schema()),
        "the runtime must hold the exact schema the caller set"
    );
}

#[test]
fn multi_provider_clears_the_response_schema_on_the_openai_runtime() {
    let provider = provider();
    let runtime = Arc::new(SchemaRecordingProvider::default());
    *provider.openai.write().unwrap() = Some(runtime.clone());
    provider.set_active_provider(ActiveProvider::OpenAI);

    provider.set_response_schema(&test_schema());
    provider.clear_response_schema();

    assert_eq!(runtime.clear_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        runtime.recorded(),
        None,
        "a released schema must not constrain the next unrelated turn"
    );
}

#[test]
fn multi_provider_releases_the_schema_even_after_a_failover_moved_the_active_runtime() {
    let provider = provider();
    let openai = Arc::new(SchemaRecordingProvider::default());
    *provider.openai.write().unwrap() = Some(openai.clone());
    provider.set_active_provider(ActiveProvider::OpenAI);
    provider.set_response_schema(&test_schema());

    // The turn failed over to another runtime before it ended.
    provider.set_active_provider(ActiveProvider::OpenRouter);
    provider.clear_response_schema();

    assert_eq!(
        openai.clear_calls.load(Ordering::SeqCst),
        1,
        "the runtime that started the turn still holds the schema and must release it"
    );
    assert_eq!(openai.recorded(), None);
}

#[test]
fn multi_provider_refuses_the_schema_when_no_runtime_can_enforce_it() {
    let provider = provider();
    provider.set_active_provider(ActiveProvider::OpenRouter);

    assert!(
        !provider.set_response_schema(&test_schema()),
        "a facade with no schema-capable runtime must refuse rather than pretend"
    );
}
