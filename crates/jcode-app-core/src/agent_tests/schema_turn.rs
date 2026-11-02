//! Does a schema-constrained turn actually keep the schema attached for the
//! model call, and release it afterwards?
//!
//! The schema is provider state, not a request field, so the two ways it can go
//! wrong are invisible from the outside: it can be attached and never released
//! (constraining every later turn), or it can be released before the model call
//! (the turn silently falls back to prompt guidance). Both are asserted here
//! against a provider that records what it saw.

use super::*;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

fn schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["title"],
        "properties": { "title": { "type": "string" } }
    })
}

#[derive(Clone)]
struct SchemaRecordingProvider {
    supports_native: bool,
    fail_complete: bool,
    attached: Arc<std::sync::Mutex<bool>>,
    set_calls: Arc<AtomicUsize>,
    clear_calls: Arc<AtomicUsize>,
    /// What `attached` held at the moment the model call was made.
    attached_during_complete: Arc<std::sync::Mutex<Option<bool>>>,
}

impl SchemaRecordingProvider {
    fn new(supports_native: bool) -> Self {
        Self {
            supports_native,
            fail_complete: false,
            attached: Arc::new(std::sync::Mutex::new(false)),
            set_calls: Arc::new(AtomicUsize::new(0)),
            clear_calls: Arc::new(AtomicUsize::new(0)),
            attached_during_complete: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    fn failing() -> Self {
        Self {
            fail_complete: true,
            ..Self::new(true)
        }
    }

    fn set_calls(&self) -> usize {
        self.set_calls.load(AtomicOrdering::SeqCst)
    }

    fn clear_calls(&self) -> usize {
        self.clear_calls.load(AtomicOrdering::SeqCst)
    }

    fn attached_now(&self) -> bool {
        *self.attached.lock().unwrap()
    }

    fn attached_during_complete(&self) -> Option<bool> {
        *self.attached_during_complete.lock().unwrap()
    }
}

#[async_trait]
impl Provider for SchemaRecordingProvider {
    async fn complete(
        &self,
        _messages: &[Message],
        _tools: &[ToolDefinition],
        _system: &str,
        _resume_session_id: Option<&str>,
    ) -> Result<EventStream> {
        *self.attached_during_complete.lock().unwrap() = Some(self.attached_now());
        if self.fail_complete {
            return Err(anyhow::anyhow!("provider blew up mid-turn"));
        }
        let (tx, rx) = tokio_mpsc::channel::<Result<StreamEvent>>(8);
        tokio::spawn(async move {
            let _ = tx
                .send(Ok(StreamEvent::TextDelta("{\"title\":\"ok\"}".to_string())))
                .await;
            let _ = tx
                .send(Ok(StreamEvent::MessageEnd {
                    stop_reason: Some("end_turn".to_string()),
                }))
                .await;
        });
        Ok(Box::pin(ReceiverStream::new(rx)))
    }

    fn set_response_schema(&self, _schema: &serde_json::Value) -> bool {
        self.set_calls.fetch_add(1, AtomicOrdering::SeqCst);
        if !self.supports_native {
            return false;
        }
        *self.attached.lock().unwrap() = true;
        true
    }

    fn clear_response_schema(&self) {
        self.clear_calls.fetch_add(1, AtomicOrdering::SeqCst);
        *self.attached.lock().unwrap() = false;
    }

    fn name(&self) -> &str {
        "schema-recording"
    }

    fn fork(&self) -> Arc<dyn Provider> {
        Arc::new(self.clone())
    }
}

async fn run_turn(
    provider: SchemaRecordingProvider,
    response_schema: Option<serde_json::Value>,
) -> anyhow::Result<()> {
    let provider: Arc<dyn Provider> = Arc::new(provider);
    let registry = Registry::new(Arc::clone(&provider)).await;
    let mut agent = Agent::new(provider, registry);
    let lease = agent.claim_server_turn().unwrap();
    let (tx, _rx) = tokio_mpsc::unbounded_channel();
    agent
        .run_reserved_streaming_mpsc_with_schema(
            &lease,
            "Return JSON",
            Vec::new(),
            None,
            tx,
            None,
            response_schema,
        )
        .await
}

#[tokio::test]
async fn the_schema_is_attached_for_the_model_call_and_released_afterwards() {
    let _guard = crate::storage::lock_test_env();
    let provider = SchemaRecordingProvider::new(true);
    run_turn(provider.clone(), Some(schema()))
        .await
        .expect("turn completes");

    assert_eq!(provider.set_calls(), 1);
    assert_eq!(
        provider.attached_during_complete(),
        Some(true),
        "the provider must see the schema while it builds the request, or the turn is prompt-only"
    );
    assert_eq!(provider.clear_calls(), 1);
    assert!(
        !provider.attached_now(),
        "a schema left attached would constrain every later turn"
    );
}

#[tokio::test]
async fn a_failed_turn_still_releases_the_schema() {
    let _guard = crate::storage::lock_test_env();
    let provider = SchemaRecordingProvider::failing();
    run_turn(provider.clone(), Some(schema()))
        .await
        .expect_err("provider failure surfaces");

    assert_eq!(provider.set_calls(), 1);
    assert_eq!(
        provider.clear_calls(),
        1,
        "the release must not be skipped just because the turn failed"
    );
    assert!(!provider.attached_now());
}

#[tokio::test]
async fn a_provider_without_native_schemas_is_never_asked_twice_and_never_cleared() {
    let _guard = crate::storage::lock_test_env();
    let provider = SchemaRecordingProvider::new(false);
    run_turn(provider.clone(), Some(schema()))
        .await
        .expect("turn completes on prompt guidance");

    assert_eq!(provider.set_calls(), 1, "the caller asks once per turn");
    assert_eq!(
        provider.clear_calls(),
        0,
        "nothing was attached, so there is nothing to release"
    );
    assert_eq!(provider.attached_during_complete(), Some(false));
}
