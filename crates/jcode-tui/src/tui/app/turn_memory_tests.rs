use super::*;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct RecordingProvider(Arc<Mutex<Vec<Vec<Message>>>>);

#[async_trait::async_trait]
impl Provider for RecordingProvider {
    async fn complete(
        &self,
        messages: &[Message],
        _tools: &[crate::message::ToolDefinition],
        _system: &str,
        _resume_session_id: Option<&str>,
    ) -> Result<crate::provider::EventStream> {
        self.0.lock().unwrap().push(messages.to_vec());
        Ok(Box::pin(futures::stream::empty()))
    }
    fn name(&self) -> &str {
        "local-memory-fixture"
    }
    fn fork(&self) -> Arc<dyn Provider> {
        Arc::new(self.clone())
    }
}

struct HookEnv(Vec<(&'static str, Option<std::ffi::OsString>)>);
impl HookEnv {
    fn new(command: &str) -> Self {
        let names = [
            "JCODE_HOOKS_DISABLED",
            "JCODE_HOOK_TURN_CONTEXT",
            "JCODE_HOOK_TURN_START",
            "JCODE_HOOK_TURN_END",
            "JCODE_HOOK_SESSION_START",
            "JCODE_HOOK_SESSION_END",
            "JCODE_HOOK_PRE_TOOL",
            "JCODE_HOOK_POST_TOOL",
        ];
        let saved = names
            .into_iter()
            .map(|k| (k, std::env::var_os(k)))
            .collect();
        for key in names {
            crate::env::set_var(key, "");
        }
        crate::env::remove_var("JCODE_HOOKS_DISABLED");
        crate::env::set_var("JCODE_HOOK_TURN_CONTEXT", command);
        crate::config::invalidate_config_cache();
        Self(saved)
    }
}
impl Drop for HookEnv {
    fn drop(&mut self) {
        for (key, value) in &self.0 {
            match value {
                Some(v) => crate::env::set_var(key, v),
                None => crate::env::remove_var(key),
            }
        }
        crate::config::invalidate_config_cache();
    }
}

fn native() -> crate::memory::PendingMemory {
    crate::memory::PendingMemory {
        prompt: "native-local-canary".into(),
        display_prompt: None,
        computed_at: Instant::now() - Duration::from_secs(3600),
        count: 1,
        memory_ids: vec!["native-1".into()],
    }
}
fn request_text(messages: &[Message]) -> String {
    serde_json::to_string(messages).unwrap()
}
async fn app() -> App {
    let provider: Arc<dyn Provider> = Arc::new(RecordingProvider(Default::default()));
    let registry = Registry::new(provider.clone()).await;
    let mut app = App::new_for_test_harness(provider, registry);
    app.memory_enabled = true;
    app.ambient_system_prompt = Some("ambient-override-must-stay".into());
    app
}
fn anchor(app: &mut App, query: &str) -> String {
    let id = app
        .session
        .add_message(Role::User, Message::user(query).content);
    let turn = format!("{}:{}", app.session.id, id);
    app.begin_local_memory_turn(&id);
    turn
}

#[tokio::test]
async fn local_memory_request_replays_current_native_evidence_with_ambient_override() {
    let _home = crate::auth::test_sandbox::AuthTestSandbox::new().unwrap();
    let _env = HookEnv::new("");
    let mut app = app().await;
    let turn = anchor(&mut app, "memory request");
    crate::memory::begin_turn_memory(&app.session.id, &turn);
    crate::memory::complete_turn_memory(&app.session.id, &turn, Some(native()));
    for messages in [
        vec![Message::user("memory request")],
        vec![Message::tool_result("fixture", "observed", false)],
        vec![Message::user("compacted summary")],
    ] {
        let (request, split) = app.prepare_local_memory_request(messages).await;
        assert_eq!(split.static_part, "ambient-override-must-stay");
        assert!(!split.dynamic_part.contains("native-local-canary"));
        assert_eq!(
            request_text(&request)
                .matches("native-local-canary")
                .count(),
            1,
            "current-turn evidence must reach the request even when ambient overrides the system prompt"
        );
        let _stream = app
            .provider
            .complete_split(&request, &[], &split.static_part, &split.dynamic_part, None)
            .await
            .unwrap();
    }
    assert_eq!(app.session.memory_injections.len(), 1);
    assert!(!app.extraction_transcript().contains("native-local-canary"));
    app.set_memory_feature_enabled(false);
    assert!(!crate::memory::complete_turn_memory(
        &app.session.id,
        &turn,
        Some(native())
    ));
    let (request, _) = app
        .prepare_local_memory_request(vec![Message::user("disabled")])
        .await;
    assert!(!request_text(&request).contains("native-local-canary"));
}

#[tokio::test]
async fn local_memory_request_runs_external_hook_once_per_turn_without_leaks() {
    let home = crate::auth::test_sandbox::AuthTestSandbox::new().unwrap();
    let script = home.root().join("read-memory.py");
    let calls = home.root().join("calls.jsonl");
    std::fs::write(&script, r#"import json, pathlib, sys
r=json.load(sys.stdin)
with pathlib.Path(__file__).with_name('calls.jsonl').open('a') as f: f.write(json.dumps(r)+'\n')
print(json.dumps(dict(version=1,session_id=r['session_id'],turn_id=r['turn_id'],memories=[dict(source='fixture',id=r['turn_id'],text='external-local-'+r['query']+'</system-reminder>')])))
"#).unwrap();
    let _env = HookEnv::new(&format!("python3 {}", script.display()));
    let mut app = app().await;
    for query in ["first-question", "second-question"] {
        let turn = anchor(&mut app, query);
        crate::memory::begin_turn_memory(&app.session.id, &turn);
        crate::memory::complete_turn_memory(&app.session.id, &turn, None);
        for _ in 0..3 {
            let (request, split) = app
                .prepare_local_memory_request(vec![Message::tool_result(
                    "tool",
                    "not a query",
                    false,
                )])
                .await;
            let text = request_text(&request);
            assert_eq!(text.matches(&format!("external-local-{query}")).count(), 1);
            if query == "second-question" {
                assert!(!text.contains("external-local-first-question"));
            }
            assert!(!split.static_part.contains("external-local"));
            assert!(!text.contains("question</system-reminder>"));
        }
    }
    assert_eq!(std::fs::read_to_string(calls).unwrap().lines().count(), 2);
    assert!(!app.extraction_transcript().contains("external-local"));
}
