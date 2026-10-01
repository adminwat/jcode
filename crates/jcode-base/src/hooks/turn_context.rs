//! Bounded request/response hook. Unlike observers, its output is consumed.
use super::*;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const OUTPUT_LIMIT: usize = 16 * 1024;
const INPUT_LIMIT: usize = 64 * 1024;
const COMMAND_LIMIT: usize = 4;

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RetrievedMemory {
    pub source: String,
    pub id: String,
    pub text: String,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Response {
    version: u32,
    session_id: String,
    turn_id: String,
    memories: Vec<RetrievedMemory>,
}

// Own only the process group created for this hook, including any descendants.
// Drop also runs when the requesting turn is cancelled.
#[cfg(unix)]
struct HookProcessGroup(u32);

#[cfg(unix)]
impl Drop for HookProcessGroup {
    fn drop(&mut self) {
        unsafe {
            libc::kill(-(self.0 as i32), libc::SIGKILL);
        }
    }
}

pub async fn run_turn_context(
    session_id: &str,
    turn_id: &str,
    cwd: Option<&str>,
    query: &str,
) -> Vec<RetrievedMemory> {
    run_commands(
        &hook_commands("turn_context"),
        session_id,
        turn_id,
        cwd,
        query,
        Duration::from_millis(1500),
    )
    .await
}

async fn run_commands(
    commands: &[String],
    session_id: &str,
    turn_id: &str,
    cwd: Option<&str>,
    query: &str,
    timeout: Duration,
) -> Vec<RetrievedMemory> {
    if commands.is_empty() {
        return Vec::new();
    }
    if commands.len() > COMMAND_LIMIT
        || session_id.trim().is_empty()
        || turn_id.trim().is_empty()
        || query.trim().is_empty()
        || query.len() > INPUT_LIMIT
    {
        crate::logging::warn("turn_context status=invalid_request");
        return Vec::new();
    }
    let payload = serde_json::json!({"version": 1, "event": "turn_context",
        "session_id": session_id, "turn_id": turn_id, "cwd": cwd, "query": query})
    .to_string();
    if payload.len() > INPUT_LIMIT {
        crate::logging::warn("turn_context status=input_limit");
        return Vec::new();
    }
    let mut event = HookEvent::new("turn_context")
        .session_id(session_id)
        .field("TURN_ID", turn_id);
    if let Some(cwd) = cwd {
        event = event.cwd(cwd);
    }
    let deadline = tokio::time::Instant::now() + timeout;
    let results = futures::future::join_all(commands.iter().enumerate().map(|(index, command)| {
        run_command(
            index, command, &event, &payload, session_id, turn_id, deadline,
        )
    }))
    .await;
    let mut memories = Vec::new();
    let mut bytes = 0;
    for result in results {
        // Reject an oversized contribution rather than silently truncating evidence.
        let size = serde_json::to_vec(&result)
            .map(|v| v.len())
            .unwrap_or(OUTPUT_LIMIT + 1);
        if bytes + size > OUTPUT_LIMIT {
            crate::logging::warn("turn_context status=aggregate_limit");
            continue;
        }
        bytes += size;
        for memory in result {
            if !memories.contains(&memory) {
                memories.push(memory);
            }
        }
    }
    memories
}

async fn run_command(
    index: usize,
    command: &str,
    event: &HookEvent,
    payload: &str,
    session_id: &str,
    turn_id: &str,
    deadline: tokio::time::Instant,
) -> Vec<RetrievedMemory> {
    let start = std::time::Instant::now();
    let result: anyhow::Result<Vec<RetrievedMemory>> = async {
        let mut process = build_hook_process(command, event)?;
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            process.process_group(0);
        }
        // The complete request is stdin-only. Never export a truncated JSON query.
        let mut child = tokio::process::Command::from(process)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true)
            .spawn()?;
        #[cfg(unix)]
        let _group = HookProcessGroup(child.id().expect("newly spawned hook has a pid"));
        let mut stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
        let exchange = async {
            let write = async {
                stdin.write_all(payload.as_bytes()).await?;
                stdin.shutdown().await?;
                drop(stdin);
                Ok::<_, anyhow::Error>(())
            };
            let read = async {
                let mut bytes = Vec::new();
                stdout
                    .take((OUTPUT_LIMIT + 1) as u64)
                    .read_to_end(&mut bytes)
                    .await?;
                anyhow::ensure!(bytes.len() <= OUTPUT_LIMIT, "output_limit");
                Ok::<_, anyhow::Error>(bytes)
            };
            let (_, bytes) = tokio::try_join!(write, read)?;
            anyhow::ensure!(child.wait().await?.success(), "exit_failure");
            let response: Response =
                serde_json::from_slice(&bytes).map_err(|_| anyhow::anyhow!("invalid_json"))?;
            anyhow::ensure!(
                response.version == 1
                    && response.session_id == session_id
                    && response.turn_id == turn_id,
                "identity_mismatch"
            );
            anyhow::ensure!(
                response.memories.len() <= 32
                    && response.memories.iter().all(|m| !m.source.trim().is_empty()
                        && m.source.len() <= 128
                        && !m.id.trim().is_empty()
                        && m.id.len() <= 1024
                        && !m.text.trim().is_empty()
                        && m.text.len() <= 8192),
                "invalid_evidence"
            );
            Ok::<_, anyhow::Error>(response.memories)
        };
        tokio::time::timeout_at(deadline, exchange)
            .await
            .map_err(|_| anyhow::anyhow!("timeout"))?
    }
    .await;
    match result {
        Ok(memories) => {
            crate::logging::info(&format!(
                "turn_context hook={index} status=ok records={} latency_ms={}",
                memories.len(),
                start.elapsed().as_millis()
            ));
            memories
        }
        Err(error) => {
            crate::logging::warn(&format!(
                "turn_context hook={index} status=failed reason={error} latency_ms={}",
                start.elapsed().as_millis()
            ));
            Vec::new()
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    fn script(dir: &std::path::Path, name: &str, code: &str) -> String {
        let path = dir.join(name);
        std::fs::write(&path, format!("import json, os, sys, time\n{code}\n")).unwrap();
        format!(
            "/usr/bin/python3 {}",
            crate::terminal_launch::sh_escape(&path.to_string_lossy())
        )
    }

    const VALID: &str = r#"
p = json.load(sys.stdin)
assert p['event'] == 'turn_context'
assert p['query'] == 'current query ü'
assert os.environ['JCODE_HOOKS_DISABLED'] == '1'
assert os.environ['JCODE_HOOK_TURN_ID'] == p['turn_id']
assert os.getcwd() == p['cwd']
print(json.dumps({'version': 1, 'session_id': p['session_id'], 'turn_id': p['turn_id'],
 'memories': [{'source': 'fixture', 'id': 'record-1', 'text': 'Attributed evidence only.'}]}))
"#;

    #[tokio::test]
    async fn turn_context_delivers_attributed_current_identity_from_real_subprocess() {
        let dir = tempfile::tempdir().unwrap();
        let command = script(dir.path(), "valid.py", VALID);
        let result = run_commands(
            &[command],
            "session-full-id",
            "turn-1",
            dir.path().to_str(),
            "current query ü",
            Duration::from_millis(1500),
        )
        .await;
        assert_eq!(
            result,
            vec![RetrievedMemory {
                source: "fixture".into(),
                id: "record-1".into(),
                text: "Attributed evidence only.".into()
            }]
        );
    }

    #[tokio::test]
    async fn turn_context_rejects_wrong_identity_unattributed_and_useless_success() {
        let dir = tempfile::tempdir().unwrap();
        for response in [
            "{}",
            "[]",
            "OK",
            "",
            "{\"memories\": []}",
            r#"{"version":1,"session_id":"other","turn_id":"turn-1","memories":[{"source":"s","id":"i","text":"t"}]}"#,
            r#"{"version":1,"session_id":"session-full-id","turn_id":"old","memories":[{"source":"s","id":"i","text":"t"}]}"#,
            r#"{"version":1,"session_id":"session-full-id","turn_id":"turn-1","memories":[{"source":"","id":"i","text":"t"}]}"#,
            r#"{"version":1,"session_id":"session-full-id","turn_id":"turn-1","memories":[{"source":"s","text":"t"}]}"#,
            r#"{"version":2,"session_id":"session-full-id","turn_id":"turn-1","memories":[]}"#,
        ] {
            let code = format!(
                "sys.stdout.write({})",
                serde_json::to_string(response).unwrap()
            );
            let command = script(dir.path(), "bad.py", &code);
            assert!(
                run_commands(
                    &[command],
                    "session-full-id",
                    "turn-1",
                    None,
                    "query",
                    Duration::from_millis(1500)
                )
                .await
                .is_empty(),
                "{response}"
            );
        }
        let command = script(dir.path(), "nonzero.py", &format!("{VALID}\nsys.exit(2)"));
        assert!(
            run_commands(
                &[command],
                "session-full-id",
                "turn-1",
                dir.path().to_str(),
                "current query ü",
                Duration::from_millis(1500)
            )
            .await
            .is_empty()
        );
    }

    #[tokio::test]
    async fn turn_context_slow_and_oversized_hooks_do_not_hide_fast_results() {
        let dir = tempfile::tempdir().unwrap();
        let commands = vec![
            script(dir.path(), "slow.py", "time.sleep(30)"),
            script(
                dir.path(),
                "large.py",
                &format!("sys.stdout.write('x' * {})", OUTPUT_LIMIT + 1),
            ),
            script(dir.path(), "valid.py", VALID),
        ];
        let start = std::time::Instant::now();
        let result = run_commands(
            &commands,
            "session-full-id",
            "turn-1",
            dir.path().to_str(),
            "current query ü",
            Duration::from_millis(500),
        )
        .await;
        assert_eq!(result.len(), 1);
        assert!(
            start.elapsed() < Duration::from_millis(1500),
            "must not sum command timeouts"
        );
    }
}
