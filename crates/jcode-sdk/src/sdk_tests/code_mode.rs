//! `SandboxSession::verify_infrastructure` against a scripted transport.
//! No socket, no daemon: the transport answers `hello` with a chosen
//! capability list and `attach_session` with a chosen session, so each
//! probe can be driven to pass or fail on its own.

use crate::client::{ConnectOptions, JcodeClient, Transport};
use crate::code_mode::{REQUIRED_CAPABILITIES, SandboxSession};
use crate::errors::ErrorKind;
use jcode_harness_api::{API_VERSION_MAJOR, ClientFrame};
use std::io::{BufRead, BufReader, Read, Write};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Reads client frames off a pipe, writes scripted replies to another.
struct Scripted {
    capabilities: Vec<String>,
    attach_session_id: String,
}

struct Pipe(Arc<Mutex<Vec<u8>>>);

impl Write for Pipe {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// A reader that blocks until the scripted server has a reply for the
/// last request the client wrote. One request at a time is all the SDK
/// issues here (hello, then attach), so a simple turn-taking loop suffices.
struct Replies {
    requests: Arc<Mutex<Vec<u8>>>,
    consumed: usize,
    buffer: Vec<u8>,
    script: Scripted,
}

impl Read for Replies {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        loop {
            if !self.buffer.is_empty() {
                let n = out.len().min(self.buffer.len());
                out[..n].copy_from_slice(&self.buffer[..n]);
                self.buffer.drain(..n);
                return Ok(n);
            }
            let pending = {
                let all = self.requests.lock().unwrap();
                all[self.consumed..].to_vec()
            };
            if let Some(nl) = pending.iter().position(|b| *b == b'\n') {
                self.consumed += nl + 1;
                let frame: ClientFrame = serde_json::from_slice(&pending[..nl]).expect("client frame");
                let reply = self.reply(&frame);
                self.buffer = serde_json::to_vec(&reply).unwrap();
                self.buffer.push(b'\n');
                continue;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
    }
}

impl Replies {
    fn reply(&self, frame: &ClientFrame) -> serde_json::Value {
        let body = match serde_json::to_value(&frame.request).unwrap()["req"].as_str() {
            Some("hello") => serde_json::json!({
                "ev": "hello_ok", "version": API_VERSION_MAJOR, "server": "scripted/0",
                "capabilities": self.script.capabilities,
            }),
            Some("attach_session") => serde_json::json!({
                "ev": "attached",
                "session": { "session_id": self.script.attach_session_id, "status": "idle" },
            }),
            other => panic!("unscripted request {other:?}"),
        };
        let mut reply = body;
        reply["v"] = serde_json::json!(API_VERSION_MAJOR);
        reply["reply_to"] = serde_json::json!(frame.id);
        reply
    }
}

impl Transport for Scripted {
    fn split(self: Box<Self>) -> crate::errors::Result<(Box<dyn BufRead + Send>, Box<dyn Write + Send>)> {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let reader = Replies {
            requests: Arc::clone(&requests),
            consumed: 0,
            buffer: Vec::new(),
            script: *self,
        };
        Ok((Box::new(BufReader::new(reader)), Box::new(Pipe(requests))))
    }
}

fn client(capabilities: &[&str], attach_session_id: &str) -> Arc<JcodeClient> {
    let transport = Scripted {
        capabilities: capabilities.iter().map(|s| s.to_string()).collect(),
        attach_session_id: attach_session_id.to_string(),
    };
    let options = ConnectOptions {
        request_timeout: Some(Duration::from_secs(5)),
        ..Default::default()
    };
    Arc::new(JcodeClient::connect_with(Box::new(transport), options).expect("handshake"))
}

const CONTRACT: &str = "# contract\n\nrules";

#[test]
fn verified_session_requires_every_capability_a_live_contract_and_an_owned_session() {
    let c = client(REQUIRED_CAPABILITIES, "s1");
    let verified = SandboxSession::new(c, "s1", CONTRACT)
        .verify_infrastructure()
        .expect("all probes pass");
    assert_eq!(verified.session_id(), "s1");
    assert_eq!(verified.proof().contract_bytes(), CONTRACT.len());
    assert_eq!(
        verified.proof().capabilities(),
        REQUIRED_CAPABILITIES.iter().map(|s| s.to_string()).collect::<Vec<_>>()
    );
}

#[test]
fn missing_capability_refuses_before_touching_the_session() {
    let c = client(&["sessions"], "s1");
    let err = SandboxSession::new(c, "s1", CONTRACT)
        .verify_infrastructure()
        .err()
        .expect("refused");
    assert_eq!(err.kind, ErrorKind::UnsupportedTransport);
    assert!(err.to_string().contains("permissions"), "{err}");
}

#[test]
fn empty_contract_refuses() {
    let c = client(REQUIRED_CAPABILITIES, "s1");
    let err = SandboxSession::new(c, "s1", "  \n")
        .verify_infrastructure()
        .err()
        .expect("refused");
    assert_eq!(err.kind, ErrorKind::InvalidOption);
}

#[test]
fn foreign_session_refuses() {
    // The server attaches a different session than the one asked for.
    let c = client(REQUIRED_CAPABILITIES, "someone-else");
    let err = SandboxSession::new(c, "s1", CONTRACT)
        .verify_infrastructure()
        .err()
        .expect("refused");
    assert_eq!(err.kind, ErrorKind::UnexpectedReply);
}
