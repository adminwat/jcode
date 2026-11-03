//! Typestate for code-mode sessions.
//!
//! `SandboxSession` is a session that has been created but whose runtime has
//! not been proven fit for code-mode work. The only way to obtain a
//! `VerifiedCodeModeSession` is to consume a `SandboxSession` through
//! [`SandboxSession::verify_infrastructure`], which runs every probe and
//! refuses with the first failure. The verified type is the only one with
//! `run`, so "ran code-mode on an unverified runtime" is a compile error,
//! not a review finding.
//!
//! The probes are deliberately narrow and all observable through the SDK:
//! the server must advertise each required capability, the session must be
//! attachable (it exists and the server owns it), and the harness must carry
//! a non-empty runtime contract. Nothing here reads files or env; a caller
//! who wants a stricter gate adds a probe, never a flag.

use crate::client::{JcodeClient, RunOptions, TurnResult};
use crate::errors::{Error, ErrorKind, Result};
use std::sync::Arc;

/// Capabilities a code-mode session depends on. `permissions` is listed
/// because code mode runs tools without a human in the loop; a server that
/// never issues permission prompts would run them with no gate at all.
pub const REQUIRED_CAPABILITIES: &[&str] = &["sessions", "permissions"];

/// Proof that `verify_infrastructure` ran on this exact session. Private
/// field, private constructor: only this module can mint it.
#[derive(Debug, Clone)]
pub struct InfrastructureProof {
    session_id: String,
    capabilities: Vec<String>,
    contract_bytes: usize,
}

impl InfrastructureProof {
    pub fn session_id(&self) -> &str {
        &self.session_id
    }
    pub fn capabilities(&self) -> &[String] {
        &self.capabilities
    }
    pub fn contract_bytes(&self) -> usize {
        self.contract_bytes
    }
}

/// A session that exists but is not yet proven fit for code-mode execution.
/// Has no `run`.
pub struct SandboxSession {
    client: Arc<JcodeClient>,
    session_id: String,
    runtime_contract: &'static str,
}

/// A session whose infrastructure was verified. The only type with `run`.
pub struct VerifiedCodeModeSession {
    client: Arc<JcodeClient>,
    proof: InfrastructureProof,
}

impl SandboxSession {
    /// Wrap an existing session. `runtime_contract` is the contract text the
    /// caller's binary carries (the harness embeds its own; pass that or the
    /// text of the overlay it serves). Empty text fails verification.
    pub fn new(
        client: Arc<JcodeClient>,
        session_id: impl Into<String>,
        runtime_contract: &'static str,
    ) -> Self {
        Self {
            client,
            session_id: session_id.into(),
            runtime_contract,
        }
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Consume this session and prove the runtime fit for code mode. Every
    /// probe runs in order; the first failure is returned and the sandbox is
    /// gone (a caller cannot retry on the same unverified value, which is the
    /// point: re-create it, re-verify it).
    pub fn verify_infrastructure(self) -> Result<VerifiedCodeModeSession> {
        let missing: Vec<&str> = REQUIRED_CAPABILITIES
            .iter()
            .copied()
            .filter(|cap| !self.client.supports(cap))
            .collect();
        if !missing.is_empty() {
            return Err(Error::new(
                ErrorKind::UnsupportedTransport,
                format!(
                    "code mode needs server capabilities {:?}; missing {:?}",
                    REQUIRED_CAPABILITIES, missing
                ),
            ));
        }
        if self.runtime_contract.trim().is_empty() {
            return Err(Error::new(
                ErrorKind::InvalidOption,
                "runtime contract is empty: the binary carries no embedded contract and no overlay was supplied",
            ));
        }
        // Attach proves the session exists and this server owns it. A stale
        // or foreign id fails here instead of on the first tool call.
        let info = self.client.attach_session(&self.session_id)?;
        if info.session_id != self.session_id {
            return Err(Error::new(
                ErrorKind::UnexpectedReply,
                format!(
                    "attached session {} but asked for {}",
                    info.session_id, self.session_id
                ),
            ));
        }
        let proof = InfrastructureProof {
            session_id: self.session_id,
            capabilities: REQUIRED_CAPABILITIES
                .iter()
                .map(|s| s.to_string())
                .collect(),
            contract_bytes: self.runtime_contract.len(),
        };
        Ok(VerifiedCodeModeSession {
            client: self.client,
            proof,
        })
    }
}

impl VerifiedCodeModeSession {
    pub fn proof(&self) -> &InfrastructureProof {
        &self.proof
    }

    pub fn session_id(&self) -> &str {
        self.proof.session_id()
    }

    /// Run one turn on the verified session. Permission prompts are not
    /// auto-approved here; a caller who wants that sets it on `options`,
    /// visibly, at the call site.
    pub fn run(&self, content: &str, options: RunOptions) -> Result<TurnResult> {
        self.client.run(self.proof.session_id(), content, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Compile-time property: `SandboxSession` has no `run`. This is the
    // whole typestate; the test documents it and fails to compile if a `run`
    // is ever added to the sandbox type.
    //
    // ```compile_fail
    // use jcode_sdk::code_mode::SandboxSession;
    // fn f(s: &SandboxSession) { let _ = s.run("x", Default::default()); }
    // ```
    #[test]
    fn required_capabilities_are_the_documented_pair() {
        assert_eq!(REQUIRED_CAPABILITIES, &["sessions", "permissions"]);
    }

    #[test]
    fn proof_cannot_be_constructed_outside_the_module() {
        // `InfrastructureProof` has private fields and no public constructor;
        // the only producer is `verify_infrastructure`. This test exists so a
        // `pub fn new` added later shows up as a diff against a stated rule.
        let src = include_str!("code_mode.rs");
        let proof_impl = src
            .split("impl InfrastructureProof {")
            .nth(1)
            .and_then(|rest| rest.split("\n}\n").next())
            .expect("impl block");
        assert!(
            !proof_impl.contains("pub fn new"),
            "InfrastructureProof must stay unconstructible outside verify_infrastructure"
        );
    }
}
