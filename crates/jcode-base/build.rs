use std::env;
use std::fs;
use std::path::PathBuf;

/// Freeze the runtime contract into the binary. The live file is a symlink
/// outside the repo (`~/.jcode/prompt-overlay.md -> <config repo>/jcode/RUNTIME-CONTRACT.md`)
/// that can be deleted or left pointing at a drifted branch; the embedded copy
/// is the one the binary carries regardless. Source precedence at build time:
/// `JCODE_RUNTIME_CONTRACT` (explicit path), then `docs/RUNTIME-CONTRACT.md`
/// in the repo, else an empty snapshot (the loader reports that).
fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let repo_copy = manifest.join("../../docs/RUNTIME-CONTRACT.md");
    println!("cargo:rerun-if-env-changed=JCODE_RUNTIME_CONTRACT");
    println!("cargo:rerun-if-changed={}", repo_copy.display());

    let source = env::var_os("JCODE_RUNTIME_CONTRACT")
        .map(PathBuf::from)
        .filter(|p| p.is_file())
        .or_else(|| repo_copy.is_file().then_some(repo_copy.clone()));
    if let Some(path) = &source {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    let contract = source
        .and_then(|p| fs::read_to_string(p).ok())
        .unwrap_or_default();

    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR")).join("runtime_contract.md");
    fs::write(out, contract).expect("write embedded runtime contract snapshot");
}
