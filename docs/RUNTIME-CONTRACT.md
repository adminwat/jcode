# Runtime contract (embedded fallback)

This file is the build-time fallback for the runtime contract jcode freezes
into its binary (`crates/jcode-base/build.rs`). An operator build sets
`JCODE_RUNTIME_CONTRACT=<path>` to embed the real contract; this stub is what
a plain checkout embeds. At runtime `~/.jcode/prompt-overlay.md` wins when it
exists; the embedded copy is served only when that file is missing or
unreadable, so a deleted symlink never silently removes the contract.
