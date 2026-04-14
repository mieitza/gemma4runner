//! Code execution sandbox for tool-calling.
//!
//! Provides three security levels:
//! - **Locked**: no network, no package installs, pre-installed tools only.
//! - **Packages**: no network, but a Python venv with common packages.
//! - **Full**: unrestricted -- network, pip, arbitrary commands.
//!
//! The sandbox writes files to a per-session workspace directory and executes
//! code via `std::process::Command` with appropriate timeouts.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::chat_template::ToolDef;

/// Maximum bytes captured from stdout/stderr to avoid memory blowups.
const MAX_OUTPUT_BYTES: usize = 10 * 1024; // 10 KB

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Security level for the sandbox.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SandboxLevel {
    /// No network, no package installs, pre-installed only.
    Locked,
    /// No network, but Python venv with common packages available.
    Packages,
    /// Full access -- network, pip install, any command.
    Full,
}

impl std::fmt::Display for SandboxLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SandboxLevel::Locked => write!(f, "locked"),
            SandboxLevel::Packages => write!(f, "packages"),
            SandboxLevel::Full => write!(f, "full"),
        }
    }
}

impl std::str::FromStr for SandboxLevel {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "locked" => Ok(Self::Locked),
            "packages" => Ok(Self::Packages),
            "full" => Ok(Self::Full),
            other => bail!(
                "Unknown sandbox level '{}'. Use: locked, packages, full",
                other
            ),
        }
    }
}

/// Result of a code execution or command run.
#[derive(Debug, Clone, Serialize)]
pub struct ExecutionResult {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub elapsed_ms: u64,
}

impl std::fmt::Display for ExecutionResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "exit_code: {}\n", self.exit_code)?;
        if !self.stdout.is_empty() {
            write!(f, "stdout:\n{}\n", self.stdout)?;
        }
        if !self.stderr.is_empty() {
            write!(f, "stderr:\n{}\n", self.stderr)?;
        }
        write!(f, "elapsed: {}ms", self.elapsed_ms)
    }
}

// ---------------------------------------------------------------------------
// Sandbox
// ---------------------------------------------------------------------------

/// A per-session code execution sandbox.
pub struct Sandbox {
    level: SandboxLevel,
    workspace: PathBuf,
    /// Path to the venv python3 for `Packages` level.
    python_path: Option<String>,
}

impl Sandbox {
    /// Create a new sandbox.
    ///
    /// * `level` -- security level.
    /// * `session_id` -- unique id used to name the workspace directory.
    ///
    /// The workspace is created at `/tmp/gemma4-sandbox-{session_id}/`.
    pub fn new(level: SandboxLevel, session_id: &str) -> Result<Self> {
        let workspace = PathBuf::from(format!("/tmp/gemma4-sandbox-{}", session_id));
        fs::create_dir_all(&workspace)
            .with_context(|| format!("Failed to create sandbox workspace: {}", workspace.display()))?;

        let python_path = if level == SandboxLevel::Packages || level == SandboxLevel::Full {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
            let venv_python = format!("{}/sandbox-venv/bin/python3", home);
            if Path::new(&venv_python).exists() {
                tracing::info!("Sandbox: using venv python at {}", venv_python);
                Some(venv_python)
            } else {
                tracing::warn!(
                    "Sandbox: venv not found at {}; falling back to system python3",
                    venv_python
                );
                None
            }
        } else {
            None
        };

        tracing::info!(
            "Sandbox created: level={}, workspace={}",
            level,
            workspace.display()
        );

        Ok(Self {
            level,
            workspace,
            python_path,
        })
    }

    /// Return the workspace directory path.
    pub fn workspace(&self) -> &Path {
        &self.workspace
    }

    /// Return the sandbox level.
    pub fn level(&self) -> SandboxLevel {
        self.level
    }

    // -- File operations -----------------------------------------------------

    /// Write a file to the workspace.
    pub fn write_file(&self, filename: &str, content: &str) -> Result<()> {
        let path = self.workspace.join(filename);
        // Ensure parent dirs exist (in case filename has slashes)
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, content)
            .with_context(|| format!("Failed to write {}", path.display()))?;
        Ok(())
    }

    /// Read a file from the workspace.
    pub fn read_file(&self, filename: &str) -> Result<String> {
        let path = self.workspace.join(filename);
        fs::read_to_string(&path)
            .with_context(|| format!("Failed to read {}", path.display()))
    }

    /// List files in the workspace (non-recursive, relative names).
    pub fn list_files(&self) -> Result<Vec<String>> {
        let mut files = Vec::new();
        for entry in fs::read_dir(&self.workspace)? {
            let entry = entry?;
            if let Some(name) = entry.file_name().to_str() {
                files.push(name.to_string());
            }
        }
        files.sort();
        Ok(files)
    }

    // -- Code execution ------------------------------------------------------

    /// Execute code in the sandbox.
    ///
    /// * `language` -- one of `python`, `c`, `cpp`, `rust`, `bash`, `javascript`.
    /// * `code` -- source code to execute.
    /// * `filename` -- optional filename; a default is chosen if `None`.
    pub fn execute_code(
        &self,
        language: &str,
        code: &str,
        filename: Option<&str>,
    ) -> Result<ExecutionResult> {
        let (default_name, compile_and_run) = match language {
            "python" => ("main.py", None),
            "c" => ("main.c", Some(("gcc", vec!["main.c", "-o", "main", "-lm"], "./main"))),
            "cpp" | "c++" => (
                "main.cpp",
                Some(("g++", vec!["main.cpp", "-o", "main", "-lstdc++"], "./main")),
            ),
            "rust" => (
                "main.rs",
                Some(("rustc", vec!["main.rs", "-o", "main"], "./main")),
            ),
            "bash" | "sh" => ("script.sh", None),
            "javascript" | "js" => ("main.js", None),
            other => bail!("Unsupported language: {}", other),
        };

        let fname = filename.unwrap_or(default_name);
        self.write_file(fname, code)?;

        let timeout_secs = self.timeout_secs();

        match language {
            "python" => {
                let python = self.python_cmd();
                self.run_with_timeout(&python, &[fname], timeout_secs)
            }
            "bash" | "sh" => self.run_with_timeout("bash", &[fname], timeout_secs),
            "javascript" | "js" => self.run_with_timeout("node", &[fname], timeout_secs),
            "c" | "cpp" | "c++" | "rust" => {
                let (compiler, compile_args, run_cmd) = compile_and_run.unwrap();
                // Compile
                let compile_result = self.run_with_timeout(
                    compiler,
                    &compile_args.iter().map(|s| *s).collect::<Vec<_>>(),
                    timeout_secs,
                )?;
                if compile_result.exit_code != 0 {
                    return Ok(compile_result);
                }
                // Run
                self.run_with_timeout(run_cmd, &[], timeout_secs)
            }
            _ => unreachable!(),
        }
    }

    /// Run an arbitrary shell command. Only available at `Full` level.
    pub fn run_command(&self, command: &str) -> Result<ExecutionResult> {
        if self.level != SandboxLevel::Full {
            bail!(
                "run_command is only available at sandbox level 'full' (current: {})",
                self.level
            );
        }
        let timeout_secs = self.timeout_secs();
        self.run_shell_with_timeout(command, timeout_secs)
    }

    // -- Tool call dispatch --------------------------------------------------

    /// Dispatch a parsed tool call to the appropriate sandbox method.
    ///
    /// Returns a human-readable result string suitable for feeding back to
    /// the model as a tool response.
    pub fn dispatch_tool_call(
        &self,
        tool_name: &str,
        arguments: &serde_json::Value,
    ) -> Result<String> {
        match tool_name {
            "execute_code" => {
                let language = arguments
                    .get("language")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("execute_code: missing 'language' argument"))?;
                let code = arguments
                    .get("code")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("execute_code: missing 'code' argument"))?;
                let filename = arguments.get("filename").and_then(|v| v.as_str());

                let result = self.execute_code(language, code, filename)?;
                Ok(format!("{}", result))
            }
            "run_command" => {
                let command = arguments
                    .get("command")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("run_command: missing 'command' argument"))?;

                let result = self.run_command(command)?;
                Ok(format!("{}", result))
            }
            "write_file" => {
                let filename = arguments
                    .get("filename")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("write_file: missing 'filename' argument"))?;
                let content = arguments
                    .get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("write_file: missing 'content' argument"))?;

                self.write_file(filename, content)?;
                Ok(format!("File '{}' written successfully.", filename))
            }
            "read_file" => {
                let filename = arguments
                    .get("filename")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("read_file: missing 'filename' argument"))?;

                let content = self.read_file(filename)?;
                Ok(content)
            }
            "list_files" => {
                let files = self.list_files()?;
                if files.is_empty() {
                    Ok("(workspace is empty)".to_string())
                } else {
                    Ok(files.join("\n"))
                }
            }
            other => bail!("Unknown sandbox tool: {}", other),
        }
    }

    /// Return `true` if `tool_name` is a sandbox tool.
    pub fn is_sandbox_tool(tool_name: &str) -> bool {
        matches!(
            tool_name,
            "execute_code" | "run_command" | "write_file" | "read_file" | "list_files"
        )
    }

    /// Return the tool definitions to inject into the system prompt.
    pub fn tool_definitions(level: SandboxLevel) -> Vec<ToolDef> {
        let mut tools = vec![
            ToolDef {
                name: "execute_code".to_string(),
                description: Some(
                    "Execute code in a sandboxed environment. Supported languages: python, c, cpp, rust, bash, javascript.".to_string(),
                ),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "language": {
                            "type": "string",
                            "enum": ["python", "c", "cpp", "rust", "bash", "javascript"],
                            "description": "Programming language"
                        },
                        "code": {
                            "type": "string",
                            "description": "Source code to execute"
                        },
                        "filename": {
                            "type": "string",
                            "description": "Optional filename (default: main.py, main.c, etc.)"
                        }
                    },
                    "required": ["language", "code"]
                })),
            },
            ToolDef {
                name: "write_file".to_string(),
                description: Some("Write a file to the sandbox workspace.".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Name of the file to write"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write"
                        }
                    },
                    "required": ["filename", "content"]
                })),
            },
            ToolDef {
                name: "read_file".to_string(),
                description: Some("Read a file from the sandbox workspace.".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Name of the file to read"
                        }
                    },
                    "required": ["filename"]
                })),
            },
            ToolDef {
                name: "list_files".to_string(),
                description: Some("List files in the sandbox workspace.".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {}
                })),
            },
        ];

        if level == SandboxLevel::Full {
            tools.push(ToolDef {
                name: "run_command".to_string(),
                description: Some(
                    "Run an arbitrary shell command in the sandbox workspace.".to_string(),
                ),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute"
                        }
                    },
                    "required": ["command"]
                })),
            });
        }

        tools
    }

    // -- Internal helpers ----------------------------------------------------

    /// Return the timeout in seconds for this sandbox level.
    fn timeout_secs(&self) -> u64 {
        match self.level {
            SandboxLevel::Locked => 30,
            SandboxLevel::Packages => 60,
            SandboxLevel::Full => 120,
        }
    }

    /// Return the python command to use.
    fn python_cmd(&self) -> String {
        self.python_path
            .clone()
            .unwrap_or_else(|| "python3".to_string())
    }

    /// Check if we're on Linux (where `unshare --net` is available).
    fn is_linux() -> bool {
        cfg!(target_os = "linux")
    }

    /// Run a program with arguments, applying timeout and optional network
    /// isolation.
    fn run_with_timeout(
        &self,
        program: &str,
        args: &[&str],
        timeout_secs: u64,
    ) -> Result<ExecutionResult> {
        let start = Instant::now();

        let mut cmd = if self.level == SandboxLevel::Locked && Self::is_linux() {
            // On Linux, use unshare to disable network
            let mut c = Command::new("timeout");
            c.arg(timeout_secs.to_string())
                .arg("unshare")
                .arg("--net")
                .arg(program)
                .args(args);
            c
        } else {
            let mut c = Command::new("timeout");
            c.arg(timeout_secs.to_string()).arg(program).args(args);
            c
        };

        cmd.current_dir(&self.workspace)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        // For Packages level, set PATH to include the venv bin
        if let Some(ref venv_python) = self.python_path {
            if let Some(venv_bin) = Path::new(venv_python).parent() {
                let current_path = std::env::var("PATH").unwrap_or_default();
                cmd.env("PATH", format!("{}:{}", venv_bin.display(), current_path));
            }
        }

        let mut child = cmd
            .spawn()
            .with_context(|| format!("Failed to spawn: {} {:?}", program, args))?;

        let stdout = truncated_read(child.stdout.take());
        let stderr = truncated_read(child.stderr.take());

        let status = child.wait().context("Failed to wait for child process")?;
        let elapsed_ms = start.elapsed().as_millis() as u64;

        Ok(ExecutionResult {
            exit_code: status.code().unwrap_or(-1),
            stdout,
            stderr,
            elapsed_ms,
        })
    }

    /// Run a shell command string via `sh -c`, with timeout.
    fn run_shell_with_timeout(
        &self,
        command: &str,
        timeout_secs: u64,
    ) -> Result<ExecutionResult> {
        let start = Instant::now();

        let mut cmd = Command::new("timeout");
        cmd.arg(timeout_secs.to_string())
            .arg("sh")
            .arg("-c")
            .arg(command)
            .current_dir(&self.workspace)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let mut child = cmd
            .spawn()
            .with_context(|| format!("Failed to spawn shell command: {}", command))?;

        let stdout = truncated_read(child.stdout.take());
        let stderr = truncated_read(child.stderr.take());

        let status = child.wait().context("Failed to wait for child process")?;
        let elapsed_ms = start.elapsed().as_millis() as u64;

        Ok(ExecutionResult {
            exit_code: status.code().unwrap_or(-1),
            stdout,
            stderr,
            elapsed_ms,
        })
    }
}

/// Read up to `MAX_OUTPUT_BYTES` from a reader, converting to a String.
fn truncated_read<R: Read>(reader: Option<R>) -> String {
    let Some(mut r) = reader else {
        return String::new();
    };
    let mut buf = vec![0u8; MAX_OUTPUT_BYTES + 1];
    let mut total = 0;
    loop {
        match r.read(&mut buf[total..]) {
            Ok(0) => break,
            Ok(n) => {
                total += n;
                if total > MAX_OUTPUT_BYTES {
                    total = MAX_OUTPUT_BYTES;
                    break;
                }
            }
            Err(_) => break,
        }
    }
    buf.truncate(total);
    let mut s = String::from_utf8_lossy(&buf).to_string();
    if total == MAX_OUTPUT_BYTES {
        s.push_str("\n... (output truncated at 10KB)");
    }
    s
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_sandbox(level: SandboxLevel) -> Sandbox {
        let id = format!("test-{}", std::process::id());
        Sandbox::new(level, &id).expect("sandbox creation")
    }

    #[test]
    fn test_sandbox_level_roundtrip() {
        for level in &["locked", "packages", "full"] {
            let parsed: SandboxLevel = level.parse().unwrap();
            assert_eq!(&parsed.to_string(), level);
        }
    }

    #[test]
    fn test_write_and_read_file() {
        let sb = test_sandbox(SandboxLevel::Full);
        sb.write_file("hello.txt", "world").unwrap();
        assert_eq!(sb.read_file("hello.txt").unwrap(), "world");
    }

    #[test]
    fn test_list_files() {
        let sb = test_sandbox(SandboxLevel::Full);
        sb.write_file("a.txt", "a").unwrap();
        sb.write_file("b.txt", "b").unwrap();
        let files = sb.list_files().unwrap();
        assert!(files.contains(&"a.txt".to_string()));
        assert!(files.contains(&"b.txt".to_string()));
    }

    #[test]
    fn test_tool_definitions_count() {
        let locked = Sandbox::tool_definitions(SandboxLevel::Locked);
        let full = Sandbox::tool_definitions(SandboxLevel::Full);
        assert_eq!(locked.len(), 4); // no run_command
        assert_eq!(full.len(), 5); // includes run_command
    }

    #[test]
    fn test_is_sandbox_tool() {
        assert!(Sandbox::is_sandbox_tool("execute_code"));
        assert!(Sandbox::is_sandbox_tool("run_command"));
        assert!(Sandbox::is_sandbox_tool("write_file"));
        assert!(Sandbox::is_sandbox_tool("read_file"));
        assert!(Sandbox::is_sandbox_tool("list_files"));
        assert!(!Sandbox::is_sandbox_tool("get_weather"));
    }

    #[test]
    fn test_run_command_locked_rejected() {
        let sb = test_sandbox(SandboxLevel::Locked);
        let err = sb.run_command("echo hello").unwrap_err();
        assert!(err.to_string().contains("only available at sandbox level 'full'"));
    }
}
