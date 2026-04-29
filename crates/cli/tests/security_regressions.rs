use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

use serde_json::Value;
use tempfile::TempDir;

fn binary_path() -> PathBuf {
    std::env::var_os("CARGO_BIN_EXE_cosmos-vanity")
        .expect("CARGO_BIN_EXE_cosmos-vanity not set")
        .into()
}

fn base_command() -> Command {
    let mut command = Command::new(binary_path());
    command.env("RUST_BACKTRACE", "0");
    command
}

#[track_caller]
fn run(args: &[&str]) -> Output {
    base_command()
        .args(args)
        .output()
        .expect("failed to run cosmos-vanity")
}

#[track_caller]
fn run_with_stdin(args: &[&str], stdin: &str) -> Output {
    let mut child = base_command()
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn cosmos-vanity");

    child
        .stdin
        .as_mut()
        .expect("stdin pipe missing")
        .write_all(stdin.as_bytes())
        .expect("failed to write stdin");

    child
        .wait_with_output()
        .expect("failed to wait on cosmos-vanity")
}

fn stdout_text(output: &Output) -> String {
    String::from_utf8(output.stdout.clone()).expect("stdout was not utf-8")
}

fn stderr_text(output: &Output) -> String {
    String::from_utf8(output.stderr.clone()).expect("stderr was not utf-8")
}

#[track_caller]
fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "expected success\nstatus: {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        stdout_text(output),
        stderr_text(output)
    );
}

#[track_caller]
fn assert_failure(output: &Output) {
    assert!(
        !output.status.success(),
        "expected failure\nstatus: {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        stdout_text(output),
        stderr_text(output)
    );
}

fn parse_json(stdout: &str) -> Value {
    serde_json::from_str(stdout.trim()).expect("failed to parse json output")
}

fn parse_trailing_json(stdout: &str) -> Value {
    let start = stdout.find('{').expect("json object missing from stdout");
    serde_json::from_str(stdout[start..].trim()).expect("failed to parse trailing json output")
}

struct WalletFixture {
    address: String,
    mnemonic: String,
}

fn generate_wallet_fixture() -> WalletFixture {
    let output = run(&[
        "generate",
        "--unsafe-print-secrets",
        "--format",
        "json",
        "--log-level",
        "error",
    ]);
    assert_success(&output);

    let json = parse_json(&stdout_text(&output));
    WalletFixture {
        address: json["address"]
            .as_str()
            .expect("address missing")
            .to_string(),
        mnemonic: json["mnemonic"]
            .as_str()
            .expect("mnemonic missing")
            .to_string(),
    }
}

#[test]
fn generate_json_redacts_mnemonic_by_default() {
    let output = run(&["generate", "--format", "json", "--log-level", "error"]);
    assert_success(&output);

    let json = parse_json(&stdout_text(&output));
    assert_eq!(json["mnemonic"], Value::Null);
    assert_eq!(json["secret_file"], Value::Null);
    assert_eq!(json["secrets_redacted"], Value::Bool(true));
}

#[test]
fn generate_secret_file_keeps_stdout_redacted_and_uses_restrictive_permissions() {
    let tempdir = TempDir::new().expect("tempdir");
    let secret_path = tempdir.path().join("wallet-secret.json");
    let secret_path_str = secret_path.display().to_string();

    let output = run(&[
        "generate",
        "--secret-file",
        &secret_path_str,
        "--format",
        "json",
        "--log-level",
        "error",
    ]);
    assert_success(&output);

    let stdout = stdout_text(&output);
    let json = parse_json(&stdout);
    assert_eq!(json["mnemonic"], Value::Null);
    assert_eq!(json["secret_file"].as_str(), Some(secret_path_str.as_str()));
    assert_eq!(json["secrets_redacted"], Value::Bool(true));

    let secret_json = parse_json(&fs::read_to_string(&secret_path).expect("read secret file"));
    let mnemonic = secret_json["mnemonic"].as_str().expect("mnemonic missing");
    assert_eq!(secret_json["address"], json["address"]);
    assert!(!mnemonic.is_empty());
    assert!(!stdout.contains(mnemonic), "stdout leaked mnemonic");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let mode = fs::metadata(&secret_path)
            .expect("metadata")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600, "secret file mode was {:o}", mode);
    }
}

#[test]
fn generate_secret_file_refuses_to_overwrite_existing_file() {
    let tempdir = TempDir::new().expect("tempdir");
    let secret_path = tempdir.path().join("wallet-secret.json");
    fs::write(&secret_path, "sentinel").expect("seed secret file");
    let secret_path_str = secret_path.display().to_string();

    let output = run(&[
        "generate",
        "--secret-file",
        &secret_path_str,
        "--format",
        "json",
        "--log-level",
        "error",
    ]);
    assert_failure(&output);

    let stderr = stderr_text(&output);
    assert!(
        stderr.contains("failed to create secret file"),
        "stderr was: {stderr}"
    );
    assert_eq!(
        fs::read_to_string(&secret_path).expect("read secret file"),
        "sentinel"
    );
}

#[test]
fn verify_accepts_mnemonic_file_and_stdin() {
    let wallet = generate_wallet_fixture();
    let tempdir = TempDir::new().expect("tempdir");
    let mnemonic_path = tempdir.path().join("mnemonic.txt");
    fs::write(&mnemonic_path, &wallet.mnemonic).expect("write mnemonic file");

    let file_output = run(&[
        "verify",
        "--mnemonic-file",
        &mnemonic_path.display().to_string(),
        "--address",
        &wallet.address,
        "--log-level",
        "error",
    ]);
    assert_success(&file_output);
    assert!(
        stdout_text(&file_output).contains(&wallet.address),
        "verify --mnemonic-file output missing address"
    );

    let stdin_output = run_with_stdin(
        &[
            "verify",
            "--mnemonic-stdin",
            "--address",
            &wallet.address,
            "--log-level",
            "error",
        ],
        &wallet.mnemonic,
    );
    assert_success(&stdin_output);
    assert!(
        stdout_text(&stdin_output).contains(&wallet.address),
        "verify --mnemonic-stdin output missing address"
    );
}

#[test]
fn cpu_raw_mode_fails_with_clear_message() {
    let output = run(&[
        "search",
        "-p",
        "q",
        "-m",
        "cpu",
        "-k",
        "raw",
        "-n",
        "1",
        "--log-level",
        "error",
    ]);
    assert_failure(&output);

    let stderr = stderr_text(&output);
    assert!(
        stderr.contains("raw key mode is only supported by the GPU raw pipeline"),
        "stderr was: {stderr}"
    );
}

#[test]
fn search_secret_file_redacts_stdout_and_persists_mnemonic() {
    let tempdir = TempDir::new().expect("tempdir");
    let secret_path = tempdir.path().join("search-secret.json");
    let secret_path_str = secret_path.display().to_string();

    let output = run(&[
        "search",
        "-p",
        "q",
        "-m",
        "cpu",
        "-k",
        "mnemonic",
        "-n",
        "1",
        "--secret-file",
        &secret_path_str,
        "--format",
        "json",
        "--log-level",
        "error",
    ]);
    assert_success(&output);

    let stdout = stdout_text(&output);
    let json = parse_trailing_json(&stdout);
    assert_eq!(json["mnemonic"], Value::Null);
    assert_eq!(json["secret_file"].as_str(), Some(secret_path_str.as_str()));
    assert_eq!(json["verified"], Value::Bool(true));
    assert_eq!(json["secrets_redacted"], Value::Bool(true));

    let secret_json = parse_json(&fs::read_to_string(&secret_path).expect("read secret file"));
    let mnemonic = secret_json["mnemonic"].as_str().expect("mnemonic missing");
    assert_eq!(
        secret_json["key_mode"],
        Value::String("mnemonic".to_string())
    );
    assert_eq!(secret_json["address"], json["address"]);
    assert_eq!(secret_json["derivation_path"], json["derivation_path"]);
    assert!(!mnemonic.is_empty());
    assert!(!stdout.contains(mnemonic), "stdout leaked mnemonic");
}
