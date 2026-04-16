#!/usr/bin/env python
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_GITIGNORE_PATTERNS = (
    ".env",
    ".env.*",
    "!.env.example",
    "*.pem",
    "*.key",
    "*.crt",
    "*.p12",
    "*.pfx",
    "*.jks",
)

REQUIRED_DOCKERIGNORE_PATTERNS = (
    ".env",
    ".env.*",
    "!.env.example",
    "*.pem",
    "*.key",
    "*.crt",
    "*.p12",
    "*.pfx",
    "*.jks",
)

REQUIRED_DASHBOARD_DOCKERIGNORE_PATTERNS = REQUIRED_DOCKERIGNORE_PATTERNS

PLACEHOLDER_MARKERS = (
    "change-me",
    "replace-with",
    "example",
    "localhost",
    "127.0.0.1",
    "postgres",
    "admin",
    "dev-internal-api-key",
    "${",
)

SUSPICIOUS_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bASIA[0-9A-Z]{16}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
)

SENSITIVE_ENV_KEYS = {
    "POSTGRES_PASSWORD",
    "JWT_SECRET_KEY",
    "INTERNAL_API_KEY",
    "DEMO_ADMIN_PASSWORD",
    "DEMO_ANALYST_PASSWORD",
    "DEMO_VIEWER_PASSWORD",
    "AWS_SECRET_ACCESS_KEY",
}

TEXT_FILE_SUFFIXES = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".json",
    ".md",
    ".yml",
    ".yaml",
    ".txt",
    ".conf",
    ".ini",
    ".sh",
    ".ps1",
}

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "dist",
    "pytest_tmp_probe",
    "pytest_tmp_codex",
    "pytest_tmp_files",
    # external dataset directories may contain real-world data with patterns
    # that look like secrets (e.g. AWS account IDs in CloudTrail fixture files).
    # These are not committed to the repo and must not be scanned.
    "external",
}

SENSITIVE_REPO_FILE_PATTERNS = (
    re.compile(r"(^|/)\.env(\.[^/]+)?$", re.IGNORECASE),
    re.compile(r"(^|/).+\.(pem|key|crt|p12|pfx|jks)$", re.IGNORECASE),
)

ALLOWED_SENSITIVE_REPO_FILES = {
    ".env.example",
}


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _load_ignore_file(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in _read_lines(path)
        if line.strip() and not line.lstrip().startswith("#")
    }


def _looks_like_placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    if not lowered:
        return True
    return any(marker in lowered for marker in PLACEHOLDER_MARKERS)


def _check_ignore_patterns(filename: str, patterns: tuple[str, ...], loaded: set[str]) -> list[str]:
    missing = [pattern for pattern in patterns if pattern not in loaded]
    return [f"{filename} is missing ignore pattern '{pattern}'" for pattern in missing]


def _iter_repo_files() -> list[Path]:
    files: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        if path.name.startswith(".env"):
            continue
        if path.suffix in TEXT_FILE_SUFFIXES or path.name.startswith(".env"):
            files.append(path)
    return files


def _git_list_paths(*args: str) -> set[str]:
    command = [
        "git",
        "-c",
        f"safe.directory={REPO_ROOT.as_posix()}",
        *args,
    ]
    try:
        output = subprocess.check_output(
            command,
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return set()
    return {line.strip().replace("\\", "/") for line in output.splitlines() if line.strip()}


def _is_sensitive_repo_file(path_str: str) -> bool:
    normalized = path_str.replace("\\", "/")
    if normalized in ALLOWED_SENSITIVE_REPO_FILES:
        return False
    return any(pattern.search(normalized) for pattern in SENSITIVE_REPO_FILE_PATTERNS)


def _check_git_tracking_of_sensitive_files() -> list[str]:
    findings: list[str] = []
    tracked = _git_list_paths("ls-files")
    staged = _git_list_paths("diff", "--cached", "--name-only", "--diff-filter=ACMR")
    staged_deleted = _git_list_paths("diff", "--cached", "--name-only", "--diff-filter=D")

    for path_str in sorted((tracked | staged) - staged_deleted):
        if _is_sensitive_repo_file(path_str):
            findings.append(f"Sensitive file is tracked or staged for commit: {path_str}")

    return findings


def _check_example_env_file(path: Path) -> list[str]:
    findings: list[str] = []
    for line in _read_lines(path):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key.startswith("VITE_") and key != "VITE_API_URL":
            findings.append(f"{path.relative_to(REPO_ROOT)} exposes unsupported frontend env var '{key}'")
        if key in SENSITIVE_ENV_KEYS and value and not _looks_like_placeholder(value):
            findings.append(f"{path.relative_to(REPO_ROOT)} contains a non-placeholder value for '{key}'")
    return findings


def _check_frontend_env_usage(path: Path) -> list[str]:
    findings: list[str] = []
    vite_pattern = re.compile(r"VITE_[A-Z0-9_]+")
    allowed = {"VITE_API_URL"}
    for line_no, line in enumerate(_read_lines(path), start=1):
        for match in vite_pattern.findall(line):
            if match not in allowed:
                findings.append(
                    f"{path.relative_to(REPO_ROOT)}:{line_no} uses unsupported frontend env var '{match}'"
                )
    return findings


def _check_suspicious_content(path: Path) -> list[str]:
    findings: list[str] = []
    is_test_fixture = "tests" in path.parts
    for line_no, line in enumerate(_read_lines(path), start=1):
        for pattern in SUSPICIOUS_SECRET_PATTERNS:
            if pattern.search(line):
                findings.append(f"{path.relative_to(REPO_ROOT)}:{line_no} matches secret pattern '{pattern.pattern}'")

        stripped = line.strip()
        if "=" not in stripped or stripped.startswith("#"):
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key in SENSITIVE_ENV_KEYS and value and not _looks_like_placeholder(value) and not is_test_fixture:
            findings.append(f"{path.relative_to(REPO_ROOT)}:{line_no} contains a non-placeholder value for '{key}'")
    return findings


def run_checks() -> list[str]:
    findings: list[str] = []

    findings.extend(_check_git_tracking_of_sensitive_files())
    findings.extend(
        _check_ignore_patterns(".gitignore", REQUIRED_GITIGNORE_PATTERNS, _load_ignore_file(REPO_ROOT / ".gitignore"))
    )
    findings.extend(
        _check_ignore_patterns(".dockerignore", REQUIRED_DOCKERIGNORE_PATTERNS, _load_ignore_file(REPO_ROOT / ".dockerignore"))
    )
    findings.extend(
        _check_ignore_patterns(
            "dashboard/.dockerignore",
            REQUIRED_DASHBOARD_DOCKERIGNORE_PATTERNS,
            _load_ignore_file(REPO_ROOT / "dashboard" / ".dockerignore"),
        )
    )

    for env_name in (".env.example",):
        findings.extend(_check_example_env_file(REPO_ROOT / env_name))

    for path in _iter_repo_files():
        if "dashboard" in path.parts:
            findings.extend(_check_frontend_env_usage(path))
        findings.extend(_check_suspicious_content(path))

    return findings


def main() -> int:
    findings = run_checks()
    if findings:
        for finding in findings:
            print(f"ERROR: {finding}")
        return 1

    print("Repository safety checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
