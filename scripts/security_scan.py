"""Scan project text files for leaked secrets, local paths, and RAG privacy risks."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKIPPED_DIRECTORIES = {
    ".git",
    ".venv",
    "__pycache__",
    ".ipynb_checkpoints",
    "storage",
}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
    ".ipynb",
    ".example",
    ".gitignore",
}
LOCAL_SECRET_PATHS = {
    ".env",
    ".streamlit/secrets.toml",
}
SECRET_ASSIGNMENT = re.compile(
    r"(?:DASHSCOPE|DEEPSEEK|OPENAI)_API_KEY\s*=\s*([\"'])([^\"']*)\1",
    re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(
    # Short ``sk-...`` fragments also occur in generated notebook HTML.
    r"(?<![A-Za-z0-9])(?:sk|dashscope|deepseek)-[A-Za-z0-9_-]{28,}",
    re.IGNORECASE,
)
WINDOWS_ABSOLUTE_PATH = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]:[\\/](?![\\/])"
)
POSIX_HOME_PATH = re.compile(r"/(?:Users|home)/[^\s\"']+")
CUSTOMER_ID_PATTERN = re.compile(r"\b[0-9]{4}-[A-Z]{5}\b")
PLACEHOLDER_MARKERS = (
    "offline",
    "test",
    "example",
    "your",
    "replace",
    "请",
    "填写",
)


def _iter_text_files() -> list[Path]:
    files: list[Path] = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(PROJECT_ROOT).parts
        if any(part in SKIPPED_DIRECTORIES for part in relative_parts):
            continue
        if path.name == ".env" or path.suffix.lower() in TEXT_SUFFIXES:
            files.append(path)
    return files


def _is_placeholder(value: str) -> bool:
    normalized = value.strip().lower()
    return not normalized or any(item in normalized for item in PLACEHOLDER_MARKERS)


def _is_git_ignored(relative_path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", relative_path],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def run_scan() -> dict[str, Any]:
    hardcoded_secrets: list[dict[str, Any]] = []
    absolute_paths: list[dict[str, Any]] = []
    knowledge_customer_ids: list[dict[str, Any]] = []
    local_secret_files: list[dict[str, Any]] = []
    scanned_files = 0

    for path in _iter_text_files():
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        scanned_files += 1

        if relative in LOCAL_SECRET_PATHS:
            contains_configured_key = any(
                match and not _is_placeholder(match.group(2))
                for match in (
                    SECRET_ASSIGNMENT.search(line)
                    for line in text.splitlines()
                )
            )
            local_secret_files.append(
                {
                    "path": relative,
                    "git_ignored": _is_git_ignored(relative),
                    "contains_configured_key": contains_configured_key,
                }
            )

        for line_number, line in enumerate(text.splitlines(), start=1):
            is_local_secret_file = relative in LOCAL_SECRET_PATHS
            assignment = SECRET_ASSIGNMENT.search(line)
            if assignment and not _is_placeholder(assignment.group(2)):
                if not is_local_secret_file:
                    hardcoded_secrets.append(
                        {
                            "path": relative,
                            "line": line_number,
                            "kind": "api_key_assignment",
                        }
                    )
            if TOKEN_PATTERN.search(line) and not is_local_secret_file:
                hardcoded_secrets.append(
                    {
                        "path": relative,
                        "line": line_number,
                        "kind": "token_pattern",
                    }
                )
            if WINDOWS_ABSOLUTE_PATH.search(line) or POSIX_HOME_PATH.search(line):
                absolute_paths.append(
                    {
                        "path": relative,
                        "line": line_number,
                    }
                )
            if relative.startswith("knowledge/") and CUSTOMER_ID_PATTERN.search(line):
                knowledge_customer_ids.append(
                    {
                        "path": relative,
                        "line": line_number,
                    }
                )

    result = {
        "scanned_files": scanned_files,
        "hardcoded_secret_findings": hardcoded_secrets,
        "absolute_path_findings": absolute_paths,
        "knowledge_customer_id_findings": knowledge_customer_ids,
        "local_secret_files": local_secret_files,
        "passed": not (
            hardcoded_secrets or absolute_paths or knowledge_customer_ids
        )
        and all(item["git_ignored"] for item in local_secret_files),
        "privacy_note": (
            "telco_customer_churn.csv 是公开演示数据；知识库未发现 customerID。"
        ),
    }
    return result


def scan_git_history() -> dict[str, Any]:
    """Inspect tracked text blobs without printing any suspected secret value."""

    objects = subprocess.run(
        ["git", "rev-list", "--objects", "--all"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    findings: list[dict[str, Any]] = []
    scanned_blobs = 0
    skipped_large_blobs = 0
    seen: set[str] = set()
    for entry in objects.stdout.splitlines():
        object_id, separator, relative = entry.partition(" ")
        if not separator or not relative or object_id in seen:
            continue
        seen.add(object_id)
        path = Path(relative)
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name != ".env":
            continue
        size_result = subprocess.run(
            ["git", "cat-file", "-s", object_id],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        if int(size_result.stdout.strip()) > 2_000_000:
            skipped_large_blobs += 1
            continue
        blob = subprocess.run(
            ["git", "cat-file", "blob", object_id],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=True,
        ).stdout.decode("utf-8", errors="replace")
        scanned_blobs += 1
        if relative in LOCAL_SECRET_PATHS:
            findings.append(
                {"path": relative, "object": object_id[:12], "kind": "tracked_secret_file"}
            )
        for line_number, line in enumerate(blob.splitlines(), start=1):
            assignment = SECRET_ASSIGNMENT.search(line)
            if assignment and not _is_placeholder(assignment.group(2)):
                findings.append(
                    {
                        "path": relative,
                        "object": object_id[:12],
                        "line": line_number,
                        "kind": "api_key_assignment",
                    }
                )
            if TOKEN_PATTERN.search(line):
                findings.append(
                    {
                        "path": relative,
                        "object": object_id[:12],
                        "line": line_number,
                        "kind": "token_pattern",
                    }
                )
    return {
        "scanned_blobs": scanned_blobs,
        "skipped_large_blobs": skipped_large_blobs,
        "findings": findings,
        "passed": not findings and skipped_large_blobs == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history", action="store_true", help="同时扫描Git历史文本对象"
    )
    args = parser.parse_args()
    result = run_scan()
    if args.history:
        result["git_history"] = scan_git_history()
        result["passed"] = result["passed"] and result["git_history"]["passed"]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
