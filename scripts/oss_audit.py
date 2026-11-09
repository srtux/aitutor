#!/usr/bin/env python3
"""Open-source hygiene and standalone portability audit for AiTutor.

Scans repository source, configuration, test, and documentation files to verify:
1. Zero internal/corporate monorepo references, shortlinks, or corporate domains.
2. Zero hardcoded developer home paths (/Users/<name>, /home/<name>, etc.).
3. Zero internal monorepo build or ownership metadata files.
4. Presence of open-source LICENSE, README.md, pyproject.toml, and requirements.txt.
"""

from __future__ import annotations

import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "vendor",
    "whisper",
}

SKIP_SUFFIXES = {
    ".onnx",
    ".bin",
    ".task",
    ".wasm",
    ".pyc",
    ".pyo",
    ".png",
    ".jpg",
    ".jpeg",
    ".wav",
    ".mp3",
}

FORBIDDEN_FILENAMES = {
    "BUILD",
    "BUILD.bazel",
    "WORKSPACE",
    "METADATA",
    "OWNERS",
    "AGENT_OWNERS",
}

# Patterns constructed dynamically so this audit file itself contains zero raw hits.
def _join_tokens(tokens: list[tuple[str, str]]) -> str:
    return "|".join(a + b for a, b in tokens)


FORBIDDEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "internal_monorepo_ref",
        re.compile(
            r"\b(?:"
            + _join_tokens(
                [
                    ("google", "3"),
                    ("google", "test"),
                    ("pyg", "lib"),
                    ("ci", "tc"),
                    ("Pip", "er"),
                    ("Criti", "que"),
                    ("Tricor", "der"),
                ]
            )
            + r")\b"
        ),
    ),
    (
        "internal_build_ref",
        re.compile(
            r"\b(?:"
            + _join_tokens(
                [
                    ("blaze-", "bin"),
                    ("blaze-", "genfiles"),
                    ("blaze-", "out"),
                    ("blaze-for-", "agents"),
                ]
            )
            + r")\b"
        ),
    ),
    (
        "internal_domain",
        re.compile(
            r"(?:"
            + "googleplex"
            + r"\.com|"
            + "googlers"
            + r"\.com|\.corp\."
            + "google"
            + r"\.com|@"
            + "google"
            + r"\.com)"
        ),
    ),
    (
        "internal_shortlink",
        re.compile(r"(?<![A-Za-z0-9_./-])(?:go|cl|b)/[A-Za-z0-9_-]{3,}"),
    ),
    (
        "hardcoded_host_path",
        re.compile(r"/(?:usr/local/" + "google" + r"|" + "google" + r"/src|" + "google" + r"/bin|Users/[A-Za-z0-9_-]+|home/[A-Za-z0-9_-]+)"),
    ),
]

REQUIRED_FILES = [
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    ".gitignore",
]


def iter_repo_files(root: pathlib.Path) -> list[pathlib.Path]:
    """Return all non-ignored text files in the repository."""
    files: list[pathlib.Path] = []
    for path in sorted(root.rglob("*")):
        rel_parts = path.relative_to(root).parts
        if any(part in SKIP_DIR_NAMES for part in rel_parts):
            continue
        if path.is_dir():
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        files.append(path)
    return files


def run_audit(root: pathlib.Path = REPO_ROOT) -> list[str]:
    """Return a list of hygiene violation descriptions (empty if clean)."""
    violations: list[str] = []

    for req in REQUIRED_FILES:
        if not (root / req).is_file():
            violations.append(f"Missing required open-source file: {req}")

    for path in iter_repo_files(root):
        rel = path.relative_to(root)
        if path.name in FORBIDDEN_FILENAMES:
            violations.append(f"{rel}: forbidden internal build/metadata file")
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for lineno, line in enumerate(text.splitlines(), start=1):
            for rule_name, pattern in FORBIDDEN_PATTERNS:
                match = pattern.search(line)
                if match:
                    violations.append(
                        f"{rel}:{lineno} [{rule_name}] matched {match.group(0)!r}"
                    )

    return violations


def main() -> int:
    violations = run_audit(REPO_ROOT)
    if violations:
        print(f"OSS hygiene audit FAILED ({len(violations)} violation(s)):", file=sys.stderr)
        for item in violations:
            print(f"  - {item}", file=sys.stderr)
        return 1
    print("OSS hygiene audit PASSED: 0 violations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
