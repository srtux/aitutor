"""Automated open-source hygiene and standalone portability test."""

from __future__ import annotations

from scripts.oss_audit import REPO_ROOT, run_audit


def test_repository_oss_hygiene() -> None:
    violations = run_audit(REPO_ROOT)
    assert not violations, "OSS hygiene violations found:\n" + "\n".join(violations)
