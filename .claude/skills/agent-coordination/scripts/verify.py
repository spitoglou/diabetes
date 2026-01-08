#!/usr/bin/env python3
"""Agent Deliverable Verification Script.

Cross-platform Python replacement for verify.sh.

Usage: python verify.py <category> <name> <date> [git-path]
Returns: 0 if all checks pass, 1 if any fail
"""

import argparse
import subprocess
import sys
from pathlib import Path


# ANSI color codes (work on most terminals including Windows 10+)
class Colors:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    NC = "\033[0m"  # No Color


def enable_windows_ansi():
    """Enable ANSI escape codes on Windows."""
    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            # Enable ANSI escape sequences
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass  # Fall back to no colors


class Verifier:
    """Verify agent deliverables."""

    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, description: str, success: bool) -> None:
        """Record and display a check result."""
        if success:
            print(f"{Colors.GREEN}[PASS]{Colors.NC} {description}")
            self.passed += 1
        else:
            print(f"{Colors.RED}[FAIL]{Colors.NC} {description}")
            self.failed += 1

    def verify(
        self, category: str, name: str, date: str, git_path: str | None = None
    ) -> int:
        """Run all verification checks.

        Args:
            category: Report category (e.g., 'analysis', 'review')
            name: Report name
            date: Report date (YYYY-MM-DD format)
            git_path: Optional path to check for git changes

        Returns:
            0 if all checks pass, 1 if any fail
        """
        # Derived paths
        report_path = Path(f".claude/reports/{category}/{name}-{date}.md")
        registry_path = Path(".claude/reports/_registry.md")

        print(f"Verifying: {report_path}")
        print("---")

        # Check 1: Report exists
        self.check("Report file exists", report_path.exists())

        # Check 2: Report has content (>10 lines)
        if report_path.exists():
            line_count = len(report_path.read_text(encoding="utf-8").splitlines())
            if line_count > 10:
                self.check(f"Report has content ({line_count} lines)", True)
            else:
                self.check(f"Report has content ({line_count} lines < 10)", False)
        else:
            self.check("Report has content", False)

        # Check 3: Registry updated
        registry_has_entry = False
        if registry_path.exists():
            registry_content = registry_path.read_text(encoding="utf-8")
            registry_has_entry = f"{name}-{date}" in registry_content
        self.check("Registry entry exists", registry_has_entry)

        # Check 4: Git changes (optional)
        if git_path:
            git_has_changes = self._check_git_changes(git_path)
            self.check(f"Git changes detected in {git_path}", git_has_changes)

        # Summary
        print("---")
        print(f"Results: {self.passed} passed, {self.failed} failed")

        return 0 if self.failed == 0 else 1

    def _check_git_changes(self, git_path: str) -> bool:
        """Check if there are git changes in the specified path."""
        try:
            result = subprocess.run(
                ["git", "status", "--short"],
                capture_output=True,
                text=True,
                check=False,
            )
            return git_path in result.stdout
        except Exception:
            return False


def main() -> int:
    """Main entry point."""
    enable_windows_ansi()

    parser = argparse.ArgumentParser(
        description="Verify agent deliverables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python verify.py analysis codebase-structure 2025-01-02
  python verify.py review pr-123 2025-01-02 src/
        """,
    )
    parser.add_argument("category", help="Report category (e.g., analysis, review)")
    parser.add_argument("name", help="Report name")
    parser.add_argument("date", help="Report date (YYYY-MM-DD)")
    parser.add_argument(
        "git_path", nargs="?", help="Optional path to check for git changes"
    )

    args = parser.parse_args()

    verifier = Verifier()
    return verifier.verify(
        category=args.category, name=args.name, date=args.date, git_path=args.git_path
    )


if __name__ == "__main__":
    sys.exit(main())
