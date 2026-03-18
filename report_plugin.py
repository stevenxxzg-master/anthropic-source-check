"""
Pytest plugin — generates a complete result report in output-check/ after every run.

Report includes:
- Endpoint info (base URL, model, timestamp)
- Summary scores (passed/failed/skipped/total, pass rate)
- Per-category breakdown (feature compliance, quality, performance)
- Full test results table with status and error details
- Compatibility matrix for proxy vs official API features
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TestResult:
    node_id: str
    outcome: str  # "passed", "failed", "skipped", "error"
    duration: float
    error_message: str
    category: str
    test_class: str
    test_name: str


def _categorize(node_id: str) -> tuple[str, str, str]:
    """Extract (category, class, name) from a pytest node ID."""
    # node_id looks like: test_api_live.py::TestMessagesAPI::test_create_message
    parts = node_id.split("::")
    file_name = parts[0] if len(parts) > 0 else "unknown"
    class_name = parts[1] if len(parts) > 1 else "unknown"
    test_name = parts[2] if len(parts) > 2 else parts[-1]

    category_map = {
        "test_claude_platform_features.py": "L1-2: Feature Catalog",
        "test_api_live.py": "L2: Live API",
        "test_eval_quality.py": "L3: Quality Eval",
        "test_perf_benchmark.py": "L4: Performance",
    }
    category = category_map.get(file_name, file_name)
    return category, class_name, test_name


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

class ReportGenerator:
    def __init__(self):
        self.results: list[TestResult] = []
        self.start_time: float = 0
        self.end_time: float = 0
        self.base_url: str = ""
        self.model: str = ""
        self.api_key_prefix: str = ""

    def add_result(self, node_id: str, outcome: str, duration: float, error_message: str):
        category, test_class, test_name = _categorize(node_id)
        self.results.append(TestResult(
            node_id=node_id,
            outcome=outcome,
            duration=duration,
            error_message=error_message,
            category=category,
            test_class=test_class,
            test_name=test_name,
        ))

    def generate(self) -> str:
        lines: list[str] = []
        now = datetime.now(timezone.utc)
        total_time = self.end_time - self.start_time

        passed = [r for r in self.results if r.outcome == "passed"]
        failed = [r for r in self.results if r.outcome == "failed"]
        skipped = [r for r in self.results if r.outcome == "skipped"]
        errored = [r for r in self.results if r.outcome == "error"]

        total_run = len(passed) + len(failed) + len(errored)
        pass_rate = (len(passed) / total_run * 100) if total_run > 0 else 0

        # Parse hostname for display
        host = urlparse(self.base_url).hostname or self.base_url if self.base_url else "not configured"

        # ── Header ──
        lines.append("=" * 80)
        lines.append("ANTHROPIC API SOURCE CHECK — FULL REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"  Timestamp:    {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(f"  Endpoint:     {self.base_url or 'https://api.anthropic.com'}")
        lines.append(f"  Model:        {self.model or 'default'}")
        lines.append(f"  API Key:      {self.api_key_prefix}...")
        lines.append(f"  Duration:     {total_time:.1f}s")
        lines.append("")

        # ── Summary ──
        lines.append("-" * 80)
        lines.append("SUMMARY")
        lines.append("-" * 80)
        lines.append(f"  Total tests:  {len(self.results)}")
        lines.append(f"  Passed:       {len(passed)}")
        lines.append(f"  Failed:       {len(failed)}")
        lines.append(f"  Skipped:      {len(skipped)}")
        if errored:
            lines.append(f"  Errors:       {len(errored)}")
        lines.append(f"  Pass rate:    {pass_rate:.1f}% ({len(passed)}/{total_run} run)")
        lines.append("")

        # ── Score gauge ──
        gauge_width = 40
        filled = int(pass_rate / 100 * gauge_width)
        bar = "█" * filled + "░" * (gauge_width - filled)
        lines.append(f"  [{bar}] {pass_rate:.1f}%")
        lines.append("")

        # ── Per-category breakdown ──
        lines.append("-" * 80)
        lines.append("BREAKDOWN BY CATEGORY")
        lines.append("-" * 80)
        lines.append(f"  {'Category':<25} {'Passed':>8} {'Failed':>8} {'Skipped':>8} {'Rate':>8}")
        lines.append(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        categories = sorted(set(r.category for r in self.results))
        for cat in categories:
            cat_results = [r for r in self.results if r.category == cat]
            cat_passed = len([r for r in cat_results if r.outcome == "passed"])
            cat_failed = len([r for r in cat_results if r.outcome == "failed"])
            cat_skipped = len([r for r in cat_results if r.outcome == "skipped"])
            cat_run = cat_passed + cat_failed
            cat_rate = f"{cat_passed}/{cat_run}" if cat_run > 0 else "n/a"
            lines.append(f"  {cat:<25} {cat_passed:>8} {cat_failed:>8} {cat_skipped:>8} {cat_rate:>8}")
        lines.append("")

        # ── Per-class breakdown ──
        lines.append("-" * 80)
        lines.append("BREAKDOWN BY TEST CLASS")
        lines.append("-" * 80)

        for cat in categories:
            cat_results = [r for r in self.results if r.category == cat]
            classes = sorted(set(r.test_class for r in cat_results))
            for cls in classes:
                cls_results = [r for r in cat_results if r.test_class == cls]
                cls_passed = len([r for r in cls_results if r.outcome == "passed"])
                cls_failed = len([r for r in cls_results if r.outcome == "failed"])
                cls_skipped = len([r for r in cls_results if r.outcome == "skipped"])
                cls_total = len(cls_results)

                if cls_skipped == cls_total:
                    status = "⊘ SKIP"
                elif cls_failed == 0:
                    status = "✓ PASS"
                elif cls_passed == 0:
                    status = "✗ FAIL"
                else:
                    status = "△ PARTIAL"

                lines.append(f"  {status}  {cls:<40} {cls_passed}/{cls_total - cls_skipped} passed")
        lines.append("")

        # ── Compatibility matrix ──
        lines.append("-" * 80)
        lines.append("FEATURE COMPATIBILITY MATRIX")
        lines.append("-" * 80)
        lines.append(f"  {'Feature':<40} {'Status':<10} {'Details'}")
        lines.append(f"  {'-'*40} {'-'*10} {'-'*30}")

        # Group by test class, show pass/fail
        for cat in categories:
            if cat.startswith("L1-2"):
                continue  # Skip catalog tests in compatibility matrix
            cat_results = [r for r in self.results if r.category == cat]
            classes = sorted(set(r.test_class for r in cat_results))
            for cls in classes:
                cls_results = [r for r in cat_results if r.test_class == cls]
                cls_passed = len([r for r in cls_results if r.outcome == "passed"])
                cls_failed = len([r for r in cls_results if r.outcome == "failed"])
                cls_skipped = len([r for r in cls_results if r.outcome == "skipped"])
                cls_total = len(cls_results)

                if cls_skipped == cls_total:
                    status = "SKIP"
                    detail = "no API key"
                elif cls_failed == 0:
                    status = "PASS"
                    detail = f"{cls_passed}/{cls_passed} tests"
                elif cls_passed == 0:
                    status = "FAIL"
                    # Get first error
                    first_err = next((r.error_message for r in cls_results if r.outcome == "failed"), "")
                    detail = first_err[:50] if first_err else "all tests failed"
                else:
                    status = "PARTIAL"
                    detail = f"{cls_passed}/{cls_passed + cls_failed} tests"

                icon = {"PASS": "✓", "FAIL": "✗", "PARTIAL": "△", "SKIP": "⊘"}.get(status, "?")
                lines.append(f"  {cls:<40} {icon} {status:<8} {detail}")
        lines.append("")

        # ── Failed tests detail ──
        if failed or errored:
            lines.append("-" * 80)
            lines.append("FAILED TESTS — DETAILS")
            lines.append("-" * 80)
            for r in failed + errored:
                lines.append(f"  ✗ {r.node_id}")
                lines.append(f"    Duration: {r.duration:.2f}s")
                if r.error_message:
                    # Truncate long errors but keep useful info
                    err = r.error_message.replace("\n", " ").strip()
                    if len(err) > 200:
                        err = err[:200] + "..."
                    lines.append(f"    Error: {err}")
                lines.append("")

        # ── All tests detail ──
        lines.append("-" * 80)
        lines.append("ALL TESTS — FULL RESULTS")
        lines.append("-" * 80)
        lines.append(f"  {'#':>3}  {'Status':<8} {'Duration':>8}  {'Test'}")
        lines.append(f"  {'-'*3}  {'-'*8} {'-'*8}  {'-'*50}")

        icons = {"passed": "✓", "failed": "✗", "skipped": "⊘", "error": "!"}
        for i, r in enumerate(self.results, 1):
            icon = icons.get(r.outcome, "?")
            status = r.outcome.upper()
            dur = f"{r.duration:.2f}s" if r.duration > 0 else "-"
            lines.append(f"  {i:>3}  {icon} {status:<6} {dur:>8}  {r.node_id}")

        lines.append("")
        lines.append("=" * 80)
        lines.append(f"END OF REPORT — {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append("=" * 80)
        lines.append("")

        return "\n".join(lines)

    def generate_json(self) -> str:
        """Generate a machine-readable JSON report."""
        now = datetime.now(timezone.utc)
        total_run = len([r for r in self.results if r.outcome in ("passed", "failed", "error")])
        passed_count = len([r for r in self.results if r.outcome == "passed"])

        report = {
            "timestamp": now.isoformat(),
            "endpoint": self.base_url or "https://api.anthropic.com",
            "model": self.model or "default",
            "duration_s": round(self.end_time - self.start_time, 2),
            "summary": {
                "total": len(self.results),
                "passed": passed_count,
                "failed": len([r for r in self.results if r.outcome == "failed"]),
                "skipped": len([r for r in self.results if r.outcome == "skipped"]),
                "pass_rate": round(passed_count / total_run * 100, 1) if total_run > 0 else 0,
            },
            "results": [
                {
                    "test": r.node_id,
                    "category": r.category,
                    "class": r.test_class,
                    "name": r.test_name,
                    "outcome": r.outcome,
                    "duration_s": round(r.duration, 3),
                    "error": r.error_message if r.outcome == "failed" else None,
                }
                for r in self.results
            ],
        }
        return json.dumps(report, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Pytest hooks
# ---------------------------------------------------------------------------

_report = ReportGenerator()


def pytest_configure(config):
    """Initialize report with config info."""
    _report.start_time = time.time()


def pytest_sessionstart(session):
    """Capture API config at session start."""
    from config import load_config
    cfg = load_config(
        base_url=session.config.getoption("base_url", default=None),
        api_key=session.config.getoption("api_key", default=None),
        model=session.config.getoption("model", default=None),
    )
    _report.base_url = cfg.base_url
    _report.model = cfg.model
    _report.api_key_prefix = cfg.api_key[:12] if cfg.api_key else "none"


def pytest_runtest_logreport(report):
    """Collect each test result."""
    if report.when == "call" or (report.when == "setup" and report.skipped):
        error_msg = ""
        if report.failed and report.longreprtext:
            # Extract the last line (usually the assertion or exception)
            err_lines = report.longreprtext.strip().split("\n")
            error_msg = err_lines[-1] if err_lines else ""

        outcome = "skipped" if report.skipped else report.outcome
        _report.add_result(
            node_id=report.nodeid,
            outcome=outcome,
            duration=getattr(report, "duration", 0),
            error_message=error_msg,
        )


def pytest_sessionfinish(session, exitstatus):
    """Generate reports after all tests complete."""
    _report.end_time = time.time()

    # Create output directory
    output_dir = Path(__file__).parent / "output-check"
    output_dir.mkdir(exist_ok=True)

    # Generate timestamp-based filename
    now = datetime.now()
    host = urlparse(_report.base_url).hostname or "localhost"
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    base_name = f"{timestamp}_{host}"

    # Write text report
    text_path = output_dir / f"{base_name}.txt"
    text_path.write_text(_report.generate(), encoding="utf-8")

    # Write JSON report
    json_path = output_dir / f"{base_name}.json"
    json_path.write_text(_report.generate_json(), encoding="utf-8")

    # Also write a "latest" symlink/copy
    latest_txt = output_dir / "latest.txt"
    latest_json = output_dir / "latest.json"
    latest_txt.write_text(_report.generate(), encoding="utf-8")
    latest_json.write_text(_report.generate_json(), encoding="utf-8")

    print(f"\n📋 Report saved to: {text_path}")
    print(f"📊 JSON saved to:   {json_path}")
