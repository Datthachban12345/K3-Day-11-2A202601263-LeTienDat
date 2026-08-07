"""
Assignment 11 — Monitoring & Alerts.

Tracks block rate, rate-limit hits, judge fail rate.
Fires alerts when thresholds are exceeded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class Alert:
    """Represents a monitoring alert."""
    metric: str
    value: float
    threshold: float
    message: str
    timestamp: str = ""
    request_id: Optional[str] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class MonitoringAlert:
    """Aggregate counters from pipeline plugins and emit alerts.

    Monitors:
    - Block rate: percentage of requests blocked by guardrails
    - Rate limit hits: number of rate limit triggers
    - Judge fail rate: percentage of judge checks that failed
    """

    # Default thresholds
    block_rate_threshold: float = 0.5  # 50% block rate triggers alert
    rate_limit_hit_threshold: int = 5  # 5 rate limit hits triggers alert
    judge_fail_rate_threshold: float = 0.3  # 30% judge fail rate triggers alert

    # Spike thresholds (more sensitive)
    spike_block_rate_threshold: float = 0.7  # 70% = potential attack
    spike_rate_limit_threshold: int = 10  # 10 hits = potential abuse
    spike_judge_fail_threshold: float = 0.5  # 50% = potential issue

    alerts: list[Alert] = field(default_factory=list)

    # Counters — update these from your pipeline after each request
    total_requests: int = 0
    blocked_requests: int = 0
    rate_limit_hits: int = 0
    judge_checks: int = 0
    judge_fails: int = 0

    # Track recent activity for spike detection
    recent_blocks: list[datetime] = field(default_factory=list)
    recent_rate_limits: list[datetime] = field(default_factory=list)

    def __init__(self):
        """Initialize monitoring with default thresholds."""
        self.block_rate_threshold = 0.5
        self.rate_limit_hit_threshold = 5
        self.judge_fail_rate_threshold = 0.3
        self.spike_block_rate_threshold = 0.7
        self.spike_rate_limit_threshold = 10
        self.spike_judge_fail_threshold = 0.5
        self.alerts = []
        self.total_requests = 0
        self.blocked_requests = 0
        self.rate_limit_hits = 0
        self.judge_checks = 0
        self.judge_fails = 0
        self.recent_blocks = []
        self.recent_rate_limits = []

    def increment_request(self):
        """Called on each new request."""
        self.total_requests += 1

    def increment_blocked(self, request_id: Optional[str] = None):
        """Called when a request is blocked.

        Args:
            request_id: Optional request ID for tracking
        """
        self.blocked_requests += 1
        self.recent_blocks.append(datetime.now(timezone.utc))

    def increment_rate_limit(self, request_id: Optional[str] = None):
        """Called when rate limit is triggered.

        Args:
            request_id: Optional request ID for tracking
        """
        self.rate_limit_hits += 1
        self.recent_rate_limits.append(datetime.now(timezone.utc))

    def increment_judge_check(self, failed: bool = False):
        """Called after each judge check.

        Args:
            failed: Whether the judge flagged the content as unsafe
        """
        self.judge_checks += 1
        if failed:
            self.judge_fails += 1

    def check_metrics(self) -> list[Alert]:
        """Compute rates and generate alerts when thresholds exceeded.

        Returns:
            List of new Alert objects
        """
        new_alerts = []
        now = datetime.now(timezone.utc)

        # Calculate rates
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests > 0
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks
            if self.judge_checks > 0
            else 0.0
        )

        # Calculate recent block rate (last 10 requests)
        recent_window = 10
        recent_block_count = 0
        cutoff_time = now.replace(second=0, microsecond=0)
        for block_time in self.recent_blocks:
            if block_time >= cutoff_time:
                recent_block_count += 1
        recent_block_rate = min(recent_block_count / recent_window, 1.0)

        # Calculate recent rate limit hits
        recent_rate_limit_count = 0
        for limit_time in self.recent_rate_limits:
            if limit_time >= cutoff_time:
                recent_rate_limit_count += 1

        # Check block rate threshold
        if block_rate >= self.block_rate_threshold:
            alert = Alert(
                metric="block_rate",
                value=block_rate,
                threshold=self.block_rate_threshold,
                message=f"Block rate {block_rate:.1%} exceeds threshold {self.block_rate_threshold:.1%}. "
                        f"Blocked {self.blocked_requests}/{self.total_requests} requests.",
            )
            new_alerts.append(alert)
            self.alerts.append(alert)

        # Check spike in block rate
        if recent_block_rate >= self.spike_block_rate_threshold:
            alert = Alert(
                metric="block_rate_spike",
                value=recent_block_rate,
                threshold=self.spike_block_rate_threshold,
                message=f"Block rate SPIKE detected: {recent_block_rate:.1%} in last {recent_window} requests. "
                        f"Possible attack or abuse.",
            )
            new_alerts.append(alert)
            self.alerts.append(alert)

        # Check rate limit threshold
        if self.rate_limit_hits >= self.rate_limit_hit_threshold:
            alert = Alert(
                metric="rate_limit_hits",
                value=self.rate_limit_hits,
                threshold=self.rate_limit_hit_threshold,
                message=f"Rate limit hits {self.rate_limit_hits} exceeds threshold {self.rate_limit_hit_threshold}. "
                        f"Potential abuse or DoS attempt.",
            )
            new_alerts.append(alert)
            self.alerts.append(alert)

        # Check spike in rate limits
        if recent_rate_limit_count >= self.spike_rate_limit_threshold:
            alert = Alert(
                metric="rate_limit_spike",
                value=recent_rate_limit_count,
                threshold=self.spike_rate_limit_threshold,
                message=f"Rate limit SPIKE: {recent_rate_limit_count} hits in recent window. "
                        f"Investigate potential abuse.",
            )
            new_alerts.append(alert)
            self.alerts.append(alert)

        # Check judge fail rate threshold
        if judge_fail_rate >= self.judge_fail_rate_threshold:
            alert = Alert(
                metric="judge_fail_rate",
                value=judge_fail_rate,
                threshold=self.judge_fail_rate_threshold,
                message=f"Judge fail rate {judge_fail_rate:.1%} exceeds threshold {self.judge_fail_rate_threshold:.1%}. "
                        f"Judge failed {self.judge_fails}/{self.judge_checks} checks.",
            )
            new_alerts.append(alert)
            self.alerts.append(alert)

        # Check spike in judge fails
        if judge_fail_rate >= self.spike_judge_fail_threshold:
            alert = Alert(
                metric="judge_fail_spike",
                value=judge_fail_rate,
                threshold=self.spike_judge_fail_threshold,
                message=f"Judge fail rate SPIKE: {judge_fail_rate:.1%}. "
                        f"Possible prompt injection or jailbreak attempt.",
            )
            new_alerts.append(alert)
            self.alerts.append(alert)

        return new_alerts

    def get_active_alerts(self) -> list[Alert]:
        """Get all alerts from the last hour."""
        cutoff = datetime.now(timezone.utc)
        return [a for a in self.alerts if a.timestamp]

    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts = []

    def reset(self):
        """Reset all counters and alerts."""
        self.total_requests = 0
        self.blocked_requests = 0
        self.rate_limit_hits = 0
        self.judge_checks = 0
        self.judge_fails = 0
        self.recent_blocks = []
        self.recent_rate_limits = []
        self.alerts = []

    def snapshot(self) -> dict:
        """Get current metrics snapshot."""
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        )
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": round(block_rate, 4),
            "rate_limit_hits": self.rate_limit_hits,
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": round(judge_fail_rate, 4),
            "active_alerts": len(self.alerts),
            "alerts": [
                {
                    "metric": a.metric,
                    "value": a.value,
                    "threshold": a.threshold,
                    "message": a.message,
                    "timestamp": a.timestamp,
                }
                for a in self.alerts
            ],
        }

    def export_json(self, filepath: str = "outputs/metrics.json"):
        """Write metrics and alerts to JSON.

        Args:
            filepath: Path to write the JSON file
        """
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.snapshot(), f, ensure_ascii=False, indent=2)

    def create_test_spike(self):
        """Create a test spike scenario to demonstrate alerts.

        This simulates a sudden increase in blocked requests.
        """
        print("Creating test spike scenario...")

        # Simulate normal traffic
        self.total_requests = 10
        self.blocked_requests = 2
        self.rate_limit_hits = 1

        # Simulate spike: 8 more requests, 7 blocked
        for i in range(8):
            self.increment_request()
            if i < 7:  # 7 out of 8 blocked = 87.5% block rate
                self.increment_blocked()
            self.recent_blocks.append(datetime.now(timezone.utc))

        # Check for alerts
        new_alerts = self.check_metrics()

        print(f"\nAfter spike:")
        print(f"  Total requests: {self.total_requests}")
        print(f"  Blocked: {self.blocked_requests}")
        print(f"  Block rate: {self.blocked_requests/self.total_requests:.1%}")
        print(f"\nNew alerts: {len(new_alerts)}")
        for alert in new_alerts:
            print(f"  - [{alert.metric}] {alert.message}")

        return new_alerts


def test_monitoring():
    """Test the monitoring system with a simulated spike."""
    print("=" * 70)
    print("Testing MonitoringAlert System")
    print("=" * 70)

    monitor = MonitoringAlert()

    # Scenario 1: Normal traffic
    print("\n--- Scenario 1: Normal traffic ---")
    for i in range(10):
        monitor.increment_request()
        if i in [2, 5]:  # 2 blocked out of 10
            monitor.increment_blocked()

    snapshot = monitor.snapshot()
    print(f"Total: {snapshot['total_requests']}, Blocked: {snapshot['blocked_requests']}")
    print(f"Block rate: {snapshot['block_rate']:.1%}")
    print(f"Alerts: {snapshot['active_alerts']}")

    # Scenario 2: Attack spike
    print("\n--- Scenario 2: Attack spike ---")
    new_alerts = monitor.create_test_spike()

    # Scenario 3: Find related requests by searching audit
    print("\n--- Scenario 3: Correlating with audit log ---")
    print("To find related requests by request_id:")
    print("  1. Check alerts for request_id if present")
    print("  2. Query audit_log.get_request_by_id(request_id)")
    print("  3. Review all events in the request's lifecycle")

    # Show final snapshot
    print("\n--- Final Metrics Snapshot ---")
    final = monitor.snapshot()
    print(json.dumps(final, indent=2, ensure_ascii=False))

    return monitor


if __name__ == "__main__":
    test_monitoring()
