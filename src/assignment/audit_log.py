"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline).

    Tracks requests from input to output with consistent request_id.
    Records layer decisions, timestamps, and reviewer actions.
    """

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open_requests: dict[str, dict] = {}  # request_id -> request data

    def _utc_now_iso(self) -> str:
        """Get current UTC time in ISO format."""
        return datetime.now(timezone.utc).isoformat()

    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        return f"req-{uuid.uuid4().hex[:12]}"

    def record_input(
        self,
        *,
        user_id: str,
        text: str,
        request_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        """Record input request and start tracking.

        Args:
            user_id: User identifier
            text: User's input text
            request_id: Optional request ID (generated if not provided)
            metadata: Additional metadata (e.g., session_id, ip_address)
        """
        request_id = request_id or self._generate_request_id()
        timestamp = self._utc_now_iso()

        # Store request data for later correlation
        self._open_requests[request_id] = {
            "request_id": request_id,
            "user_id": user_id,
            "input_text": text,
            "input_timestamp": timestamp,
            "metadata": metadata or {},
        }

        # Record input event
        log_entry = {
            "event_type": "input_received",
            "request_id": request_id,
            "user_id": user_id,
            "timestamp": timestamp,
            "layer": "input",
            "text_preview": text[:200] if len(text) > 200 else text,
            "text_length": len(text),
            "metadata": metadata or {},
        }
        self.logs.append(log_entry)

    def record_layer_processing(
        self,
        *,
        request_id: str,
        layer: str,
        decision: str,  # "passed", "blocked", "redacted", "escalated"
        details: Optional[str] = None,
        blocked_by: Optional[str] = None,  # e.g., "injection_detector", "topic_filter"
    ):
        """Record processing at a specific layer.

        Args:
            request_id: The request being processed
            layer: Layer name (e.g., "input_guardrail", "output_guardrail", "hitl")
            decision: Processing decision
            details: Additional details
            blocked_by: What specifically blocked the request
        """
        timestamp = self._utc_now_iso()

        log_entry = {
            "event_type": "layer_processing",
            "request_id": request_id,
            "timestamp": timestamp,
            "layer": layer,
            "decision": decision,
            "details": details,
            "blocked_by": blocked_by,
        }
        self.logs.append(log_entry)

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: Optional[str] = None,
        request_id: Optional[str] = None,
        decision: str = "sent",  # "sent", "blocked", "timeout"
        reviewer_id: Optional[str] = None,
        reviewer_notes: Optional[str] = None,
    ):
        """Record output response and complete the request.

        Args:
            user_id: User identifier
            text: Agent's response text
            blocked: Whether response was blocked
            layer: Which layer blocked (if blocked)
            request_id: Request ID (looks up input if not provided)
            decision: Final decision (sent, blocked, timeout)
            reviewer_id: HITL reviewer ID (if applicable)
            reviewer_notes: Reviewer's notes (if applicable)
        """
        timestamp = self._utc_now_iso()
        request_id = request_id or self._generate_request_id()

        # Get input timestamp if available
        input_timestamp = None
        if request_id in self._open_requests:
            input_timestamp = self._open_requests[request_id].get("input_timestamp")

        # Calculate processing time
        processing_time_ms = None
        if input_timestamp:
            input_dt = datetime.fromisoformat(input_timestamp.replace("Z", "+00:00"))
            output_dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            processing_time_ms = (output_dt - input_dt).total_seconds() * 1000

        # Record output event
        log_entry = {
            "event_type": "output_sent",
            "request_id": request_id,
            "user_id": user_id,
            "timestamp": timestamp,
            "layer": layer,
            "blocked": blocked,
            "decision": decision,
            "text_preview": text[:200] if len(text) > 200 else text,
            "text_length": len(text),
            "processing_time_ms": processing_time_ms,
            "reviewer_id": reviewer_id,
            "reviewer_notes": reviewer_notes,
        }
        self.logs.append(log_entry)

        # Mark request as complete
        if request_id in self._open_requests:
            self._open_requests[request_id]["output_timestamp"] = timestamp
            self._open_requests[request_id]["completed"] = True
            self._open_requests[request_id]["decision"] = decision

    def record_hitl_review(
        self,
        *,
        request_id: str,
        decision_point_id: int,
        decision: str,  # "approved", "rejected", "timeout"
        reviewer_id: Optional[str] = None,
        reviewer_notes: Optional[str] = None,
    ):
        """Record HITL review decision.

        Args:
            request_id: The request under review
            decision_point_id: Which HITL decision point
            decision: Reviewer's decision
            reviewer_id: Staff member who reviewed
            reviewer_notes: Reviewer's notes
        """
        timestamp = self._utc_now_iso()

        log_entry = {
            "event_type": "hitl_review",
            "request_id": request_id,
            "timestamp": timestamp,
            "layer": "hitl_review",
            "decision_point_id": decision_point_id,
            "decision": decision,
            "reviewer_id": reviewer_id,
            "reviewer_notes": reviewer_notes,
        }
        self.logs.append(log_entry)

    def get_request_by_id(self, request_id: str) -> list[dict]:
        """Get all log entries for a specific request_id.

        Args:
            request_id: The request to find

        Returns:
            List of log entries for this request
        """
        return [log for log in self.logs if log.get("request_id") == request_id]

    def get_requests_by_layer(self, layer: str) -> list[dict]:
        """Get all log entries for a specific layer.

        Args:
            layer: Layer name to filter

        Returns:
            List of log entries for this layer
        """
        return [log for log in self.logs if log.get("layer") == layer]

    def get_blocked_requests(self) -> list[dict]:
        """Get all blocked requests."""
        return [
            log for log in self.logs
            if log.get("event_type") == "output_sent" and log.get("blocked")
        ]

    def export_json(self, filepath: str = "outputs/audit_log.json"):
        """Write logs to disk (JSON array).

        Args:
            filepath: Path to write the JSON file
        """
        # Ensure parent directories exist
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write logs as JSON array
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=2)

    def export_csv(self, filepath: str = "outputs/audit_log.csv"):
        """Export logs as CSV for easier analysis.

        Args:
            filepath: Path to write the CSV file
        """
        import csv

        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.logs:
            return

        # Get all unique keys from logs
        fieldnames = list(self.logs[0].keys())

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.logs)

    def summary(self) -> dict:
        """Get summary statistics."""
        total_requests = len(set(
            log.get("request_id") for log in self.logs
            if log.get("request_id")
        ))

        blocked_requests = len(self.get_blocked_requests())

        layer_counts = {}
        for log in self.logs:
            layer = log.get("layer", "unknown")
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

        return {
            "total_requests": total_requests,
            "total_log_entries": len(self.logs),
            "blocked_requests": blocked_requests,
            "block_rate": blocked_requests / total_requests if total_requests > 0 else 0,
            "layer_counts": layer_counts,
        }


def utc_now_iso() -> str:
    """Get current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()
