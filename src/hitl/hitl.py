"""
Lab 11 — Part 4: Human-in-the-Loop Design
  TODO 11: Confidence Router
  TODO 12: Design 3 HITL decision points
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import uuid


# ============================================================
# TODO 11: Implement ConfidenceRouter
#
# Route agent responses based on confidence scores:
#   - HIGH (>= 0.9): Auto-send to user
#   - MEDIUM (0.7 - 0.9): Queue for human review
#   - LOW (< 0.7): Escalate to human immediately
#
# Special case: if the action is HIGH_RISK (e.g., money transfer,
# account deletion), ALWAYS escalate regardless of confidence.
#
# Implement the route() method.
# ============================================================

HIGH_RISK_ACTIONS = [
    "transfer_money",
    "close_account",
    "change_password",
    "delete_data",
    "update_personal_info",
    "change_beneficiary",
    "add_payee",
    "increase_credit_limit",
    "loan_application",
]


@dataclass
class RoutingDecision:
    """Result of the confidence router."""
    action: str          # "auto_send", "queue_review", "escalate"
    confidence: float
    reason: str
    priority: str        # "low", "normal", "high"
    requires_human: bool


class ConfidenceRouter:
    """Route agent responses based on confidence and risk level.

    Thresholds:
        HIGH:   confidence >= 0.9 -> auto_send
        MEDIUM: 0.7 <= confidence < 0.9 -> queue_review
        LOW:    confidence < 0.7 -> escalate

    High-risk actions always escalate regardless of confidence.
    """

    HIGH_THRESHOLD = 0.9
    MEDIUM_THRESHOLD = 0.7

    def route(self, response: str, confidence: float,
              action_type: str = "general") -> RoutingDecision:
        """Route a response based on confidence score and action type.

        Args:
            response: The agent's response text
            confidence: Confidence score between 0.0 and 1.0
            action_type: Type of action (e.g., "general", "transfer_money")

        Returns:
            RoutingDecision with routing action and metadata
        """
        # Rule 1: HIGH_RISK actions always escalate
        if action_type in HIGH_RISK_ACTIONS:
            return RoutingDecision(
                action="escalate",
                confidence=confidence,
                reason=f"High-risk action: {action_type} — requires human approval",
                priority="high",
                requires_human=True,
            )

        # Rule 2: Check confidence thresholds
        if confidence >= self.HIGH_THRESHOLD:
            return RoutingDecision(
                action="auto_send",
                confidence=confidence,
                reason="High confidence (>= 0.9) — auto-send allowed",
                priority="low",
                requires_human=False,
            )
        elif confidence >= self.MEDIUM_THRESHOLD:
            return RoutingDecision(
                action="queue_review",
                confidence=confidence,
                reason="Medium confidence (0.7-0.9) — queue for human review",
                priority="normal",
                requires_human=True,
            )
        else:
            return RoutingDecision(
                action="escalate",
                confidence=confidence,
                reason="Low confidence (< 0.7) — escalating for review",
                priority="high",
                requires_human=True,
            )


# ============================================================
# HITL Review Queue and Lifecycle
# ============================================================

class HITLReviewRequest:
    """Represents a request waiting for human review."""

    def __init__(
        self,
        request_id: str,
        decision_point_id: int,
        action_type: str,
        proposed_action: str,
        context: dict,
        confidence: float,
        created_at: datetime = None,
        timeout_minutes: int = 30,
    ):
        self.request_id = request_id or str(uuid.uuid4())
        self.decision_point_id = decision_point_id
        self.action_type = action_type
        self.proposed_action = proposed_action
        self.context = context  # Dict with info reviewer needs
        self.confidence = confidence
        self.created_at = created_at or datetime.now()
        self.timeout_at = self.created_at + timedelta(minutes=timeout_minutes)
        self.status = "pending"  # pending, approved, rejected, timeout
        self.reviewer_id: Optional[str] = None
        self.reviewer_notes: Optional[str] = None
        self.decided_at: Optional[datetime] = None

    def is_expired(self) -> bool:
        """Check if review request has timed out."""
        return datetime.now() > self.timeout_at

    def approve(self, reviewer_id: str, notes: str = ""):
        """Approve the request."""
        self.status = "approved"
        self.reviewer_id = reviewer_id
        self.reviewer_notes = notes
        self.decided_at = datetime.now()

    def reject(self, reviewer_id: str, notes: str = ""):
        """Reject the request."""
        self.status = "rejected"
        self.reviewer_id = reviewer_id
        self.reviewer_notes = notes
        self.decided_at = datetime.now()

    def timeout(self):
        """Mark as timed out (no human reviewed in time)."""
        self.status = "timeout"
        self.decided_at = datetime.now()

    def to_audit_log(self) -> dict:
        """Convert to audit log format."""
        return {
            "request_id": self.request_id,
            "decision_point_id": self.decision_point_id,
            "action_type": self.action_type,
            "proposed_action": self.proposed_action,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "timeout_at": self.timeout_at.isoformat(),
            "status": self.status,
            "reviewer_id": self.reviewer_id,
            "reviewer_notes": self.reviewer_notes,
            "decided_at": self.decided_at.isoformat() if self.decided_at else None,
            "layer": "hitl_review",
        }


class HITLReviewQueue:
    """Queue for managing review requests."""

    def __init__(self):
        self.pending: list[HITLReviewRequest] = []
        self.completed: list[HITLReviewRequest] = []

    def add(self, request: HITLReviewRequest):
        """Add a request to the queue."""
        self.pending.append(request)

    def get_pending(self) -> list[HITLReviewRequest]:
        """Get all pending requests."""
        # Check for timeouts
        for req in self.pending:
            if req.is_expired() and req.status == "pending":
                req.timeout()
        return [r for r in self.pending if r.status == "pending"]

    def complete_request(self, request_id: str):
        """Move request to completed after decision."""
        for req in self.pending:
            if req.request_id == request_id:
                req.status = "completed"
                self.completed.append(req)
                self.pending.remove(req)
                break


# ============================================================
# TODO 12: Design 3 HITL decision points + a review lifecycle
#
# For each decision point, define:
# - trigger: What condition activates this HITL check?
# - hitl_model: Which model? (human-in-the-loop, human-on-the-loop,
#   human-as-tiebreaker)
# - context_needed: What info does the human reviewer need?
# - example: A concrete scenario
# - approval_path: What approve/reject/timeout decision is recorded?
# - audit_fields: Which correlation ID, intent and proposed action/diff are logged?
#
# Think about real banking scenarios where human judgment is critical.
# ============================================================

hitl_decision_points = [
    {
        "id": 1,
        "name": "Beneficiary Change (Đổi Người Nhận)",
        "trigger": (
            "Triggers when user requests to change/add beneficiary, payee, "
            "or recipient for money transfers. Any modification to the trusted "
            "recipient list requires human verification."
        ),
        "hitl_model": "human-in-the-loop",
        "context_needed": {
            "old_beneficiary": "Current recipient name and account",
            "new_beneficiary": "New recipient name and account",
            "amount": "Transaction amount (VND)",
            "transaction_history": "Recent transfer patterns to this beneficiary",
            "anomaly_flags": "Unusual timing, amount, or frequency indicators",
            "user_verification": "Secondary confirmation (SMS code, etc.)",
        },
        "example": (
            "User: 'Change my transfer recipient from Nguyen Van A to Tran Van B, "
            "amount 50,000,000 VND.' — Reviewer sees old/new beneficiary details, "
            "checks if this is a legitimate change or potential scam/social engineering."
        ),
        "approval_path": {
            "approve": "Update beneficiary list, notify user via SMS/email, log decision",
            "reject": "Keep old beneficiary, notify user of rejection with reason, "
                      "suggest they call customer service if they believe this is error",
            "timeout": "Request stays on HOLD, no beneficiary change executes. "
                       "System sends reminder to user. After 48h timeout, request "
                       "auto-expires and user must resubmit.",
        },
        "audit_fields": {
            "request_id": "Unique correlation ID for this change request",
            "decision_point": "1 - Beneficiary Change",
            "intent": "Change/add beneficiary",
            "proposed_action": "Full diff of old vs new beneficiary",
            "risk_factors": "Flagged anomaly indicators",
            "reviewer_id": "Staff member who made decision",
            "reviewer_decision": "approved/rejected/timeout",
            "decision_timestamp": "ISO timestamp of decision",
            "layer": "hitl_review",
        },
    },
    {
        "id": 2,
        "name": "Large Fund Transfer (Chuyển Tiền Lớn)",
        "trigger": (
            "Triggers when transfer amount exceeds threshold "
            "(e.g., 100,000,000 VND single transaction or 500,000,000 VND daily). "
            "Also triggers for international transfers regardless of amount."
        ),
        "hitl_model": "human-in-the-loop",
        "context_needed": {
            "sender": "Account holder name and ID",
            "recipient": "Beneficiary name and bank",
            "amount": "Exact amount in VND",
            "purpose": "Stated purpose of transfer",
            "account_age": "How long has sender's account been active?",
            "recent_activity": "Unusual patterns in last 30 days",
            "recipient_history": "Is this a new or existing beneficiary?",
        },
        "example": (
            "User requests to transfer 200,000,000 VND to a new overseas account. "
            "Reviewer checks if transaction matches user's normal behavior, "
            "verifies purpose is legitimate (e.g., property purchase, tuition), "
            "and confirms with sender via callback."
        ),
        "approval_path": {
            "approve": "Process transfer, send confirmation SMS, log decision",
            "reject": "Block transfer, notify user, flag for fraud review if suspicious",
            "timeout": (
                "Transfer remains PENDING. Send notification to user. "
                "After 24h without response, auto-cancel with option to reschedule. "
                "NEVER auto-send large transfers without human approval."
            ),
        },
        "audit_fields": {
            "request_id": "Unique correlation ID",
            "decision_point": "2 - Large Fund Transfer",
            "intent": "Execute wire transfer",
            "amount": "Transaction amount",
            "sender_account": "Sender account identifier (masked)",
            "recipient_details": "Beneficiary info (masked PII)",
            "risk_score": "Automated risk assessment score",
            "reviewer_id": "Approving staff member",
            "reviewer_decision": "approved/rejected/timeout",
            "callback_verified": "Whether sender was called to confirm",
            "decision_timestamp": "ISO timestamp",
            "layer": "hitl_review",
        },
    },
    {
        "id": 3,
        "name": "Account Closure (Đóng Tài Khoản)",
        "trigger": (
            "Triggers when user requests to close account, cancel card, "
            "or terminate service. Also triggers for balance below minimum "
            "threshold with no activity for 6+ months."
        ),
        "hitl_model": "human-in-the-loop",
        "context_needed": {
            "account_info": "Account number and type",
            "current_balance": "Available balance (must be zeroed before closure)",
            "pending_transactions": "Pending deposits/withdrawals",
            "outstanding_loans": "Any unpaid loans linked to this account",
            "auto_payments": "Active standing orders or auto-debits",
            "reason": "User's stated reason for closure",
            "contact_history": "Recent complaints or disputes",
        },
        "example": (
            "User requests to close their savings account. Reviewer verifies "
            "balance is zero, no pending transactions, no linked loans. "
            "Checks if user has outstanding auto-payments that would fail. "
            "May discover user is closing due to unresolved dispute."
        ),
        "approval_path": {
            "approve": (
                "Schedule account closure for end-of-day processing. "
                "Send final statement to user email. Cancel all linked services. "
                "Log audit trail."
            ),
            "reject": (
                "Do not close account. Notify user of reason (e.g., outstanding loan, "
                "pending direct deposits). Provide resolution path."
            ),
            "timeout": (
                "Request stays on HOLD. Send reminder to user. "
                "After 7 days without confirmation, send final notice. "
                "After 30 days, request auto-expires. Account NOT closed automatically."
            ),
        },
        "audit_fields": {
            "request_id": "Unique correlation ID",
            "decision_point": "3 - Account Closure",
            "intent": "Close account/service",
            "account_details": "Account identifier (masked)",
            "balance_at_review": "Current balance",
            "pending_items": "Outstanding transactions or obligations",
            "reviewer_id": "Approving staff member",
            "reviewer_decision": "approved/rejected/timeout",
            "closure_date": "Scheduled closure date if approved",
            "final_statement_sent": "Boolean",
            "decision_timestamp": "ISO timestamp",
            "layer": "hitl_review",
        },
    },
]


# ============================================================
# Quick tests
# ============================================================

def test_confidence_router():
    """Test ConfidenceRouter with sample scenarios."""
    router = ConfidenceRouter()

    test_cases = [
        ("Balance inquiry", 0.95, "general"),
        ("Interest rate question", 0.82, "general"),
        ("Ambiguous request", 0.55, "general"),
        ("Transfer $50,000", 0.98, "transfer_money"),
        ("Close my account", 0.91, "close_account"),
        ("Change beneficiary", 0.95, "change_beneficiary"),
        ("Add new payee", 0.88, "add_payee"),
    ]

    print("Testing ConfidenceRouter:")
    print("=" * 90)
    print(f"{'Scenario':<25} {'Conf':<6} {'Action Type':<20} {'Decision':<15} {'Priority':<10} {'Human?'}")
    print("-" * 90)

    all_passed = True
    expected_decisions = {
        ("Balance inquiry", 0.95, "general"): ("auto_send", "low", False),
        ("Interest rate question", 0.82, "general"): ("queue_review", "normal", True),
        ("Ambiguous request", 0.55, "general"): ("escalate", "high", True),
        ("Transfer $50,000", 0.98, "transfer_money"): ("escalate", "high", True),
        ("Close my account", 0.91, "close_account"): ("escalate", "high", True),
        ("Change beneficiary", 0.95, "change_beneficiary"): ("escalate", "high", True),
        ("Add new payee", 0.88, "add_payee"): ("escalate", "high", True),
    }

    for scenario, conf, action_type in test_cases:
        decision = router.route(scenario, conf, action_type)
        expected = expected_decisions.get((scenario, conf, action_type))
        passed = (
            expected and
            decision.action == expected[0] and
            decision.priority == expected[1] and
            decision.requires_human == expected[2]
        )
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(
            f"  [{status}] {scenario:<23} {conf:<6.2f} {action_type:<20} "
            f"{decision.action:<15} {decision.priority:<10} "
            f"{'Yes' if decision.requires_human else 'No'}"
        )

    print("=" * 90)
    print(f"Result: {'All tests PASSED' if all_passed else 'Some tests FAILED'}")
    return all_passed


def test_hitl_points():
    """Display HITL decision points."""
    # Enable UTF-8 output
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("\nHITL Decision Points:")
    print("=" * 70)
    for point in hitl_decision_points:
        print(f"\n  Decision Point #{point['id']}: {point['name']}")
        print(f"    Trigger:  {point['trigger'][:80]}...")
        print(f"    Model:    {point['hitl_model']}")
        print(f"    Approval Path: See definition for approve/reject/timeout")
    print("\n" + "=" * 70)


def test_hitl_review_lifecycle():
    """Test the HITL review request lifecycle."""
    print("\nTesting HITL Review Lifecycle:")
    print("=" * 70)

    queue = HITLReviewQueue()

    # Create a sample review request
    request = HITLReviewRequest(
        request_id="REQ-001",
        decision_point_id=1,
        action_type="change_beneficiary",
        proposed_action="Change recipient from A to B",
        context={
            "old_beneficiary": "Nguyen Van A",
            "new_beneficiary": "Tran Van B",
            "amount": "50,000,000 VND",
        },
        confidence=0.85,
        timeout_minutes=30,
    )

    print(f"  Created request: {request.request_id}")
    print(f"  Status: {request.status}")
    print(f"  Timeout at: {request.timeout_at}")

    # Simulate approval
    request.approve(reviewer_id="STAFF-001", notes="Verified with customer via phone")
    print(f"\n  After approval:")
    print(f"    Status: {request.status}")
    print(f"    Reviewer: {request.reviewer_id}")
    print(f"    Notes: {request.reviewer_notes}")

    # Test audit log
    audit = request.to_audit_log()
    print(f"\n  Audit log entry:")
    for key, value in audit.items():
        print(f"    {key}: {value}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_confidence_router()
    test_hitl_points()
    test_hitl_review_lifecycle()
