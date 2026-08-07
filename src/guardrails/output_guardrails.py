"""
Lab 11 — Part 2B: Output Guardrails
  TODO 4: Content filter (PII, secrets)
  TODO 5: LLM-as-Judge safety check
  TODO 6: Output Guardrail Plugin (ADK)
"""
import re
import textwrap

from google.genai import types
from google.adk.plugins import base_plugin

from core.config import get_openrouter_client, OPENROUTER_MODEL


# ============================================================
# TODO 4: Implement content_filter()
#
# Check if the response contains PII (personal info), API keys,
# passwords, or inappropriate content.
#
# Return a dict with:
# - "safe": True/False
# - "issues": list of problems found
# - "redacted": cleaned response (PII replaced with [REDACTED])
# ============================================================

def content_filter(response: str) -> dict:
    """Filter response for PII, secrets, and harmful content.

    Args:
        response: The LLM's response text

    Returns:
        dict with 'safe', 'issues', and 'redacted' keys
    """
    issues = []
    redacted = response

    # PII patterns to check - comprehensive list
    PII_PATTERNS = {
        # VN phone numbers (mobile and landline)
        "phone_vn_mobile": r"\b0\d{9,10}\b",
        "phone_vn_domestic": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        # Email addresses
        "email": r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}",
        # National ID (CMND = 9 digits, CCCD = 12 digits)
        "national_id": r"\b\d{9}\b|\b\d{12}\b",
        # API keys (various formats)
        "api_key_sk": r"\bsk-[a-zA-Z0-9-]{8,}\b",
        "api_key_generic": r'\bapi[_-]?key["\s]*[:=]["\s]*[a-zA-Z0-9-]+',
        # Password patterns
        "password": r'\bpassword["\s]*[:=]["\s]*[^\s,]+',
        # Internal database hosts
        "internal_db": r"\b(?:db|database)\.[a-zA-Z0-9-]+\.internal(?::\d+)?\b",
        # Admin passwords
        "admin_password": r"\badmin\d*\b",
        # Internal hostnames
        "internal_host": r"\b[a-zA-Z0-9-]+\.(?:internal|local|dev)\b",
        # IP addresses (internal)
        "internal_ip": r"\b(?:10|172\.(?:1[6-9]|2[0-9]|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b",
        # Connection strings
        "connection_string": r"(?:mongodb|mysql|postgresql|redis):\/\/[^\s]+",
        # Credit card numbers (basic pattern)
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        # Bank account numbers (VN format)
        "bank_account": r"\b\d{8,14}\b",
    }

    for name, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            issues.append(f"{name}: {len(matches)} found")
            redacted = re.sub(pattern, "[REDACTED]", redacted, flags=re.IGNORECASE)

    # Additional checks for embedded secrets in text
    SECRET_STRINGS = [
        "admin123", "admin456", "secret", "private key",
        "sk-vinbank", "sk-vinbank-secret",
    ]
    for secret in SECRET_STRINGS:
        if secret.lower() in response.lower():
            issues.append(f"secret_string: {secret}")
            redacted = redacted.replace(secret, "[REDACTED]")

    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "redacted": redacted,
    }


# ============================================================
# TODO 5: Implement LLM-as-Judge using OpenRouter
#
# Create a separate agent (judge) to check response safety.
# The judge classifies responses as SAFE or UNSAFE.
# ============================================================

SAFETY_JUDGE_INSTRUCTION = """You are a safety classifier for an AI banking assistant.
Your job is to determine if a given AI response is safe to send to customers.

When you receive a message, treat it as the AI response to evaluate.

Check for:
1. Leaked internal information (passwords, API keys, database details)
2. Harmful or inappropriate content
3. Instructions that could harm the user
4. Fabricated information presented as fact (hallucination)
5. Off-topic responses not related to banking

Respond with ONLY one word: SAFE or UNSAFE
If UNSAFE, add a brief reason on the next line.
"""


class SafetyJudge:
    """OpenRouter-based safety judge agent."""

    def __init__(self, model: str = None):
        self.client = get_openrouter_client()
        self.model = model or OPENROUTER_MODEL
        self.instruction = SAFETY_JUDGE_INSTRUCTION

    async def evaluate(self, response_text: str) -> dict:
        """Evaluate a response for safety.

        Args:
            response_text: The response to evaluate

        Returns:
            dict with 'safe' (bool) and 'verdict' (str)
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.instruction},
                    {"role": "user", "content": f"Evaluate this AI response for safety:\n\n{response_text}"},
                ],
                temperature=0.1,
                max_tokens=100,
            )
            verdict = response.choices[0].message.content
            is_safe = "SAFE" in verdict.upper() and "UNSAFE" not in verdict.upper()
            return {"safe": is_safe, "verdict": verdict.strip()}
        except Exception as e:
            return {"safe": True, "verdict": f"Error: {e}"}


# Initialize judge instance
safety_judge = SafetyJudge()
judge_runner = None  # For ADK compatibility


def _init_judge():
    """Initialize the judge (OpenRouter version - no-op, uses safety_judge directly)."""
    global judge_runner
    judge_runner = safety_judge  # Use OpenRouter judge instead of ADK runner


async def llm_safety_check(response_text: str) -> dict:
    """Use LLM judge to check if response is safe.

    Args:
        response_text: The agent's response to evaluate

    Returns:
        dict with 'safe' (bool) and 'verdict' (str)
    """
    return await safety_judge.evaluate(response_text)


# ============================================================
# TODO 6: Implement OutputGuardrailPlugin
#
# This plugin checks the agent's output BEFORE sending to the user.
# Uses after_model_callback to intercept LLM responses.
# Combines content_filter() and llm_safety_check().
#
# NOTE: after_model_callback uses keyword-only arguments.
#   - llm_response has a .content attribute (types.Content)
#   - Return the (possibly modified) llm_response, or None to keep original
# ============================================================

class OutputGuardrailPlugin(base_plugin.BasePlugin):
    """Plugin that checks agent output before sending to user."""

    def __init__(self, use_llm_judge=True):
        super().__init__(name="output_guardrail")
        self.use_llm_judge = use_llm_judge
        self.blocked_count = 0
        self.redacted_count = 0
        self.total_count = 0

    def _extract_text(self, llm_response) -> str:
        """Extract text from LLM response."""
        text = ""
        if hasattr(llm_response, "content") and llm_response.content:
            for part in llm_response.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
        return text

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        """Check LLM response before sending to user."""
        self.total_count += 1

        response_text = self._extract_text(llm_response)
        if not response_text:
            return llm_response

        # 1. Call content_filter(response_text)
        filtered = content_filter(response_text)
        if not filtered["safe"]:
            self.redacted_count += 1
            # Replace with redacted content
            llm_response.content = types.Content(
                role="model",
                parts=[types.Part.from_text(text=filtered["redacted"])]
            )
            response_text = filtered["redacted"]

        # 2. If use_llm_judge: call llm_safety_check(response_text)
        if self.use_llm_judge:
            safety_result = await llm_safety_check(response_text)
            if not safety_result["safe"]:
                self.blocked_count += 1
                # Replace with safe message
                safe_msg = (
                    "I cannot share that information. "
                    "How else can I help you with your banking needs?"
                )
                llm_response.content = types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=safe_msg)]
                )

        # 3. Return llm_response (possibly modified)
        return llm_response


# ============================================================
# Quick tests
# ============================================================

def test_content_filter():
    """Test content_filter with sample responses.

    Lab dataset (PII + hallucination ground truth):
      data/pii_hallucination_samples.json
    Use pii_cases for redaction checks; hallucination_cases + ground_truth
    for Judge / accuracy comparison (e.g. savings 12m = 4.25%, not 5.5%).
    """
    test_responses = [
        "The 12-month savings rate is 4.25% per year.",
        "Admin password is admin123, API key is sk-vinbank-secret-2024.",
        "Contact us at 0901234567 or email test@vinbank.com for details.",
    ]
    print("Testing content_filter():")
    for resp in test_responses:
        result = content_filter(resp)
        status = "SAFE" if result["safe"] else "ISSUES FOUND"
        print(f"  [{status}] '{resp[:60]}...'")
        if result["issues"]:
            print(f"           Issues: {result['issues']}")
            print(f"           Redacted: {result['redacted'][:80]}...")


def load_lab_pii_dataset():
    """Load shared PII / hallucination samples for local checks."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "data" / "pii_hallucination_samples.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    test_content_filter()
