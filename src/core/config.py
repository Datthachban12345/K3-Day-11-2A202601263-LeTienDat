"""
Lab 11 — Configuration & API Key Setup
"""
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()


def setup_api_key():
    """Load API key from environment or prompt."""
    # Check for OpenRouter first (preferred)
    if os.environ.get("OPENROUTER_API_KEY"):
        print(f"OpenRouter API key loaded (model: {os.environ.get('OPENROUTER_MODEL', 'google/gemini-3.5-flash')})")
        return

    # Fallback to Google API
    if "GOOGLE_API_KEY" not in os.environ or not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = input("Enter Google API Key: ")
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "0"
    print("Google API key loaded.")


# ============ OpenRouter Configuration ============
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemini-3.5-flash")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


# ============ OpenRouter Client (OpenAI-compatible) ============
def get_openrouter_client():
    """Get OpenAI-compatible client for OpenRouter."""
    from openai import OpenAI
    return OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )


async def chat_with_openrouter(system_prompt: str, user_message: str, model: str = None) -> str:
    """Simple chat function using OpenRouter (OpenAI-compatible API).

    Args:
        system_prompt: System instructions for the agent
        user_message: User's message
        model: OpenRouter model ID (defaults to OPENROUTER_MODEL)

    Returns:
        Assistant's response text
    """
    client = get_openrouter_client()
    target_model = model or OPENROUTER_MODEL

    response = client.chat.completions.create(
        model=target_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        max_tokens=1024,
    )

    return response.choices[0].message.content


# Allowed banking topics (used by topic_filter)
ALLOWED_TOPICS = [
    "banking", "account", "transaction", "transfer",
    "loan", "interest", "savings", "credit",
    "deposit", "withdrawal", "balance", "payment",
    "tai khoan", "giao dich", "tiet kiem", "lai suat",
    "chuyen tien", "the tin dung", "so du", "vay",
    "ngan hang", "atm",
]

# Blocked topics (immediate reject)
BLOCKED_TOPICS = [
    "hack", "exploit", "weapon", "drug", "illegal",
    "violence", "gambling", "bomb", "kill", "steal",
]
