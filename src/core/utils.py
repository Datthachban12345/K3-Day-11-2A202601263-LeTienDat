"""
Lab 11 — Helper Utilities using OpenRouter
"""
from openai import OpenAI

from core.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL


def get_client():
    """Get OpenRouter client (OpenAI-compatible)."""
    return OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )


async def chat_with_agent(agent, runner, user_message: str, session_id=None):
    """Send a message to the agent and get the response.

    Supports both OpenRouterAgent (from agent.py) and legacy ADK agents.

    Args:
        agent: The agent instance (OpenRouterAgent or LlmAgent)
        runner: The runner instance (MockRunner or InMemoryRunner)
        user_message: Plain text message to send
        session_id: Optional session ID (for compatibility)

    Returns:
        Tuple of (response_text, session)
    """
    # Check if it's an OpenRouter-based agent
    if hasattr(agent, 'chat'):
        # OpenRouterAgent or ProtectedAgent
        if hasattr(agent, 'chat_with_guardrails'):
            response = await agent.chat_with_guardrails(user_message)
        else:
            response = await agent.chat(user_message)

        # Mock session for compatibility
        class MockSession:
            id = session_id or "default_session"

        return response, MockSession()

    # Legacy ADK agent (fallback)
    from google.genai import types
    from google.adk.agents.invocation_context import InvocationContext

    user_id = "student"
    app_name = runner.app_name

    session = None
    if session_id is not None:
        try:
            session = await runner.session_service.get_session(
                app_name=app_name, user_id=user_id, session_id=session_id
            )
        except (ValueError, KeyError):
            pass

    if session is None:
        try:
            session = await runner.session_service.create_session(
                app_name=app_name, user_id=user_id
            )
        except Exception:
            session = await runner.session_service.create_session(
                app_name=app_name, user_id=user_id
            )

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    )

    final_response = ""
    async for event in runner.run_async(
        user_id=user_id, session_id=session.id, new_message=content
    ):
        if hasattr(event, "content") and event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    final_response += part.text

    return final_response, session
