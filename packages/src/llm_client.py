import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are BrowseGuard's in-page assistant. You are given the visible text "
    "content of the web page the user is currently looking at, followed by "
    "their question. Answer the question using that page content as context. "
    "If the page content doesn't contain the answer, say so plainly instead "
    "of guessing."
)

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        base_url = os.environ["INFERENCE_BASE_URL"]
        logger.info("Creating inference client for base_url=%s", base_url)
        _client = OpenAI(api_key=os.environ["INFERENCE_API_KEY"], base_url=base_url)
    return _client


def ask(user_text: str, page_content: str) -> str:
    """Ask the assistant a question, using page_content as context."""
    logger.info(
        "ask() called | user_text=%r | page_content_chars=%d",
        user_text, len(page_content),
    )

    client = _get_client()

    user_prompt = f"Page content:\n{page_content}\n\nQuestion: {user_text}"
    logger.debug("Built user prompt (%d chars)", len(user_prompt))

    model = os.environ["INFERENCE_MODEL"]
    logger.info("Sending request to model=%s", model)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    logger.info("Received response from model=%s", model)

    answer = response.choices[0].message.content
    logger.debug("Answer (%d chars): %r", len(answer or ""), answer)
    return answer


if __name__ == "__main__":
    print(ask("Say hello in one short sentence.", ""))
