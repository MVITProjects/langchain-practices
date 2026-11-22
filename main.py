"""Runs the LangChain examples against a local OpenAI-compatible LLM."""
from collections.abc import Callable

from dotenv import load_dotenv

# Must run before importing langchain_examples: those modules read
# LLM_BASE_URL / LLM_MODEL / LLM_API_KEY from the environment at import time.
load_dotenv()

import langchain_examples  # noqa: E402  pylint: disable=wrong-import-position


def run_example(title: str, example: Callable[[], str | None]) -> None:
    """Run one example, printing its result or the error without stopping the rest."""
    print(f"\n=== {title} ===")
    try:
        print(example())
    except Exception as error:  # pylint: disable=broad-exception-caught
        print(f"failed: {type(error).__name__}: {error}")


def main() -> None:
    """main function"""

    run_example(
        "Basic LLM call",
        lambda: langchain_examples.basic_example_calling_llm("Hello!"),
    )

    run_example(
        "Prompt template: Python",
        lambda: langchain_examples.basic_example_calling_llm("Python")
    )

    run_example("Chat models example", langchain_examples.chat_models_example)

if __name__ == "__main__":
    main()
