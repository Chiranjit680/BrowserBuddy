"""Manual smoke checks — run this file, read the output.

Not a pytest suite: every one of these makes a live model call, and two of
them need a browser. P0.1 in TASKS.md is where these become real tests.
"""

import agents
import llm_client
import log_setup


def test_ask():
    """The one-shot client: no tools, no browser."""
    result = llm_client.ask("What is the capital of France?", "")
    print(result)


def test_agent_ask():
    """The agent path, on a question needing no tool call."""
    print(agents.ask("Say hello in one short sentence."))


def test_agent_reads_browser():
    """The agent's tools end to end.

    Needs tracking to be up — start app.py or browser_track.py first, in
    another process. Without it the tools report the browser as unavailable
    and the agent says so, which is itself a passing result for the wiring.
    """
    print(agents.ask("What tabs do I have open, and what is each one about?"))


if __name__ == "__main__":
    log_setup.configure()
    test_ask()
    test_agent_ask()
    test_agent_reads_browser()
