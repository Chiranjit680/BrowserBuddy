

import llm_client


def test_ask():
    result = llm_client.ask("What is the capital of France?", "")
    print(result)

if __name__ == "__main__":
    test_ask()