"""Test Prediction Guard API wrapper."""

import pytest

from langchain_community.llms.predictionguard import PredictionGuard


def test_predictionguard_call() -> None:
    """Test valid call to prediction guard."""
    llm = PredictionGuard(model="Hermes-2-Pro-Llama-3-8B")  # type: ignore[call-arg]
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


def test_predictionguard_pii() -> None:
    llm = PredictionGuard(
        model="Hermes-2-Pro-Llama-3-8B",
        predictionguard_input={
            "pii": "block",
        },
        max_tokens=100,
        temperature=1.0,
    )

    messages = [
        "Hello, my name is John Doe and my SSN is 111-22-3333",
    ]

    with pytest.raises(ValueError, match=r"personal identifiable information detected"):
        llm.invoke(messages)
