"""Test Prediction Guard API wrapper."""

from langchain_community.llms.predictionguard import PredictionGuard


def test_predictionguard_call() -> None:
    """Test valid call to prediction guard."""
    llm = PredictionGuard(model="Hermes-2-Pro-Llama-3-8B")  # type: ignore[call-arg]
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)
