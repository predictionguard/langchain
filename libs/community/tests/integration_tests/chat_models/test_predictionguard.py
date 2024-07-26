"""Test Prediction Guard API wrapper"""


from langchain_community.chat_models.predictionguard import ChatPredictionGuard


def test_predictionguard_call() -> None:
    """Test a valid call to Prediction Guard."""
    llm = ChatPredictionGuard(
        model="Hermes-2-Pro-Llama-3-8B",
        max_tokens=100,
        temperature=1.0
    )

    messages = [
        (
            "system",
            "You are a helpful chatbot",
        ),
        ("human", "Tell me a joke.")
    ]

    output = llm.invoke(messages)
    assert isinstance(output, str)


def test_predictionguard_stream() -> None:
    """Test a valid call with streaming to Prediction Guard"""
    llm = ChatPredictionGuard(
        model="Hermes-2-Pro-Llama-3-8B",
        stream=True
    )

    messages = [
        (
            "system",
            "You are a helpful chatbot."
        ),
        ("human", "Tell me a joke.")
    ]

    test_iter = 0
    for out in llm.stream(messages):
        if test_iter == 2:
            break
        assert isinstance(out, str)
        test_iter += 1