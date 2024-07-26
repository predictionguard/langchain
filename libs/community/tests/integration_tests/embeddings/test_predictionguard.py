"""Test Prediction Guard API wrapper"""


from langchain_community.embeddings.predictionguard import PredictionGuardEmbeddings


def test_predictionguard_call() -> None:
    """Test a valid call to Prediction Guard."""
    embeddings = PredictionGuardEmbeddings(model="bridgetower-large-itm-mlm-itc")

    text = "This text is for testing."
    query_result = embeddings.embed_query(text)

    assert isinstance(query_result[0], float)