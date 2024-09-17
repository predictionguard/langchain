import logging
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain_core.utils import get_from_dict_or_env

from langchain_community.

logger = logging.getLogger(__name__)


class ChatPredictionGuard(BaseChatModel):
    """Prediction Guard chat models.
    
    To use, you should have the ``predictionguard`` python package installed, and the
    environment variable ``PREDICTIONGUARD_API_KEY`` set with your api_key, or pass
    it as a named parameter to the constructor.
    
    Example:
        .. code-block:: python
        
            pgllm = PredictionGuard(model="Hermes-2-Pro-Llama-3-8B",
                                    api_key="my-api-key"
                                    )
    """

    client: Any
    model: Optional[str] = "Hermes-2-Pro-Llama-3-8B"
    """Model name to use."""

    input: Optional[Dict[str, Any]] = None
    """The input check to run over the prompt before sending to the LLM."""

    output: Optional[Dict[str, Any]] = None
    """The output check to run the LLM output against."""

    max_tokens: int = 256
    """Denotes the number of tokens to predict per generation."""

    temperature: float = 0.75
    """A non-negative float that tunes the degree of randomness in generation."""

    top_p: float = 0.1
    """A non-negative float that controls the diversity of the generated tokens."""

    api_key: Optional[str] = None
    """Your Prediction Guard api_key."""

    stop: Optional[List[str]] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid