import logging
from typing import Any, Dict, List, Optional, Iterator

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Extra
from langchain_core.utils import get_from_dict_or_env, pre_init
from langchain_community.adapters.openai import (
    convert_dict_to_message,
    convert_message_to_dict,
)

from pydantic import BaseModel, Extra
logger = logging.getLogger(__name__)


class CompletionInput(BaseModel):
    """Options to affect the input of the request."""

    block_prompt_injection: Optional[bool] = None
    """Set to true to detect prompt injection attacks."""
    pii: Optional[str] = None
    """Set to either 'block' or 'replace'."""
    pii_replace_method: Optional[str] = None
    """Set to either 'random', 'fake', 'category', 'mask'."""


class CompletionOutput(BaseModel):
    """Options to affect the output of the response."""

    factuality: Optional[bool] = None
    """Set to true to turn on factuality processing."""
    toxicity: Optional[bool] = None
    """Set to true to turn on toxicity processing."""


class ChatPredictionGuard(BaseChatModel):
    """Prediction Guard chat models.

    To use, you should have the ``predictionguard`` python package installed, and the
    environment variable ``PREDICTIONGUARD_API_KEY`` set with your API key, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            chat = ChatPredictionGuard(
                predictionguard_api_key="<your API key>",
                model="Hermes-2-Pro-Llama-3-8B",
            )
    """

    client: Any

    model: Optional[str] = "Hermes-2-Pro-Llama-3-8B"
    """Model name to use."""

    max_tokens: Optional[int] = 256
    """The maximum number of tokens in the generated completion."""

    temperature: Optional[float] = 0.75
    """The temperature parameter for controlling randomness in completions."""

    top_p: Optional[float] = 0.1
    """The diversity of the generated text based on nucleus sampling."""

    top_k: Optional[int] = None
    """The diversity of the generated text based on top-k sampling."""

    stop: Optional[List[str]] = None

    predictionguard_input: Optional[CompletionInput] = None
    """The input check to run over the prompt before sending to the LLM."""

    predictionguard_output: Optional[CompletionOutput] = None
    """The output check to run the LLM output against."""

    predictionguard_api_key: Optional[str] = None
    """Prediction Guard API key."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @property
    def _llm_type(self) -> str:
        return "predictionguard-chat"

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        pg_api_key = get_from_dict_or_env(
                values, "predictionguard_api_key", "PREDICTIONGUARD_API_KEY"
        )

        try:
            from predictionguard import PredictionGuard

            values["client"] = PredictionGuard(
                api_key=pg_api_key,
            )

        except ImportError:
            raise ImportError(
                "Could not import predictionguard python package. "
                "Please install it with `pip install predictionguard --upgrade`."
            )

        return values

    def _get_parameters(self, **kwargs) -> Dict[str, Any]:
        # input kwarg conflicts with LanguageModelInput on BaseChatModel
        input = kwargs.pop("predictionguard_input", self.predictionguard_input)
        output = kwargs.pop("predictionguard_output", self.predictionguard_output)

        params = {
            **{
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "input": input.dict() if isinstance(input, BaseModel) else input,
                "output": output.dict() if isinstance(output, BaseModel) else output,
            },
            **kwargs,
        }

        return params

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts = [convert_message_to_dict(m) for m in messages]

        params = self._get_parameters(**kwargs)
        params["stream"] = True

        result = self.client.chat.completions.create(
            model=self.model,
            messages=message_dicts,
            **params,
        )
        for part in result:
            # get the data from SSE
            if "data" in part:
                part = part["data"]
            if len(part["choices"]) == 0:
                continue
            content = part["choices"][0]["delta"]["content"]
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(id=part["id"], content=content)
            )
            yield chunk

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts = [convert_message_to_dict(m) for m in messages]
        params = self._get_parameters(**kwargs)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=message_dicts,
            **params,
        )

        generations = []
        for res in response["choices"]:
            if res.get("status", "").startswith("error: "):
                err_msg = res["status"].removeprefix("error: ")
                raise ValueError(f"Error from PredictionGuard API: {err_msg}")

            message = convert_dict_to_message(res["message"])
            gen = ChatGeneration(message=message)
            generations.append(gen)

        return ChatResult(generations=generations)
