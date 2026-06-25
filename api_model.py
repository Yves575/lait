import os
from pathlib import Path
import time
from abc import ABC, abstractmethod
from typing import Any

from google.genai import types


class LLMResponseError(RuntimeError):
    """Raised when an LLM provider returns an unusable response."""


class APIModel(ABC):
    name: str = ""
    api_key: str = ""
    base_url: str = ""

    def __init__(self, model_checkpoint: str | None, reasoning_effort: str | None = None) -> None:
        self.model_checkpoint = model_checkpoint
        self.reasoning_effort = reasoning_effort
        if not self.model_checkpoint:
            raise ValueError("A model name must be provided.")

        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.getenv(self.api_key)
        base_url = os.getenv(self.base_url) if self.base_url else None
        if not api_key:
            raise EnvironmentError(f"{self.api_key} is not found. Check you .env file.")

        self.client = self.create_client(api_key, base_url)

    @abstractmethod
    def create_client(self, api_key: str, base_url: str | None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def direct_message(self, message: str) -> str:
        raise NotImplementedError


class Gemini(APIModel):
    name = "gemini"
    api_key = "GEMINI_API_KEY"
        
    def create_client(self, api_key: str, base_url: str | None) -> Any:
        from google import genai

        return genai.Client(
            # api_key=api_key, 
            vertexai=True, 
            project="project-0f0a3047-6fbb-4b0f-8c1",
            )

    def submit_requests(self, request_path: str, request_name: str, poll_interval: int,) -> None:

        # Upload the dataset in Gemini
        uploaded_file = self.client.files.upload(
            file=request_path,
            config=types.UploadFileConfig(
                display_name=request_name,
                mime_type="jsonl",
            ),
        )

        print(f"Uploaded file: {uploaded_file.name}")

        file_batch_job = self.client.batches.create(
            model=self.model_checkpoint,
            src=uploaded_file.name,
            config={
                "display_name": request_name,
            },
        )

        print(f"Created batch job: {file_batch_job.name}")


        # Wait untill the response is generated
        job_name = file_batch_job.name
        print(f"Polling status for job: {job_name}")
        while True:
            batch_job = self.client.batches.get(name=job_name)
            job_state = batch_job.state.name
            if job_state in {
                "JOB_STATE_SUCCEEDED",
                "JOB_STATE_FAILED",
                "JOB_STATE_CANCELLED",
                "JOB_STATE_EXPIRED",
            }:
                break
            print(f"Job not finished. Current state: {batch_job.state.name}. Waiting {poll_interval} seconds...")
            time.sleep(poll_interval)

        print(f"Job finished with state: {batch_job.state.name}")

        if batch_job.state.name != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(
                f"Gemini batch job {job_name} ended with state {batch_job.state.name}."
            )

        result_file_name = batch_job.dest.file_name
        print("Result file:", result_file_name)

        result_jsonl = self.client.files.download(file=result_file_name).decode("utf-8")
        result_path = Path(request_path).with_name(f"{request_name}.jsonl")
        result_path.write_text(result_jsonl, encoding="utf-8")
        
        print(f"Saved to {result_path}")

    def direct_message(self, message: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_checkpoint,
            contents=message,
        )
        return response.text


class GPT(APIModel):
    name = "gpt"
    api_key = "AZURE_OPENAI_API_KEY" #" or OPENAI_API_KEY"
    base_url = "AZURE_OPENAI_BASE_URL"

    def create_client(self, api_key: str, base_url: str | None) -> Any:
        from openai import AzureOpenAI

        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
        return AzureOpenAI(
            api_key=api_key,
            azure_endpoint=base_url,
            api_version=api_version,
        )

    def direct_message(self, message: str) -> str:
        kwargs = {}
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        response = self.client.responses.create(
            model=self.model_checkpoint,
            input=message,
            **kwargs,
        )
        return response.output_text


class OpenRouter(APIModel):
    name = "openrouter"
    api_key = "OPENROUTER_API_KEY"
    base_url = "OPENROUTER_BASE_URL"

    def create_client(self, api_key: str, base_url: str | None) -> Any:
        from openai import OpenAI

        return OpenAI(
            api_key=api_key,
            base_url=base_url or "https://openrouter.ai/api/v1",
        )

    def direct_message(self, message: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_checkpoint,
            messages=[
                {"role": "user", "content": message},
            ],
        )
        return self._extract_response_text(response)

    def _extract_response_text(self, response: Any) -> str:
        """Return text content from an OpenRouter chat-completion response."""
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMResponseError("OpenRouter response did not contain choices.")

        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason not in (None, "stop"):
            raise LLMResponseError(
                f"OpenRouter response did not finish cleanly: finish_reason={finish_reason!r}."
            )

        message = getattr(choice, "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if isinstance(content, str) and content.strip():
            return content

        reasoning = getattr(message, "reasoning", None) if message is not None else None
        reasoning_chars = len(reasoning) if isinstance(reasoning, str) else 0
        raise LLMResponseError(
            "OpenRouter response did not contain text content "
            f"(reasoning_chars={reasoning_chars})."
        )


class DeepSeek(APIModel):
    name = "deepseek"
    api_key = "DEEPSEEK_API_KEY"

    def create_client(self, api_key: str, base_url: str | None) -> Any:
        from openai import OpenAI

        return OpenAI(
            api_key=api_key,
            base_url=base_url or "https://api.deepseek.com",
        )

    def direct_message(self, message: str) -> str:
        kwargs = {
            "model": self.model_checkpoint,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": message},
            ],
            "stream": False,
            "extra_body": {"thinking": {"type": "enabled"}},
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
