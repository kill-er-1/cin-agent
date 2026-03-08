import os
from typing import Optional
from hello_agents import HelloAgentsLLM
from openai import OpenAI

class MyLLM(HelloAgentsLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = "auto",
        **kwargs
    ):
      if provider == "deepseek":
        self.model = model or os.getenv("DEEPSEEK_MODEL_ID")
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        
        if not self.api_key:
            raise ValueError("API key is required for DeepSeek provider. Please set it via the 'api_key' parameter or the 'DEEPSEEK_API_KEY' environment variable.")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL")
        self.provider = "deepseek"
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens")
        self.timeout = kwargs.get("timeout", 60)
        
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url,timeout=self.timeout)
      elif provider == "kimi":
        self.model = model or os.getenv("KIMI_MODEL_ID")
        self.api_key = api_key or os.getenv("KIMI_API_KEY")
        
        if not self.api_key:
            raise ValueError("API key is required for Kimi provider. Please set it via the 'api_key' parameter or the 'KIMI_API_KEY' environment variable.")
        self.base_url = base_url or os.getenv("KIMI_BASE_URL")
        self.provider = "kimi"
        self.temperature = 1.0
        self.max_tokens = kwargs.get("max_tokens")
        self.timeout = kwargs.get("timeout", 60)
        
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url,timeout=self.timeout)
      elif provider == "qwen":
        self.model = model or os.getenv("QWEN_MODEL_ID")
        self.api_key = api_key or os.getenv("QWEN_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required for Qwen provider. Please set it via the 'api_key' parameter or the 'QWEN_API_KEY' environment variable.")
        self.base_url = base_url or os.getenv("QWEN_BASE_URL")
        self.provider = "qwen"
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens")
        self.timeout = kwargs.get("timeout", 60)

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url,timeout=self.timeout)
      else:
        super().__init__(model=model, api_key=api_key, base_url=base_url, provider=provider, **kwargs)
        
        