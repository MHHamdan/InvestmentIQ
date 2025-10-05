"""
LLM Factory for InvestmentIQ MVAS

Provides unified interface for different LLM providers (Hugging Face, OpenAI, Anthropic).
Uses Hugging Face as primary provider with fallback options.
"""

import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

load_dotenv()


class LLMFactory:
    """
    Factory for creating LLM instances with provider abstraction.

    Supports:
    - Hugging Face (primary, free hosted models)
    - OpenAI (optional, if API key provided)
    - Anthropic (optional, if API key provided)
    """

    def __init__(self):
        self.hf_api_key = os.getenv("HUGGING_FACE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

        # Determine available providers
        self.providers = []
        if self.hf_api_key and self.hf_api_key != "your_huggingface_key_here":
            self.providers.append("huggingface")
        if self.openai_api_key and self.openai_api_key != "your_openai_key_here":
            self.providers.append("openai")
        if self.anthropic_api_key and self.anthropic_api_key != "your_anthropic_key_here":
            self.providers.append("anthropic")

        # Default to Hugging Face
        self.default_provider = self.providers[0] if self.providers else "huggingface"

    def create_chat_model(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        **kwargs
    ):
        """
        Create a chat model instance.

        Args:
            provider: LLM provider ('huggingface', 'openai', 'anthropic')
            model_name: Specific model name (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific arguments

        Returns:
            LangChain-compatible chat model instance
        """
        provider = provider or self.default_provider

        if provider == "huggingface":
            return self._create_huggingface_model(model_name, temperature, max_tokens, **kwargs)
        elif provider == "openai":
            return self._create_openai_model(model_name, temperature, max_tokens, **kwargs)
        elif provider == "anthropic":
            return self._create_anthropic_model(model_name, temperature, max_tokens, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _create_huggingface_model(
        self,
        model_name: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs
    ):
        """Create Hugging Face model using LangChain integration."""
        try:
            from langchain_huggingface import HuggingFaceEndpoint

            # Default to good free models for financial/business analysis
            if not model_name:
                # Using meta-llama/Llama-3.2-3B-Instruct (free, good for analysis)
                model_name = "meta-llama/Llama-3.2-3B-Instruct"

            model = HuggingFaceEndpoint(
                repo_id=model_name,
                huggingfacehub_api_token=self.hf_api_key,
                temperature=temperature,
                max_new_tokens=max_tokens,
                **kwargs
            )

            return model

        except ImportError:
            raise ImportError(
                "langchain-huggingface not installed. "
                "Run: uv pip install langchain-huggingface"
            )

    def _create_openai_model(
        self,
        model_name: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs
    ):
        """Create OpenAI model (fallback)."""
        try:
            from langchain_openai import ChatOpenAI

            model_name = model_name or "gpt-4o-mini"

            return ChatOpenAI(
                model=model_name,
                api_key=self.openai_api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

        except ImportError:
            raise ImportError(
                "langchain-openai not installed. "
                "Run: uv pip install langchain-openai"
            )

    def _create_anthropic_model(
        self,
        model_name: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs
    ):
        """Create Anthropic model (fallback)."""
        try:
            from langchain_anthropic import ChatAnthropic

            model_name = model_name or "claude-3-5-sonnet-20241022"

            return ChatAnthropic(
                model=model_name,
                api_key=self.anthropic_api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

        except ImportError:
            raise ImportError(
                "langchain-anthropic not installed. "
                "Run: uv pip install langchain-anthropic"
            )

    def create_embeddings(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Create embeddings model for RAG applications.

        Args:
            provider: Embedding provider
            model_name: Specific embedding model

        Returns:
            LangChain-compatible embeddings instance
        """
        provider = provider or self.default_provider

        if provider == "huggingface":
            return self._create_huggingface_embeddings(model_name)
        elif provider == "openai":
            return self._create_openai_embeddings(model_name)
        else:
            # Fallback to HuggingFace embeddings
            return self._create_huggingface_embeddings(model_name)

    def _create_huggingface_embeddings(self, model_name: Optional[str]):
        """Create Hugging Face embeddings."""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            # Default to sentence-transformers model optimized for semantic search
            if not model_name:
                model_name = "sentence-transformers/all-MiniLM-L6-v2"

            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
                encode_kwargs={'normalize_embeddings': True}
            )

        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: uv pip install sentence-transformers"
            )

    def _create_openai_embeddings(self, model_name: Optional[str]):
        """Create OpenAI embeddings (fallback)."""
        try:
            from langchain_openai import OpenAIEmbeddings

            model_name = model_name or "text-embedding-3-small"

            return OpenAIEmbeddings(
                model=model_name,
                api_key=self.openai_api_key
            )

        except ImportError:
            raise ImportError(
                "langchain-openai not installed. "
                "Run: uv pip install langchain-openai"
            )

    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers based on API keys."""
        return self.providers

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about configured providers."""
        return {
            "default_provider": self.default_provider,
            "available_providers": self.providers,
            "huggingface_available": "huggingface" in self.providers,
            "openai_available": "openai" in self.providers,
            "anthropic_available": "anthropic" in self.providers
        }


# Singleton instance
_llm_factory = None


def get_llm_factory() -> LLMFactory:
    """Get singleton LLM factory instance."""
    global _llm_factory
    if _llm_factory is None:
        _llm_factory = LLMFactory()
    return _llm_factory
