"""
Direct Hugging Face Inference API client.

Provides simple HTTP-based access to free Hugging Face models.
Avoids provider routing issues with HuggingFaceEndpoint.
"""

import os
import requests
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class HuggingFaceClient:
    """
    Simple client for Hugging Face Inference API.

    Uses direct HTTP requests to avoid provider routing complexity.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize HF client.

        Args:
            api_key: Hugging Face API key
        """
        self.api_key = api_key or os.getenv("HUGGING_FACE_API_KEY", "")
        self.base_url = "https://api-inference.huggingface.co/models"

        if not self.api_key or self.api_key == "your_huggingface_key_here":
            logger.warning("No valid Hugging Face API key provided")

    def generate_text(
        self,
        prompt: str,
        model: str = "facebook/bart-large-cnn",
        max_tokens: int = 200,
        temperature: float = 0.3,
        timeout: int = 30
    ) -> str:
        """
        Generate text using Hugging Face model via summarization task.

        Uses BART model which is free and optimized for text generation/summarization.

        Args:
            prompt: Input prompt
            model: Model ID (default: facebook/bart-large-cnn)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            timeout: Request timeout in seconds

        Returns:
            Generated text
        """
        url = f"{self.base_url}/{model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # Use summarization task for BART
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": max_tokens,
                "min_length": 30,
                "do_sample": True if temperature > 0 else False
            }
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout
            )

            if response.status_code == 200:
                result = response.json()

                # Handle different response formats
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict) and "summary_text" in result[0]:
                        return result[0]["summary_text"].strip()
                    elif isinstance(result[0], dict) and "generated_text" in result[0]:
                        return result[0]["generated_text"].strip()
                    elif isinstance(result[0], str):
                        return result[0].strip()
                elif isinstance(result, dict):
                    if "summary_text" in result:
                        return result["summary_text"].strip()
                    elif "generated_text" in result:
                        return result["generated_text"].strip()

                # Fallback
                logger.warning(f"Unexpected response format: {result}")
                return ""

            elif response.status_code == 503:
                logger.warning(f"Model {model} is loading, please retry in a few moments")
                return ""
            else:
                logger.error(
                    f"HF API error {response.status_code}: {response.text}"
                )
                return ""

        except Exception as e:
            logger.error(f"Error calling HF API: {e}")
            return ""

    def classify_sentiment(
        self,
        text: str,
        model: str = "ProsusAI/finbert",
        timeout: int = 10
    ) -> Dict[str, Any]:
        """
        Classify financial sentiment using FinBERT.

        Args:
            text: Input text
            model: Sentiment model ID
            timeout: Request timeout

        Returns:
            Sentiment classification result
        """
        url = f"{self.base_url}/{model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        payload = {"inputs": text}

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout
            )

            if response.status_code == 200:
                result = response.json()

                if isinstance(result, list) and len(result) > 0:
                    # FinBERT returns [{"label": "positive", "score": 0.xx}, ...]
                    return {"status": "success", "data": result[0]}

                return {"status": "success", "data": result}

            elif response.status_code == 503:
                return {"status": "loading", "data": None}
            else:
                return {"status": "error", "data": None}

        except Exception as e:
            logger.error(f"Error calling sentiment API: {e}")
            return {"status": "error", "data": None}


# Singleton instance
_hf_client = None


def get_hf_client() -> HuggingFaceClient:
    """Get singleton HF client instance."""
    global _hf_client
    if _hf_client is None:
        _hf_client = HuggingFaceClient()
    return _hf_client
