"""LLM-based reason generator for trading decisions using Ollama."""

import os

import requests

from src.logger import logger


class LLMReasonGenerator:
    """Generate human-readable trading decision reasons using Ollama LLM."""

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        """Initialize LLM reason generator.

        Args:
            ollama_url: Ollama API endpoint
        """
        self.ollama_url = ollama_url
        self.model = "gpt-oss:120b-cloud"  # Cloud-hosted 120B model via Ollama
        self.timeout = 30  # Cloud models can be slow (30-60s)
        self.cache: dict[str, str] = {}  # In-memory cache for reasons
        self.use_llm = False  # Cloud model is too slow (30+s) - use fast fallback
        # To enable: Set use_llm=True and wait 30s per unique reason

    def generate_reason(
        self,
        symbol: str,
        composite: float,
        technical: float,
        fundamental: float,
        sentiment: float,
        thresholds: dict[str, float],
    ) -> str:
        """Generate a concise trading decision reason using LLM.

        Args:
            symbol: Stock ticker symbol
            composite: Composite score (0-100)
            technical: Technical analysis score (0-100)
            fundamental: Fundamental analysis score (0-100)
            sentiment: News sentiment score (0-100)
            thresholds: Dict with min_composite and min_factor thresholds

        Returns:
            Human-readable reason string
        """
        # Check cache first
        cache_key = self._get_cache_key(
            composite, technical, fundamental, sentiment, thresholds
        )

        if cache_key in self.cache:
            cached_reason: str = self.cache[cache_key]
            return cached_reason

        # Check if qualified
        qualified = (
            composite >= thresholds["min_composite"]
            and technical >= thresholds["min_factor"]
            and sentiment >= thresholds["min_factor"]
        )

        # Build concise prompt

        # Find the weakest scores
        weaknesses = []
        if composite < thresholds["min_composite"]:
            weaknesses.append(
                f"composite {composite:.0f} (need {thresholds['min_composite']:.0f})"
            )
        if technical < thresholds["min_factor"]:
            weaknesses.append(
                f"technical {technical:.0f} (need {thresholds['min_factor']:.0f})"
            )
        if sentiment < thresholds["min_factor"]:
            weaknesses.append(
                f"sentiment {sentiment:.0f} (need {thresholds['min_factor']:.0f})"
            )

        # Use cloud model for better reasons (if enabled)
        if not self.use_llm:
            # Skip LLM, use fast fallback
            reason = self._generate_fallback_reason(
                qualified, composite, technical, sentiment, thresholds
            )
            self.cache[cache_key] = reason
            return reason

        if qualified:
            prompt = f"In 8 words max, explain why {symbol} qualified for trading. Scores: composite {composite:.0f}, technical {technical:.0f}, sentiment {sentiment:.0f}. Be enthusiastic but concise."
        else:
            prompt = f"In 8 words max, explain why {symbol} was rejected. Problems: {', '.join(weaknesses)}. Be direct."

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.4,
                        "num_predict": 20,  # Very short response
                        "stop": ["\n", ".", "!"],  # Stop at first sentence
                    },
                },
                timeout=self.timeout,
            )

            if response.status_code == 200:
                result = response.json()
                reason = result.get("response", "").strip()

                if reason:
                    # Add status emoji and cache
                    prefix = "✓" if qualified else "✗"
                    full_reason = f"{prefix} {reason}"
                    self.cache[cache_key] = full_reason
                    logger.info("llm_reason_generated", symbol=symbol, cached=False)
                    return full_reason

        except Exception as e:
            # Fall through to fallback on any error
            logger.debug("llm_reason_fallback", error=str(e)[:100], symbol=symbol)

        # Fallback: Use simple rule-based reason
        reason = self._generate_fallback_reason(
            qualified, composite, technical, sentiment, thresholds
        )
        self.cache[cache_key] = reason
        return reason

    def _get_cache_key(
        self,
        composite: float,
        technical: float,
        fundamental: float,
        sentiment: float,
        thresholds: dict[str, float],
    ) -> str:
        """Generate cache key from scores (rounded to reduce cache misses)."""
        # Round scores to nearest 5 for better cache hits
        c = round(composite / 5) * 5
        t = round(technical / 5) * 5
        s = round(sentiment / 5) * 5
        return f"{c}_{t}_{s}_{thresholds['min_composite']}_{thresholds['min_factor']}"

    def _generate_fallback_reason(
        self,
        qualified: bool,
        composite: float,
        technical: float,
        sentiment: float,
        thresholds: dict[str, float],
    ) -> str:
        """Generate fallback reason when LLM is unavailable."""
        if qualified:
            strengths = []
            if technical >= 70:
                strengths.append("strong technicals")
            if sentiment >= 70:
                strengths.append("positive sentiment")
            if strengths:
                return f"✓ {', '.join(strengths).capitalize()}"
            return "✓ All criteria met"

        # Find weakest area
        scores = [
            (composite, "composite", thresholds["min_composite"]),
            (technical, "technical", thresholds["min_factor"]),
            (sentiment, "sentiment", thresholds["min_factor"]),
        ]

        weakest = min(scores, key=lambda x: x[0] - x[2])
        gap = weakest[2] - weakest[0]

        return f"✗ {weakest[1].capitalize()} {gap:.0f} pts below threshold"


# Singleton instance
_llm_generator: LLMReasonGenerator | None = None


def get_llm_reason_generator(ollama_url: str | None = None) -> LLMReasonGenerator:
    """Get or create LLM reason generator singleton.

    Args:
        ollama_url: Ollama API endpoint (uses env var OLLAMA_URL if not provided)
    """
    global _llm_generator

    if _llm_generator is None:
        # Get URL from env or use VPS default
        if ollama_url is None:
            ollama_url = os.getenv("OLLAMA_URL", "http://69.62.64.51:11434")

        _llm_generator = LLMReasonGenerator(ollama_url)
        logger.info("llm_reason_generator_initialized", url=ollama_url)

    return _llm_generator
