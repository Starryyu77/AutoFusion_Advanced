"""
DeepSeek API Client
-------------------
LLM client with caching, retry, cost tracking, and budget control.

Features:
- Request caching to avoid duplicate API calls
- Exponential backoff retry mechanism
- Real-time cost tracking and budget enforcement
- API response logging for debugging
"""

import os
import json
import hashlib
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class CostStats:
    """API cost statistics"""
    total_calls: int = 0
    cache_hits: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost_yuan: float = 0.0
    budget_limit_yuan: float = 10000.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def budget_remaining(self) -> float:
        return self.budget_limit_yuan - self.total_cost_yuan

    @property
    def cache_hit_rate(self) -> float:
        total = self.total_calls + self.cache_hits
        return self.cache_hits / total if total > 0 else 0.0


class DeepSeekClient:
    """
    DeepSeek API Client with caching and cost control.

    Usage:
        client = DeepSeekClient(
            api_key=os.environ['DEEPSEEK_API_KEY'],
            cache_dir='.cache/llm',
            budget_limit_yuan=5000.0
        )
        code = client.generate(prompt, architecture_hash)
    """

    # DeepSeek-V3 pricing (per 1K tokens)
    PRICE_PER_1K_PROMPT = 0.5  # yuan
    PRICE_PER_1K_COMPLETION = 2.0  # yuan

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = '.cache/llm',
        budget_limit_yuan: float = 10000.0,
        model: str = 'deepseek-chat',
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """
        Initialize DeepSeek client.

        Args:
            api_key: DeepSeek API key (default: from DEEPSEEK_API_KEY env var)
            cache_dir: Directory for caching API responses
            budget_limit_yuan: Maximum budget in yuan
            model: Model name (default: deepseek-chat)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            max_retries: Maximum retry attempts
            retry_delay: Initial retry delay (exponential backoff)
        """
        self.api_key = api_key or os.environ.get('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key required. Set DEEPSEEK_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.cache_dir = cache_dir
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize cost stats
        self.stats = CostStats(budget_limit_yuan=budget_limit_yuan)

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # Initialize OpenAI client (DeepSeek uses OpenAI-compatible API)
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url='https://api.deepseek.com/v1'
            )
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )

        # Log file for API calls
        self.log_file = os.path.join(cache_dir, 'api_calls.log')

    def _get_cache_key(self, prompt: str, architecture_hash: str) -> str:
        """Generate cache key from prompt and architecture hash."""
        content = f"{self.model}:{self.temperature}:{self.max_tokens}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        """Get cache file path for a cache key."""
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        """Load response from cache if exists."""
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                return data.get('response')
            except Exception:
                return None
        return None

    def _save_to_cache(self, cache_key: str, response: str, metadata: Dict[str, Any]):
        """Save response to cache."""
        cache_path = self._get_cache_path(cache_key)
        data = {
            'response': response,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat(),
        }
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate API cost in yuan."""
        prompt_cost = (prompt_tokens / 1000) * self.PRICE_PER_1K_PROMPT
        completion_cost = (completion_tokens / 1000) * self.PRICE_PER_1K_COMPLETION
        return prompt_cost + completion_cost

    def _check_budget(self, estimated_cost: float = 0.5) -> bool:
        """Check if there's enough budget remaining."""
        return (self.stats.total_cost_yuan + estimated_cost) <= self.stats.budget_limit_yuan

    def _log_api_call(self, prompt_hash: str, tokens: int, cost: float, cached: bool = False):
        """Log API call for tracking."""
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'prompt_hash': prompt_hash,
            'tokens': tokens,
            'cost_yuan': cost,
            'cached': cached,
            'model': self.model,
        }
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Warning: Failed to log API call: {e}")

    def _call_api_with_retry(self, prompt: str) -> tuple[str, Dict[str, Any]]:
        """
        Call DeepSeek API with exponential backoff retry.

        Returns:
            (generated_code, metadata)
        """
        delay = self.retry_delay
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                )

                # Extract response
                content = response.choices[0].message.content

                # Extract token usage
                usage = response.usage
                metadata = {
                    'prompt_tokens': usage.prompt_tokens,
                    'completion_tokens': usage.completion_tokens,
                    'total_tokens': usage.total_tokens,
                    'model': response.model,
                }

                return content, metadata

            except Exception as e:
                last_error = e
                print(f"API call attempt {attempt + 1}/{self.max_retries} failed: {e}")

                if attempt < self.max_retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff

        # All retries failed
        raise Exception(f"API call failed after {self.max_retries} attempts: {last_error}")

    def generate(self, prompt: str, architecture_hash: str = '') -> str:
        """
        Generate code using DeepSeek API with caching.

        Args:
            prompt: The prompt to send to LLM
            architecture_hash: Hash of architecture description for caching

        Returns:
            Generated code string

        Raises:
            BudgetExceededError: If budget limit reached
            APIError: If API call fails after all retries
        """
        # Generate cache key
        cache_key = self._get_cache_key(prompt, architecture_hash)

        # Check cache
        cached_response = self._load_from_cache(cache_key)
        if cached_response is not None:
            self.stats.cache_hits += 1
            self._log_api_call(cache_key[:16], 0, 0.0, cached=True)
            print(f"[Cache Hit] Using cached response (total cache hits: {self.stats.cache_hits})")
            return cached_response

        # Check budget
        if not self._check_budget():
            raise BudgetExceededError(
                f"Budget exceeded: {self.stats.total_cost_yuan:.2f} / {self.stats.budget_limit_yuan:.2f} yuan"
            )

        # Call API with retry
        print(f"[API Call] Calling DeepSeek API (attempt 1/{self.max_retries})...")
        start_time = time.time()

        response, metadata = self._call_api_with_retry(prompt)

        elapsed = time.time() - start_time
        print(f"[API Call] Completed in {elapsed:.2f}s")

        # Update statistics
        prompt_tokens = metadata['prompt_tokens']
        completion_tokens = metadata['completion_tokens']
        total_tokens = metadata['total_tokens']

        cost = self._calculate_cost(prompt_tokens, completion_tokens)

        self.stats.total_calls += 1
        self.stats.total_tokens += total_tokens
        self.stats.prompt_tokens += prompt_tokens
        self.stats.completion_tokens += completion_tokens
        self.stats.total_cost_yuan += cost

        # Log the call
        self._log_api_call(cache_key[:16], total_tokens, cost)

        print(f"[Cost] This call: {cost:.4f} yuan | Total: {self.stats.total_cost_yuan:.4f} yuan | "
              f"Remaining: {self.stats.budget_remaining:.4f} yuan")

        # Save to cache
        self._save_to_cache(cache_key, response, metadata)

        return response

    def get_stats(self) -> CostStats:
        """Get current cost statistics."""
        return self.stats

    def print_stats(self):
        """Print cost statistics."""
        print("\n" + "=" * 50)
        print("DeepSeek API Cost Statistics")
        print("=" * 50)
        print(f"Total API Calls:     {self.stats.total_calls}")
        print(f"Cache Hits:          {self.stats.cache_hits}")
        print(f"Cache Hit Rate:      {self.stats.cache_hit_rate:.2%}")
        print(f"Total Tokens:        {self.stats.total_tokens}")
        print(f"  - Prompt:          {self.stats.prompt_tokens}")
        print(f"  - Completion:      {self.stats.completion_tokens}")
        print(f"Total Cost:          {self.stats.total_cost_yuan:.4f} yuan")
        print(f"Budget Limit:        {self.stats.budget_limit_yuan:.4f} yuan")
        print(f"Budget Remaining:    {self.stats.budget_remaining:.4f} yuan")
        print("=" * 50 + "\n")

    def save_stats(self, filepath: str):
        """Save statistics to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.stats.to_dict(), f, indent=2)


class BudgetExceededError(Exception):
    """Raised when API budget is exceeded."""
    pass


class APIError(Exception):
    """Raised when API call fails."""
    pass


# Convenience function for quick usage
def create_deepseek_client(
    api_key: Optional[str] = None,
    budget_limit_yuan: float = 10000.0,
    **kwargs
) -> DeepSeekClient:
    """
    Create a DeepSeek client with common defaults.

    Args:
        api_key: API key (default: from DEEPSEEK_API_KEY env var)
        budget_limit_yuan: Maximum budget in yuan
        **kwargs: Additional arguments for DeepSeekClient

    Returns:
        DeepSeekClient instance
    """
    return DeepSeekClient(
        api_key=api_key,
        budget_limit_yuan=budget_limit_yuan,
        **kwargs
    )
