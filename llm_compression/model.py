"""The only Hugging Face-specific part of the compressor.

Both directions use exactly the same one-token-at-a-time inference path. Even
teacher-forcing a whole block during compression can change floating-point
rounding enough to break an arithmetic decoder, so we intentionally avoid it.
"""

from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
import platform
import re
from typing import Protocol

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / ".cache" / "huggingface"
TOTAL = 1 << 24


class Predictor(Protocol):
    """Implement this interface to use another source of token probabilities."""

    metadata: dict

    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: Sequence[int]) -> str: ...
    def cdf(self, previous: int | None, position: int) -> list[int]: ...
    def token_label(self, token: int) -> str: ...


def runtime_signature() -> dict:
    import torch

    return {
        "torch": version("torch"),
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "transformers": version("transformers"),
        "tokenizers": version("tokenizers"),
        "numpy": version("numpy"),
        "machine": platform.machine(),
        "system": platform.system(),
        "python": platform.python_version(),
    }


def probabilities_to_cdf(probabilities) -> list[int]:
    """Quantize a normalized distribution, reserving one count per token.

    Round cumulative mass, rather than independently rounding frequencies, so
    the final total is exact. Positive counts keep unlikely tokens encodable.
    """
    import numpy as np

    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 1 or not 0 < len(probs) < TOTAL:
        raise ValueError("Vocabulary size must be between 1 and 2**24 - 1")
    if not np.isfinite(probs).all() or (probs < 0).any() or probs.sum() <= 0:
        raise ValueError("Probabilities must be finite, nonnegative, and nonempty")
    probs = probs / probs.sum()
    cumulative = np.minimum(np.cumsum(probs), 1.0)
    cdf = np.empty(len(probs) + 1, dtype=np.int64)
    cdf[0] = 0
    cdf[1:] = np.floor(cumulative * (TOTAL - len(probs))).astype(np.int64)
    cdf[1:] += np.arange(1, len(probs) + 1, dtype=np.int64)
    cdf[-1] = TOTAL
    return cdf.tolist()


class HuggingFacePredictor:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        revision: str | None = None,
        context_tokens: int = 256,
        cache_dir: str | Path = DEFAULT_CACHE,
        offline: bool = False,
    ) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        if context_tokens <= 0:
            raise ValueError("context_tokens must be positive")
        if revision is None:
            revision = DEFAULT_REVISION if model_id == DEFAULT_MODEL else "main"
        cache_dir = str(cache_dir)
        common = {"cache_dir": cache_dir, "local_files_only": offline, "trust_remote_code": False}
        config = AutoConfig.from_pretrained(model_id, revision=revision, **common)
        commit = config._commit_hash
        if not commit or not re.fullmatch(r"[a-f0-9]{40}", commit):
            raise ValueError("Use a Hugging Face model ID with a resolvable commit revision")
        limit = getattr(config, "max_position_embeddings", getattr(config, "n_positions", None))
        if limit is not None and context_tokens > limit:
            raise ValueError(f"context_tokens exceeds this model's context limit ({limit})")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=commit, **common)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=commit,
            config=config,
            torch_dtype=torch.float32,
            attn_implementation="eager",
            use_safetensors=True,
            **common,
        ).to("cpu").eval()
        self.torch = torch
        self.context_tokens = context_tokens
        self.past = None
        start = self.tokenizer.bos_token_id
        if start is None:
            start = self.tokenizer.eos_token_id
        if start is None:
            start = 0
        vocab_size = self.model.get_output_embeddings().weight.shape[0]
        if not 0 <= start < vocab_size or not 0 < vocab_size < TOTAL:
            raise ValueError("Unsupported model vocabulary or beginning-of-sequence token")
        self.start_token = start
        self.metadata = {
            "model_id": model_id,
            "revision": commit,
            "context_tokens": context_tokens,
            "start_token": start,
            "vocab_size": vocab_size,
            "cdf_total": TOTAL,
            "inference": "cpu-float32-eager-single-thread-kv-v1",
            "runtime": runtime_signature(),
        }

    def encode(self, text: str) -> list[int]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if self.decode(tokens) != text:
            raise ValueError(
                "This tokenizer changes the input text; choose a reversible tokenizer "
                "(the default SmolLM2 tokenizer is reversible)"
            )
        if any(not 0 <= token < self.metadata["vocab_size"] for token in tokens):
            raise ValueError("Tokenizer returned a token outside the model vocabulary")
        return tokens

    def decode(self, tokens: Sequence[int]) -> str:
        return self.tokenizer.decode(
            list(tokens), skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    def token_label(self, token: int) -> str:
        return str(self.tokenizer.convert_ids_to_tokens(token))

    def cdf(self, previous: int | None, position: int) -> list[int]:
        if position % self.context_tokens == 0:
            if getattr(self.model, "_supports_cache_class", False):
                from transformers.cache_utils import DynamicCache

                self.past = DynamicCache()
            else:
                self.past = None
            token = self.start_token
        else:
            if previous is None:
                raise ValueError("Previous token is required inside a context block")
            token = previous
        with self.torch.inference_mode():
            output = self.model(
                input_ids=self.torch.tensor([[token]], dtype=self.torch.long),
                past_key_values=self.past,
                use_cache=True,
                return_dict=True,
            )
            self.past = output.past_key_values
            probabilities = self.torch.softmax(output.logits[0, -1].double(), dim=-1).numpy()
        return probabilities_to_cdf(probabilities)
