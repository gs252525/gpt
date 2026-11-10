from typing import Protocol, Sequence


class Tokenizer(Protocol):
    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...

    def state_dict(self) -> dict: ...


class CharTokenizer:
    name = "char"

    def __init__(self, chars: Sequence[str]):
        self.chars = list(chars)
        if not self.chars or len(set(self.chars)) != len(self.chars):
            raise ValueError("chars must contain a non-empty, unique vocabulary")
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @classmethod
    def from_state_dict(cls, state: dict) -> "CharTokenizer":
        return cls(state["chars"])

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        missing = set(text).difference(self.stoi)
        if missing:
            sample = "".join(sorted(missing)[:8])
            raise ValueError(f"text contains characters outside the tokenizer vocabulary: {sample!r}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.chars[int(i)] for i in ids)

    def state_dict(self) -> dict:
        return {"type": self.name, "chars": self.chars}


class HuggingFaceBPETokenizer:
    name = "hf-bpe"

    def __init__(self, tokenizer, model_name: str, revision: str | None):
        self.tokenizer = tokenizer
        self.model_name = model_name
        init_kwargs = getattr(tokenizer, "init_kwargs", {})
        self.revision = (
            getattr(tokenizer, "_commit_hash", None)
            or init_kwargs.get("_commit_hash")
            or revision
        )

        backend_model = getattr(getattr(tokenizer, "backend_tokenizer", None), "model", None)
        if backend_model is None or type(backend_model).__name__.lower() != "bpe":
            actual = type(backend_model).__name__ if backend_model is not None else "unknown"
            raise ValueError(f"{model_name!r} uses {actual}, not a BPE tokenizer")

    @classmethod
    def from_pretrained(cls, model_name: str, revision: str | None = None):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face tokenizer support requires transformers. Install it with "
                "`conda run -n transformer python -m pip install transformers`."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            use_fast=True,
        )
        return cls(tokenizer, model_name, revision)

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def state_dict(self) -> dict:
        return {
            "type": self.name,
            "model_name": self.model_name,
            "revision": self.revision,
            "vocab_size": self.vocab_size,
        }


def make_tokenizer(
    name: str,
    text: str | None = None,
    state: dict | None = None,
    pretrained_name: str = "gpt2",
    revision: str | None = None,
) -> Tokenizer:
    if state is not None and state.get("type") != name:
        raise ValueError(f"checkpoint tokenizer is {state.get('type')!r}, requested {name!r}")

    if name == CharTokenizer.name:
        if state is not None:
            return CharTokenizer.from_state_dict(state)
        if text is None:
            raise ValueError("text is required when creating a new character tokenizer")
        return CharTokenizer.from_text(text)

    if name == HuggingFaceBPETokenizer.name:
        if state is not None:
            pretrained_name = state["model_name"]
            revision = state.get("revision")
        tokenizer = HuggingFaceBPETokenizer.from_pretrained(pretrained_name, revision)
        if state is not None and tokenizer.vocab_size != state["vocab_size"]:
            raise ValueError("Hugging Face tokenizer vocabulary differs from the checkpoint")
        return tokenizer

    raise ValueError(f"unsupported tokenizer: {name}")
