"""Standalone inference adapter using the existing de_ch_translator prompts."""
from pathlib import Path

DIALECT_NAMES = {"bs": "Basel", "sg": "St. Gallen", "zh": "Zurich", "lu": "Lucerne", "be": "Bern"}


def translation_prompt(text: str, dialect: str) -> str:
    code = str(dialect).lower().removeprefix("ch_")
    if code == "ch":
        return f"translate German to Swiss German: {text}"
    if code not in DIALECT_NAMES:
        raise ValueError(f"Unsupported target dialect: {dialect}")
    return f"translate German to {DIALECT_NAMES[code]} Swiss German: {text}"


class T5Translator:
    def __init__(self, model_path: Path):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path)).to(self.device)
        self.model.eval()

    def translate(self, texts, dialect, batch_size=16, max_input_length=512, **generation):
        if isinstance(texts, str):
            texts = [texts]
        dialects = [dialect] * len(texts) if isinstance(dialect, str) else list(dialect)
        if len(texts) != len(dialects):
            raise ValueError("Every input needs exactly one target dialect")
        predictions = []
        for start in range(0, len(texts), batch_size):
            prompts = [translation_prompt(text, code) for text, code in
                       zip(texts[start:start + batch_size], dialects[start:start + batch_size])]
            encoded = self.tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
            if encoded["input_ids"].shape[1] > max_input_length:
                raise ValueError("Input exceeds max_input_length; split the source instead of truncating it")
            with self.torch.inference_mode():
                result = self.model.generate(**encoded.to(self.device), **generation)
            predictions.extend(self.tokenizer.batch_decode(result, skip_special_tokens=True))
        return predictions
