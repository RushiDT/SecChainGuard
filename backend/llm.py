import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Base model on Hugging Face
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Where your fine-tuned LoRA adapters will live
ADAPTER_PATH = os.path.join(
    os.path.dirname(__file__),  # .../backend
    "..",
    "fine_tuned_models",
    "tinyllama-iot-sec-lora",
)


class LLMWrapper:
    def __init__(self, device: str = None):
        # Optional seed for slightly more stable behavior
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[LLM] Device: {self.device}")

        print(f"[LLM] Loading base model: {BASE_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        )

        # Load LoRA adapters if folder exists
        if os.path.isdir(ADAPTER_PATH):
            print(f"[LLM] Loading LoRA adapters from: {ADAPTER_PATH}")
            self.model = PeftModel.from_pretrained(self.model, ADAPTER_PATH)
        else:
            print(
                f"[LLM] WARNING: LoRA adapter path not found: {ADAPTER_PATH}. "
                "Using base model only."
            )

        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 600) -> str:
        """Generate answer for a given prompt."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # deterministic
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

        # Strip prompt echo if present
        if full_text.startswith(prompt):
            full_text = full_text[len(prompt):]

        return full_text.strip()
