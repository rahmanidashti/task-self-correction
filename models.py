import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from litellm import completion

# Models served through an API (via LiteLLM) instead of being loaded from HuggingFace.
# LiteLLM reads the provider keys from the environment (GEMINI_API_KEY, OPENAI_API_KEY).
API_MODELS = ("gemini/gemini-2.0-flash", "gpt-4o")


# --- Minimal dummy tokenizer for LiteLLM API models ---
# It only implements the minimal interface required by the generation file,
# without performing any real tokenization.
class DummyLitellmTokenizer:
    def apply_chat_template(self, prompt, add_generation_prompt=False, return_tensors=None, tokenize=True):
        return prompt

    def __call__(self, text, return_tensors=None):
        # When called, simply return the text wrapped in a dict.
        # This mimics the interface expected by generation.py.
        if return_tensors == "pt":
            return {"input_ids": text, "attention_mask": None}
        return text

    def decode(self, token_ids, skip_special_tokens=True):
        # Since token_ids is actually just a raw string, return it unmodified.
        return token_ids

# --- LiteLLM model wrapper ---
class LiteLLMModel:
    def __init__(self, model_name: str, tokenizer):
        self.model_name = model_name  # e.g. "gemini/gemini-2.0-flash" or "gpt-4o"
        self.tokenizer = tokenizer
        self.device = "cpu"
        # Dummy config attributes to satisfy generation.py
        self.config = type("Config", (), {})()
        self.config.max_position_embeddings = 2048  # default max context length
        self.config._name_or_path = model_name
        self.generation_config = type("GenConfig", (), {})()
        self.generation_config.pad_token_id = None
        self.generation_config.temperature = None
        self.generation_config.top_p = None

    def generate(self, prompt, **kwargs):
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.95)
        max_new_tokens = kwargs.get("max_new_tokens", 500)
        response = completion(
            model=self.model_name,
            messages=prompt,
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=top_p
        )
        return response.choices[0].message.content


class Models:

    def __init__(self, model_name: str):
        self.model_name = model_name

    def load_model(self):
        # flash attn 2 is required for Qwen2.5-14B (see install_cuda_toolkit.sh)
        if self.model_name == "Qwen/Qwen2.5-14B-Instruct":
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(self.model_name, attn_implementation='flash_attention_2',
                                                         torch_dtype=torch.bfloat16, device_map="auto")
            model.generation_config.temperature = None
            model.generation_config.top_p = None
            return tokenizer, model
        elif self.model_name in ("meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"):
            model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            return tokenizer, model
        elif self.model_name == "HuggingFaceTB/SmolLM2-1.7B-Instruct":
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16, device_map="auto")
            return tokenizer, model
        elif self.model_name in API_MODELS:
            # For LiteLLM API–based models, we don't need a real tokenizer.
            tokenizer = DummyLitellmTokenizer()
            model = LiteLLMModel(self.model_name, tokenizer)
            return tokenizer, model
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
