
class Generation:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def _get_max_new_tokens(self, input_ids):
        # Get the model's maximum context length (e.g., 4096 for LLaMA-2)
        max_context_length = self.model.config.max_position_embeddings
        # Calculate max_new_tokens
        input_token_count = input_ids.shape[1]
        max_new_tokens = max_context_length - input_token_count
        return max_new_tokens

    def generate(self, prompt:str, is_api = False, temperature:float = 0.7, top_k: float = 20, top_p: float = 0.95, sampling:bool = True) -> str:
        if is_api == True:
            outputs = self.model.generate(prompt=prompt, temperature=temperature, top_k=top_k, top_p=top_p)
            return outputs

        formatted_input = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt", tokenize=False)
        input_tensor = self.tokenizer(formatted_input, return_tensors="pt")
        input_ids, attention_mask = input_tensor["input_ids"], input_tensor["attention_mask"]
        max_new_tokens = self._get_max_new_tokens(input_ids=input_ids)
        if sampling:
            try:
                outputs = self.model.generate(input_ids.to(self.model.device), attention_mask=attention_mask.to(self.model.device),
                                              do_sample=True, temperature=temperature, top_k=top_k, top_p=top_p,
                                              max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
            except Exception as e:
                print(e)
                return f"ERROR: {e}"
        else:
            outputs = self.model.generate(input_ids.to(self.model.device), attention_mask=attention_mask.to(self.model.device),
                                          do_sample=False,
                                          max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)

        result = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        return result
