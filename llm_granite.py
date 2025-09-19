from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class GraniteLLM:
    def __init__(self, model_name: str = "ibm/granite-2b-instruct-v1", max_new_tokens: int = 512, temperature: float = 0.7, device: str = "cpu"):
        """
        Initialize the Granite LLM for IBM Granite 2B Instruct model from local cache.
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, local_files_only=True).to(self.device)

    def generate(self, query: str, context: str = None, system_prompt: str = None, **kwargs) -> str:
        """
        Generate a response using IBM Granite 2B Instruct model.
        """
        if not context:
            return "No context provided for answering the question."
        prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Return only the answer part
        if "Answer:" in decoded:
            return decoded.split("Answer:", 1)[-1].strip()
        return decoded.strip()
