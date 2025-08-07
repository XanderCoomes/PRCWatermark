from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import torch.nn.functional as F

class LLM: 
    def __init__(self, tokenizer, model): 
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        return 
    def gen_response(self, prompt, max_tokens): 
        input_ids = self.tokenizer(prompt.strip(), return_tensors="pt")
        output = self.model.generate(
            **input_ids,
            max_new_tokens = max_tokens,
            do_sample = True,
            temperature = 0.85,
            top_p = 0.9,
            repetition_penalty = 1.15,
            no_repeat_ngram_size = 3,
            eos_token_id = self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response
    def gen_watermarked_response(self, prompt, max_tokens, encoding_key): 
        return
    
    def detect_watermarked_response(self, response, decoding_key):
        return
    
    def gen_token_logits(self, prompt):
        return
    
    def calc_response_entropy(self, prompt, max_tokens, response): 
        return
    