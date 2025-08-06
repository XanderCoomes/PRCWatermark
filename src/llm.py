from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import torch.nn.functional as F

 
class LLM: 
    def __init__(self, tokenizer, model): 
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        return 
    def generate_response(self, prompt, max_tokens): 
        return
    def generate_watermarked_response(self, prompt, max_tokens, encoding_key): 
        return
    
    def detect_watermarked_response(self, response, decoding_key):
        return
    
    def generate_token_logits(self, prompt):
        return
    
    def calculate_response_entropy(self, prompt, max_tokens, response): 
        return
    