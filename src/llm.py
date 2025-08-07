class LLM: 
    def __init__(self, tokenizer, model): 
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        self.hash = self.simple_hash

        # Generation parameters
        self.do_sample = True
        self.temperature = 0.85
        self.top_p = 0.9
        self.repetition_penalty = 1.15
        self.no_repeat_ngram_size = 3
        return 

    def simple_hash(token_id): 
        if(token_id % 2 == 0): 
            return 0
        else: 
            return 1
    def print_vocab(self): 
        for token, idx in self.tokenizer.get_vocab().items():
            print(f"{idx}\t{token}")
        return
    def gen_response(self, prompt, max_tokens): 
        input_ids = self.tokenizer(prompt.strip(), return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **input_ids,
            max_new_tokens = max_tokens,
            do_sample = self.do_sample,
            temperature = self.temperature,
            top_p = self.tokenizer,
            repetition_penalty = self.repetition_penalty,
            no_repeat_ngram_size = self.no_repeat_ngram_size,
            eos_token_id = self.tokenizer.eos_token_id
        )
        decoded = self.tokenizer.decode(output[0], skip_special_tokens = True)

        if prompt.strip() in decoded:
            response = decoded.split(prompt.strip())[-1].strip()
        else:
            response = decoded.strip()

        return response
    def gen_watermarked_response(self, prompt, max_tokens, encoding_key): 
        return
    
    def response_to_token_ids(self, response): 
        if response is None or len(response.strip()) == 0:
            return []
        token_ids = self.tokenizer(response, return_tensors="pt")["input_ids"][0].tolist()
        return token_ids

    def detect_watermarked_response(self, response, decoding_key):\
        
        return
    
    def gen_token_logits(self, prompt):
        return
    
    def calc_response_entropy(self, prompt, max_tokens, response): 
        return
    