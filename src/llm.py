import numpy as np
import torch
import torch.nn.functional as F

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

    @staticmethod
    def simple_hash(token_id): 
        if(token_id % 2 == 0): 
            return 0
        else: 
            return 1
        
    def print_vocab_info(self): 
        vocab_dict = self.tokenizer.get_vocab()
        index_to_token = {idx: token for token, idx in vocab_dict.items()}
        sorted_tokens = sorted(index_to_token.items()) 
        print("\nVocabulary size:", len(vocab_dict))
        print("Select tokens in the Vocabulary")
        print("Index\tToken")
        for idx, token in sorted_tokens[1000:1100]:
            print(f"{idx}\t{token}")
        
        return
       
    def gen_response(self, prompt, max_tokens): 
        input_ids = self.tokenizer(prompt.strip(), return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **input_ids,
            max_new_tokens = max_tokens,
            do_sample = self.do_sample,
            temperature = self.temperature,
            top_p = self.top_p,
            repetition_penalty = self.repetition_penalty,
            no_repeat_ngram_size = self.no_repeat_ngram_size,
            eos_token_id = self.tokenizer.eos_token_id
        )
        input_len = input_ids["input_ids"].shape[1]
        generated_text = output[0][input_len:]  # slice only new tokens
        response = self.tokenizer.decode(generated_text, skip_special_tokens=True).strip()

        return response
    def gen_watermarked_response(self, prompt, max_tokens, encoding_key):
        return
    

    def modify_probs(self, probs, codeword_val):
        # probs: shape (1, vocab_size)
        vocab_size = probs.shape[-1]
        device = probs.device

        # Vectorize the hash function
        indices = torch.arange(vocab_size, device=device)
        hash_vals = torch.tensor([self.hash(i) for i in range(vocab_size)], device=device)
        
        # Create masks
        mask = (hash_vals == codeword_val)  # shape: (vocab_size,)
        
        # Apply scaling
        scale = torch.where(mask, torch.tensor(500.0, device=device), torch.tensor(0.2, device=device))
        probs = probs * scale  # broadcast over (1, vocab_size)
        
        # Normalize
        probs = probs / probs.sum()

        return probs

    def gen_watermarked_response(self, prompt, max_tokens, codeword): 
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generated_ids = input_ids["input_ids"]

        # Sampling config
        max_new_tokens = max_tokens
        temperature = self.temperature
        top_p = self.top_p
        repetition_penalty = self.repetition_penalty
        no_repeat_ngram_size = self.no_repeat_ngram_size

        # Initial decoded text
        decoded_so_far = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        for token_idx in range(max_new_tokens):
            # Get logits
            outputs = self.model(input_ids=generated_ids)
            logits = outputs.logits[:, -1, :]

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated_ids[0].tolist()):
                    logits[0, token_id] /= repetition_penalty

            # No-repeat ngram
            if no_repeat_ngram_size > 0 and generated_ids.size(1) >= no_repeat_ngram_size:
                banned_tokens = []
                context = generated_ids[0].tolist()
                prev_ngram = tuple(context[-(no_repeat_ngram_size - 1):])
                for i in range(len(context) - no_repeat_ngram_size + 1):
                    ngram = tuple(context[i:i + no_repeat_ngram_size])
                    if tuple(ngram[:-1]) == prev_ngram:
                        banned_tokens.append(ngram[-1])
                for token in banned_tokens:
                    logits[0, token] = -float("inf")

            # Temperature
            logits = logits / temperature

            # Top-p sampling
            probs = F.softmax(logits, dim=-1) 
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[0, indices_to_remove] = 0
            probs = probs / probs.sum()

            probs = self.modify_probs(probs, codeword[token_idx])
            # Sample token
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Decode full text so far
            new_decoded = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # Find the new part
            new_piece = new_decoded[len(decoded_so_far):]
            print(new_piece, end='', flush=True)

            # Update tracker
            decoded_so_far = new_decoded

            # Stop if EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        input_len = input_ids["input_ids"].shape[1]
        response = self.tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True).strip()

        return response
            

    def response_to_token_ids(self, response): 
        if response is None or len(response.strip()) == 0:
            return []
        token_ids = self.tokenizer(response, return_tensors="pt")["input_ids"][0].tolist()
        return token_ids

    def detect_watermarked_response(self, response, decoding_key):
        token_ids = self.response_to_token_ids(response) 
        codeword = np.empty(0)
        for token in token_ids:
            codeword = np.append(codeword, self.hash(token))
        print(codeword)
        return
    
    def gen_token_logits(self, prompt):
        return
    
    def calc_response_entropy(self, prompt, max_tokens, response): 
        return
    