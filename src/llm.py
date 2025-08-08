import numpy as np
import torch
import torch.nn.functional as F
from zero_bit_prc import ZeroBitPRC
from key_manager import fetch_key, gen_key
import galois

GF = galois.GF(2)

class LLM: 
    def __init__(self, tokenizer, model): 
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        self.hash = self.simple_hash

        # Generation parameters
        self.do_sample = True
        self.temperature = 1.5
        self.top_p = 0.9
        self.repetition_penalty = 1.2
        self.no_repeat_ngram_size = 3
        
        return 

    @staticmethod
    def simple_hash(token_id): 
        if(token_id % 2 == 0): 
            return 0
        else: 
            return 1
    
    def gen_codeword(self, num_tokens): 
        PRC = ZeroBitPRC(codeword_len = num_tokens)
        if(fetch_key(PRC.codeword_len) is None):
            gen_key(PRC.codeword_len, sparsity = 1)
        generator_matrix, parity_check_matrix, one_time_pad = fetch_key(PRC.codeword_len)
        encoding_key = (generator_matrix, one_time_pad)
        decoding_key = (parity_check_matrix, one_time_pad)
        codeword = PRC.encode(encoding_key, noise_rate = 0.00)
        return codeword
    
    def gen_response(self, prompt, num_tokens, is_watermarked):
        codeword = np.zeros(num_tokens, dtype = int)
        if(is_watermarked == True): 
            codeword = self.gen_codeword(num_tokens)

        response = self.sample_response(prompt, codeword, is_watermarked)
        return response

    def apply_repetition_penalty(self, logits, generated_ids):
        if self.repetition_penalty != 1.0:
            for token_id in set(generated_ids[0].tolist()):
                logits[0, token_id] /= self.repetition_penalty
        return logits

    def apply_no_repeat_ngram(self, logits, generated_ids):
        if self.no_repeat_ngram_size > 0 and generated_ids.size(1) >= self.no_repeat_ngram_size:
                banned_tokens = []
                context = generated_ids[0].tolist()
                prev_ngram = tuple(context[-(self.no_repeat_ngram_size - 1):])
                for i in range(len(context) - self.no_repeat_ngram_size + 1):
                    ngram = tuple(context[i:i + self.no_repeat_ngram_size])
                    if tuple(ngram[:-1]) == prev_ngram:
                        banned_tokens.append(ngram[-1])
                for token in banned_tokens:
                    logits[0, token] = -float("inf")
        return logits
    
    def apply_top_p_sampling(self, probs): 
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[0, indices_to_remove] = 0
        probs = probs / probs.sum()
        return probs
    
    #Bias probabilities towards so that the sampled token's hash matches bit
    def bias_probs(self, probs, bit):
        vocab_size = probs.shape[-1]
        device = probs.device

        # Vectorize hash over all indices once
        token_hash_vals = torch.tensor([self.hash(i) for i in range(vocab_size)], device=device)

        # Mask where hash == bit
        bits_to_boost = (token_hash_vals == bit)
        
        # Extract total probabilitiy of sampling a token with hash == bit
        group_probs = probs[0][bits_to_boost]
        unbiased_bit_prob = group_probs.sum().item()

        # Handle edge cases - avoid a divide by zero
        if unbiased_bit_prob == 0.0 or unbiased_bit_prob == 1.0:
            return probs

        # Compute biased sum
        biased_bit_prob = 1.0 if unbiased_bit_prob > 0.5 else 2 * unbiased_bit_prob

        #Boost the probabilities of bits which agree with 'bit' and reduce others
        biased_probs = probs.clone()
        biased_probs[0, bits_to_boost] = (probs[0, bits_to_boost] / unbiased_bit_prob) * biased_bit_prob
        biased_probs[0, ~bits_to_boost] = (probs[0, ~bits_to_boost] / (1 - unbiased_bit_prob)) * (1 - biased_bit_prob)

        # Normalize
        biased_probs = biased_probs / biased_probs.sum()

        return biased_probs
    
    def sample_response(self, prompt, codeword, is_watermarked): 
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generated_ids = input_ids["input_ids"]

        # Initial decoded text
        decoded_so_far = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        for token_idx in range(len(codeword)):
            # Get logits
            outputs = self.model(input_ids=generated_ids)
            logits = outputs.logits[:, -1, :]

            # Apply sampling strategies
            logits = self.apply_repetition_penalty(logits, generated_ids)
            logits = self.apply_no_repeat_ngram(logits, generated_ids)
            logits = logits / self.temperature
            probs = F.softmax(logits, dim=-1)
            probs = self.apply_top_p_sampling(probs)

            # Bias probabilities towards desired bit is we are watermarking
            desired_bit = codeword[token_idx]
            if(is_watermarked == True):
                probs = self.bias_probs(probs, desired_bit)

            # Sample token
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Print out newly sampled token
            full_decoded_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            new_token = full_decoded_text[len(decoded_so_far):]
            print(new_token, end = '', flush = True)
            decoded_so_far = full_decoded_text

            # Stop if EOS
            if next_token.item() == self.tokenizer.eos_token_id and token_idx >= len(codeword):
                break

        # Remove prompt from output and return response
        input_len = input_ids["input_ids"].shape[1]
        response = self.tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True).strip()

        return response

    def response_to_token_ids(self, response): 
        if response is None or len(response.strip()) == 0:
            return []
        token_ids = self.tokenizer(response, return_tensors="pt")["input_ids"][0].tolist()
        return token_ids

    def detect_watermarked_response(self, response):
        token_ids = self.response_to_token_ids(response)
        noisy_codeword = np.empty(0, dtype = int)
        for token in token_ids:
            noisy_codeword = np.append(noisy_codeword, self.hash(token))

        if(fetch_key(len(noisy_codeword)) is not None):
            generator_matrix, parity_check_matrix, one_time_pad = fetch_key(len(noisy_codeword))
        elif(fetch_key(len(noisy_codeword) - 1) is not None): 
            generator_matrix, parity_check_matrix, one_time_pad = fetch_key(len(noisy_codeword) - 1)
            noisy_codeword = noisy_codeword[:-1]
        else: 
            return False

        decoding_key = (parity_check_matrix, one_time_pad)
        PRC = ZeroBitPRC(codeword_len = len(noisy_codeword))
        is_watermarked = PRC.decode(decoding_key, GF(noisy_codeword))
        return is_watermarked
    