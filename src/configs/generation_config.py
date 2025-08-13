

class GenerationConfig(): 
    def __init__(self,  temperature, top_p, repetition_penalty, no_repeat_ngram_size, token_buffer, skip_special_tokens, add_special_tokens): 
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.token_buffer = token_buffer
        self.skip_special_tokens = skip_special_tokens
        self.add_special_tokens = add_special_tokens