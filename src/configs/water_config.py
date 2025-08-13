

class WaterConfig(): 
    def __init__(self, sparsity_function, encoding_noise_rate, majority_encoding_rate, key_dir): 
        self.sparsity_function = sparsity_function
        self.encoding_noise_rate = encoding_noise_rate
        self.majority_encoding_rate = majority_encoding_rate
        self.key_dir = key_dir