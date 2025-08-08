import numpy as np
import galois

GF = galois.GF(2)

class ZeroBitPRC(): 
    def __init__(self, codeword_len, sparsity = None):
        self.codeword_len = codeword_len
        if(sparsity is None):
            self.sparsity = int(np.log2(codeword_len))
        else: 
            self.sparsity = sparsity
        self.secret_len = int(np.log2(codeword_len)) ** 2
        self.num_parity_checks = int(0.99 * codeword_len)

    def KeyGen(self):
        # Generate a random sparse parity check matrix, P
        parity_check_matrix = np.zeros((self.num_parity_checks, self.codeword_len), dtype = int)

        for i in range (self.num_parity_checks):
            row = np.zeros(self.codeword_len, dtype=int)
            ones_indices = np.random.choice(self.codeword_len, size = self.sparsity, replace = False)
            row[ones_indices] = 1
            parity_check_matrix[i] = row 

        parity_check_matrix = GF(parity_check_matrix)
        null_space = parity_check_matrix.null_space()
        null_space = null_space.T 

        # Generate a random generator matrix G, such that PG = 0
        generator_matrix = np.zeros((self.codeword_len, self.secret_len), dtype = int)
        for i in range (self.secret_len): 
            rand_null_vector = null_space @ GF.Random(null_space.shape[1])
            generator_matrix[:, i] = rand_null_vector
        
        generator_matrix = GF(generator_matrix)
        one_time_pad = GF.Random(self.codeword_len)
        
        return generator_matrix, parity_check_matrix, one_time_pad
        
    def Encode(self, encoding_key, noise_rate):
        generator_matrix, one_time_pad = encoding_key
        # Generate a random secret
        secret = GF.Random(self.secret_len)

        # Error is added to ensure pseudorandomness
        error = GF(np.random.binomial(1, noise_rate, self.codeword_len))

        codeword = (generator_matrix @ secret + one_time_pad + error)
        return codeword
    
    def Decode(self, decoding_key, codeword): 
        parity_check_matrix, one_time_pad = decoding_key
        codeword = codeword + one_time_pad

        # Threshold for detection, can be tunes according to desired False Positive Rate
        threshold = (1/2 - self.num_parity_checks ** (-0.25)) * self.num_parity_checks
        
        syndrome = parity_check_matrix @ codeword
        failed_parity_checks = np.sum(syndrome == 1)
        return failed_parity_checks < threshold    

    def print_parameter_info(self): 
        print("codeword_len:", self.codeword_len)
        print("sparsity:", self.sparsity)
        print("secret_len:", self.secret_len)
        print("num_parity_checks:", self.num_parity_checks)




    


