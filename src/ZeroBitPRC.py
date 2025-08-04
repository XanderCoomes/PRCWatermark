#Note, code is borrowed & inspired by Xuandong Zhao and Sam Gunn's work on PRCs, their github is linked in references
import numpy as np
import galois
from scipy.sparse import csr_matrix

GF = galois.GF(2)

class ZeroBitPRC(): 
    def __init__(self, codeword_len: int, noise_rate: float):
        self.codeword_len = codeword_len
        self.sparsity = int(np.log2(codeword_len))
        self.secret_len = self.sparsity ** 2
        self.num_parity_checks = int(0.99 * codeword_len)
        self.noise_rate = noise_rate
    
    def KeyGen(self):
        generator_matrix = GF.Random((self.codeword_len, self.secret_len))
        row_indices = []
        col_indices = []
        data = []
        # Sample the last n - r rows of the generator along with the parity check matrix
        # Note for small n, this may not sample the generator matrix uniformly
        for row in range(self.num_parity_checks):
            chosen_indices = np.random.choice(self.codeword_len - self.num_parity_checks + row, self.sparsity - 1, replace = False)
            chosen_indices = np.append(chosen_indices, self.codeword_len - self.num_parity_checks + row)
            row_indices.extend([row] * self.sparsity)
            col_indices.extend(chosen_indices)
            data.extend([1] * self.sparsity)
            #Add dependencies into the generator matrix
            generator_matrix[self.codeword_len - self.num_parity_checks + row] = generator_matrix[chosen_indices[:-1]].sum (axis=0)
        parity_check_matrix = GF(csr_matrix((data, (row_indices, col_indices))).toarray() % 2)
        one_time_pad = GF.Random(self.codeword_len)
        return generator_matrix, parity_check_matrix, one_time_pad
    
    def Encode(self, key):
        secret = GF.Random(self.secret_len)
        error = GF(np.random.binomial(1, self.noise_rate, self.codeword_len))
        codeword = (generator_matrix @ secret @  + error) % 2
    def Decode(self, parity_check_matrix, codeword): 
        pass
    def print_field_info(self): 
        print("codeword_len:", self.codeword_len,"sparsity:", self.sparsity,"secret_len:", self.secret_len,"num_parity_checks:", self.num_parity_checks, '\n')




    


