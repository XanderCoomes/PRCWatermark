import numpy as np
import galois

GF = galois.GF(2)

class ZeroBitPRC(): 
    def __init__(self, codeword_len: int):
        self.codeword_len = codeword_len
        self.sparsity = int(np.log2(codeword_len))
        self.secret_len = self.sparsity ** 2
        self.num_parity_checks = int(0.99 * codeword_len)
    def KeyGen(self): 
        pass
    def Encode(self, key):
        pass
    def Decode(self, key, codeword): 
        pass
    def print_field_info(self): 
        print("codeword_len:", self.codeword_len,"sparsity:", self.sparsity,"secret_len:", self.secret_len,"num_parity_checks:", self.num_parity_checks)    




    


