#Code & Inspo from Xuandong Zhao & Sam Gunn's work
import numpy as np
class ZeroBitPRC(): 
    def __init__(self, codeword_len: int):
        self.codeword_len = codeword_len
        self.sparsity = int(math.log(codeword_len, 2))
        self.secret_len = int(math.pow(math.log(codeword_len, 2), 2)) 
        print(type(self.sparsity), type(self.secret_len))
    def GenKey(self): 
        pass
    def Encode(self):
        pass
    def Decode(self): 
        pass


ZeroBitPRC = ZeroBitPRC(1024)

    


