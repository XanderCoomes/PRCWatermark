
import os
import numpy as np
from prc.zero_bit_prc import key_gen
import galois

GF = galois.GF(2)

class KeyManager:
    def __init__(self, model_name, base_dir):
        self.model_name = model_name
        self.base_dir = base_dir

    def _model_dir(self):
        return os.path.join(self.base_dir, self.model_name)

    def _key_filename(self, codeword_len, sparsity):
        s_str = "None" if sparsity is None else str(sparsity)
        return f"n={codeword_len}_sparsity={s_str}_keys.npz"

    def _key_path(self, codeword_len, sparsity):
        return os.path.join(self._model_dir(), self._key_filename(codeword_len, sparsity))
                            
    def gen_key(self, codeword_len, sparsity): 
        model_dir = self._model_dir()
        os.makedirs(model_dir, exist_ok = True)
        path = self._key_path(codeword_len, sparsity)

        g, p, otp, perm = key_gen(codeword_len, sparsity)
        np.savez(path,
                    generator_matrix = g,
                    parity_check_matrix = p,
                    one_time_pad=otp,
                    permutation=perm)
        return g, p, otp, perm

    def fetch_key(self, codeword_len, sparsity):
        path = self._key_path(codeword_len, sparsity)
        if(os.path.exists(path)):
            data = np.load(path, allow_pickle = False)
            g = data["generator_matrix"]
            p = data["parity_check_matrix"]
            otp = data["one_time_pad"]
            perm = data["permutation"]
            return GF(g), GF(p), GF(otp), perm
        else:
            return None
        
    
    def clear_all_keys(self): 
        model_dir = self._model_dir()
        if not os.path.exists(model_dir):
            return  # Nothing to clear

        for filename in os.listdir(model_dir):
            if filename.endswith(".npz"):
                file_path = os.path.join(model_dir, filename)
                try:
                    os.remove(file_path)
                except OSError as e:
                    print(f"Error deleting {file_path}: {e}")
