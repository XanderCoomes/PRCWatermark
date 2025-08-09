from zero_bit_prc import ZeroBitPRC
from key_manager import fetch_key, clear_key, clear_all_keys, gen_key
import numpy as np
import galois
import pytest
import os

GF = galois.GF(2)

#Testing Parameters
noise_levels = np.linspace(0, 0.3, 20)
noise_levels = noise_levels.round(2)

clear_key = False

def test_setup():
    if clear_key:
        clear_all_keys()
    assert True
    return


#Setup the ZeroBitPRC instance
@pytest.fixture
def prc(codeword_len = 2 ** 6): 
    return ZeroBitPRC(codeword_len)


# For undetectability, n ^ t should be greater than 2 ** 100
@pytest.mark.parametrize("noise", noise_levels)
def test_encode_with_noise(prc, noise, sparsity = None): 
    if(fetch_key(prc.codeword_len) is None):
        gen_key(prc.codeword_len, sparsity)
    generator_matrix, parity_check_matrix, one_time_pad = fetch_key(prc.codeword_len)
    encoding_key = generator_matrix, one_time_pad
    decoding_key = parity_check_matrix, one_time_pad
    codeword = prc.encode(encoding_key, noise)
    is_detected = prc.decode(decoding_key, codeword)
    assert (is_detected == True), "PRC codeword not detected"

def test_fpr(prc, num_trials = 1000): 
    generator_matrix, parity_check_matrix, one_time_pad = fetch_key(prc.codeword_len)
    decoding_key = parity_check_matrix, one_time_pad
    false_positive_count = 0
    for trial in range(num_trials):
        random_bits = GF.Random(prc.codeword_len)
        is_detected = prc.decode(decoding_key, random_bits)
        if is_detected:
            false_positive_count += 1
    false_positive_rate =  (false_positive_count / num_trials)
    assert(false_positive_rate < 0.01), "False positive rate exceeded threshold"


