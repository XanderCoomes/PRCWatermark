import galois
import numpy as np

GF = galois.GF(2)
def determine_majority(codeword): 
    count = 0
    for bit in codeword: 
        if(bit == 1): 
            count = count + 1
        else: 
            count = count - 1
    if(count > 0): 
        return 1
    elif(count < 0): 
        return 0
    else: 
        return codeword[0]

def majority_encode(codeword, encoding_rate):
    majority_encoding = np.empty(0, dtype = int)
    for bit in codeword: 
        majority_sample = GF.Random(encoding_rate)
        while(determine_majority(majority_sample) != bit): 
            majority_sample = GF.Random(encoding_rate)
        majority_encoding = np.append(majority_encoding, majority_sample)
    return majority_encoding


def majority_decode(majority_codeword, codeword_len): 
    partitions = partition_equal_np(majority_codeword, codeword_len)
    codeword = np.empty(0, dtype = int)
    for partition in partitions: 
        bit = determine_majority(partition)
        codeword = np.append(codeword, bit)
    return codeword

def partition_equal_np(codeword: np.ndarray, n: int):
    length = len(codeword)
    base_size = length // n
    remainder = length % n

    chunks = []
    start = 0
    for i in range(n):
        chunk_size = base_size + (1 if i < remainder else 0)
        chunks.append(codeword[start:start + chunk_size])
        start += chunk_size

    return chunks
