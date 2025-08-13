from prc.majority_encoding import majority_decode, majority_encode
import numpy as np


def test_majority_encode_and_decode(): 
    length = 10
    encoding_rate = 3
    original_array = np.ones(length)
    majority = majority_encode(original_array, encoding_rate)
    majority = np.append(majority, 1)
    majoirity = np.append(majority, 0)
    print(majority)
    final_array = majority_decode(majority, length)

    print(final_array)

    assert (np.array_equal(original_array, final_array))