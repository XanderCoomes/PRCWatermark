# def KeyGen(self):
#         generator_matrix = GF.Random((self.codeword_len, self.secret_len))
#         row_indices = []
#         col_indices = []
#         data = []
#         # Sample the last n - r rows of the generator along with the parity check matrix
#         # Note for small n, this may not sample the generator matrix uniformly
#         for row in range(self.num_parity_checks):
#             chosen_indices = np.random.choice(self.codeword_len - self.num_parity_checks + row, self.sparsity - 1, replace = False)
#             chosen_indices = np.append(chosen_indices, self.codeword_len - self.num_parity_checks + row)
#             row_indices.extend([row] * self.sparsity)
#             col_indices.extend(chosen_indices)
#             data.extend([1] * self.sparsity)
#             #Add dependencies into the generator matrix
#             generator_matrix[self.codeword_len - self.num_parity_checks + row] = generator_matrix[chosen_indices[:-1]].sum (axis=0)
#         parity_check_matrix = GF(csr_matrix((data, (row_indices, col_indices))).toarray() % 2)
#         one_time_pad = GF.Random(self.codeword_len)
#         return generator_matrix, parity_check_matrix, one_time_pad

    # def print_vocab_info(self): 
    #     vocab_dict = self.tokenizer.get_vocab()
    #     index_to_token = {idx: token for token, idx in vocab_dict.items()}
    #     sorted_tokens = sorted(index_to_token.items()) 
    #     print("\nVocabulary size:", len(vocab_dict))
    #     print("Select tokens in the Vocabulary")
    #     print("Index\tToken")
    #     for idx, token in sorted_tokens[1000:1100]:
    #         print(f"{idx}\t{token}")
        
    #     return