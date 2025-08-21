import math
import numpy as np
from softmax import Onlinesoftmax
import time

class BlockWiseAttention():
    def __init__(self,q,k,v, block_size_q=32,block_size_kv=64):
        self.q = q
        self.k = k
        self.v = v
        self.block_size_q = block_size_q
        self.block_size_kv = block_size_kv

    def flash_attention(self):
        seq_len, d_model = self.q.shape
        output = np.zeros_like(self.q)

        num_q_blocks = (seq_len + self.block_size_q - 1) // self.block_size_q
        num_kv_blocks = (seq_len + self.block_size_kv - 1) // self.block_size_kv

        for i in range(num_q_blocks):

            start_q = i*self.block_size_q
            end_q = min(start_q + self.block_size_q,seq_len)
            Q_block  = self.q[start_q:end_q]

            block_size = Q_block.shape[0]
            online_softmax_list = [Onlinesoftmax() for _ in range(block_size)]
            block_op = np.zeros_like(Q_block)

            for j in range(num_kv_blocks):
                start_kv = j*self.block_size_kv
                end_kv = min(start_kv + self.block_size_kv,seq_len)
                K_block = self.k[start_kv:end_kv]
                V_block = self.v[start_kv:end_kv] 

                score = Q_block @ K_block.T 

                for query_idx in range(block_size):
                    query_score = score[query_idx, :]

                    online_softmax_list[query_idx].update(query_score.tolist())

                    current_prob = online_softmax_list[query_idx].prob

                    block_op[query_idx] = np.zeros(d_model)
                    prob_start = 0 

                    for block_idx in range(j+1):
                        start_kv_recompute = block_idx * self.block_size_kv
                        end_kv_recompute = min(start_kv_recompute + self.block_size_kv,seq_len)
                        V_block_recompute = self.v[start_kv_recompute:end_kv_recompute]

                        block_size_kv = V_block_recompute.shape[0]
                        block_prob = current_prob[prob_start: prob_start + block_size_kv]
                        block_op[query_idx] += np.array(block_prob) @ V_block_recompute

                        prob_start += block_size_kv

            output[start_q:end_q] = block_op 
        return output
        
import numpy as np
import time

def standard_attention(Q, K, V):
    scores = Q @ K.T
    max_scores = np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return weights @ V

def test():
    # Big sequences that will break standard attention
    sizes = [1024, 2048, 4096, 8192]
    
    for seq_len in sizes:
        print(f"\n Testing {seq_len} sequence length")
        
        # Random data
        Q = np.random.randn(seq_len, 64).astype(np.float32)
        K = np.random.randn(seq_len, 64).astype(np.float32) 
        V = np.random.randn(seq_len, 64).astype(np.float32)
        
        # Flash attention
        start = time.time()
        flash_attn = BlockWiseAttention(Q, K, V, block_size_q=128, block_size_kv=128)
        flash_result = flash_attn.flash_attention()
        flash_time = time.time() - start
        
        print(f"Flash: {flash_time:.2f}s, Shape: {flash_result.shape}")
        print(f"Range: [{np.min(flash_result):.3f}, {np.max(flash_result):.3f}]")
        print(f"NaN/Inf: {'YES' if np.any(np.isnan(flash_result)) or np.any(np.isinf(flash_result)) else 'NO'}")
        
        # Only compare with standard if small enough
        if seq_len <= 2048:
            start = time.time()
            std_result = standard_attention(Q, K, V)
            std_time = time.time() - start
            
            error = np.max(np.abs(std_result - flash_result))
            print(f"Standard: {std_time:.2f}s")
            print(f"Max Error: {error:.2e}")
            print(f"Correct: {'YES' if error < 1e-3 else 'NO'}")
        else:
            print("Standard: TOO BIG - would crash")

if __name__ == "__main__":
    test()
