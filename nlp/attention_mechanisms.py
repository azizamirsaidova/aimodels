#https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention

import torch.nn as nn
import torch

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out_kq, d_out_v):
        super().__init__()
        self.d_out_kq = d_out_kq
        self.W_query = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out_v))
    
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        attn_scores = queries @ keys.T  # unnormalized attention weights    
        attn_weights = torch.softmax(
            attn_scores / self.d_out_kq**0.5, dim=-1
        )
        
        context_vec = attn_weights @ values
        return context_vec

def main():
    sentence = 'Life is short, eat dessert first'
    dc = {s:i for i,s 
        in enumerate(sorted(sentence.replace(',', '').split()))}
    sentence_int = torch.tensor(
        [dc[s] for s in sentence.replace(',', '').split()]
    )
    vocab_size = 50_000
    torch.manual_seed(123)
    embed = torch.nn.Embedding(vocab_size, 3)
    embedded_sentence = embed(sentence_int).detach()

    print(embedded_sentence)
    print(embedded_sentence.shape)
    torch.manual_seed(123)

    # reduce d_out_v from 4 to 1, because we have 4 heads
    d_in, d_out_kq, d_out_v = 3, 2, 4
    sa = SelfAttention(d_in, d_out_kq, d_out_v)
    print(sa(embedded_sentence))

if __name__ == '__main__':
    main()