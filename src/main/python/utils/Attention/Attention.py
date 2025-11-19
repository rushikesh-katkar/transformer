import torch
import numpy 
import torch.nn.functional as F

class attentionLayer:

    def __init__(self, head_dim, num_heads, max_len, autoRegressive = False):

        self.head_dim    = head_dim

        self.max_len    = max_len

        self.num_heads = num_heads

        self.W_q        = self._get_qkv_weights()

        self.W_k        = self._get_qkv_weights()

        self.W_v        = self._get_qkv_weights()

    def _get_qkv_weights(self):

        # Input X dim: batch_dim, seq_len, head_dim 

        # dimentions of Wq: head_dim x head_dim - simple choice

        # He Intialization
        fan_in = self.head_dim

        std = (2/fan_in) ** 0.5

        W = torch.randn(self.head_dim, self.head_dim) * std

        W.requires_grad_(True)

        return W


    def forward(self, x):

        B, T, C = x.shape  # batch_size, seq_len, d_model

        Q = x.view(B, T, self.num_heads, self.head_dim) @ self.W_q # batch_size, seq_len, head_dim

        K = x.view(B, T, self.num_heads, self.head_dim) @ self.W_k # batch_size, seq_len, head_dim

        V = x.view(B, T, self.num_heads, self.head_dim) @ self.W_v # batch_size, seq_len, head_dim

        sim_logits = Q @ K.transpose(-2,-1)/ ((self.head_dim)**0.5)

        AttentionWeights = F.softmax(sim_logits, dim =-1) @ V

        return AttentionWeights


    def __call__(self, x):

        return self.forward(x)


    def parameters(self):
        return [self.W_k, self.W_q, self.W_v]

class multiHeads:

    def __init__(self, num_heads, d_model,  max_len):

        self.num_heads = num_heads

        self.max_len   = max_len

        self.d_model   = d_model

        self.heads = [attentionLayer(head_dim = d_model//num_heads, num_heads = num_heads, max_len = max_len) for i in range(num_heads)]

    def __call__(self, x):

        heads_out = [head(x) for head in self.heads]

        return heads_out

        # return torch.cat(heads_out, dim =-1)
    
    def parameters(self):
        params = []

        for ls in [_.parameters() for _ in self.heads]:

            params += ls

        return params

class residualConnection:

    def __init__(self):
        pass

    def __call__(self, x_before, x_after):

        return x_before + x_after