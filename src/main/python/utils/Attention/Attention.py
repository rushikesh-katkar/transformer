import torch
import numpy 
import torch.nn.functional as F

class attentionLayer:

    def __init__(self, num_heads, d_model, max_len, autoRegressive = False):

        self.d_model    = d_model

        self.max_len    = max_len

        self.num_heads  = num_heads 

        self.W_q        = self._get_qkv_weights()

        self.W_k        = self._get_qkv_weights()

        self.W_v        = self._get_qkv_weights()

    def _get_qkv_weights(self):

        # Input X dim: batch_dim, seq_len, d_model 

        # dimentions of Wq: d_model x d_model - simple choice

        # He Intialization
        fan_in = self.d_model

        std = (2/fan_in) ** 0.5

        return torch.randn(self.d_model, self.d_model// self.num_heads) * std


    def forward(self, x):

        Q = x @ self.W_q # batch_size, seq_len, d_model

        K = x @ self.W_k # batch_size, seq_len, d_model

        V = x @ self.W_v # batch_size, seq_len, d_model

        sim_logits = Q @ K.transpose(1,2)/ (self.d_model**0.5)

        AttentionWeights = F.softmax(sim_logits, dim =1) @ V

        return AttentionWeights


    def __call__(self, x):

        return self.forward(x)

class multiHeads:

    def __init__(self, num_heads, d_model,  max_len):

        self.num_heads = num_heads

        self.max_len   = max_len

        self.d_model   = d_model

        self.heads = [attentionLayer(d_model = d_model, num_heads = num_heads, max_len = max_len) for i in range(num_heads)]

    def __call__(self, x):

        heads_out = [head(x) for head in self.heads]

        return torch.cat(heads_out, dim =2)

class residualConnection:

    def __init__(self):
        pass

    def __call__(self, x_before, x_after):

        return x_before + x_after