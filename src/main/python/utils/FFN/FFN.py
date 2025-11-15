import torch
import math 
import numpy
import torch.nn.functional as F

class feedForward:

    def __init__(self, d_model, max_len, vocab_size):

        self.d_model    = d_model
        self.max_len    = max_len
        self.vocab_size = vocab_size

        self.layer_1    = torch.randn(d_model,   2*d_model)
        self.layer_2    = torch.randn(2*d_model, d_model)
        self.linear     = torch.randn(d_model,   vocab_size)

    def forward(self, x):

        # x shape = batch size, seq_len, d_model
        # w shape =           , d_model, vocab_size
        # out shape = batch_size, seq_len, vocab_size

        out_1 = x @ self.layer_1

        return out_1 @ self.layer_2
    
    def __call__(self, x):

        logits =  self.forward(x) @ self.linear

        out    =  F.softmax(logits, dim = 2)

        return out
    




    

 



