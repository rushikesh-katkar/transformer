import torch
import math 
import numpy
import torch.nn.functional as F

class feedForward:

    def __init__(self, d_model, max_len, vocab_size, device = None):

        self.d_model    = d_model
        self.max_len    = max_len
        self.vocab_size = vocab_size

        if device == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device

        self.layer_1    = torch.randn(d_model,   2*d_model, requires_grad=True, device = self.device)
        self.layer_2    = torch.randn(2*d_model, d_model, requires_grad=True, device = self.device)
        self.linear     = torch.randn(d_model,   vocab_size, requires_grad=True, device = self.device)

    def forward(self, x):

        # x shape = batch size, seq_len, d_model
        # w shape =           , d_model, vocab_size
        # out shape = batch_size, seq_len, vocab_size

        out_1 = x @ self.layer_1

        out_2 = out_1 @ self.layer_2

        return out_2 @ self.linear
    
    def __call__(self, x):

        return self.forward(x)
    
    def parameters(self):
        return [self.layer_1, self.layer_2, self.linear]
    

# class Linear:

#     def __init__(self, d_model, max_len, vocab_size):

#         self.d_model    = d_model
#         self.max_len    = max_len
#         self.vocab_size = vocab_size

#         self.linear     = torch.randn(d_model,   vocab_size, requires_grad=True)

#     def forward(self, x):

#         return x @ self.lnear
    
#     def __call__(self, x):

#         logits =  self.forward(x)

#         out    =  F.softmax(logits, dim = 2)

#         return out
    
#     def parameters(self):
#         return [self.layer_1, self.layer_2, self.linear]


    

 



