import torch
import numpy 
import torch.nn.functional as F

class LayerNorm:

    def __init__(self, d_model, seq_len, eps=1e-5, device = None):

        if device == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device

        self.seq_len = seq_len
        self.d_model = d_model
        self.eps     = eps
        self.gamma   = torch.ones(self.d_model,  requires_grad= True, device = self.device)
        self.beta    = torch.zeros(self.d_model, requires_grad = True, device = self.device)
        # self.gamma   = self.gamma.view(1,1,-1)
        # self.beta    = self.beta.view(1,1,-1) 

    def forward(self, x):

        mean = x.mean(dim = -1, keepdim = True)

        var  = x.var(dim = -1, keepdim = True, unbiased = False)

        out = self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta

        return out

    def __call__(self, x):

        return self.forward(x)
    
    def parameters(self):

        return [self.gamma, self.beta]