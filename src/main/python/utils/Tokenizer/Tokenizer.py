import torch
import numpy
import torch.nn as nn
import matplotlib.pyplot as plt


class SimpleTokenizer:

    def __init__(self, vocab):
        
        self.vocab = vocab

        self.stoi  = {s:i for s,i in zip(self.vocab, list(range(1, len(self.vocab) + 1)))}

        self.stoi['<e>'] = len(self.vocab) + 1

        self.stoi['<s>'] = 0

        self.itos  = {i:s for s,i in self.stoi.items()}

        
    def encode(self, x, pad = False):
        if pad:
            return torch.tensor([0]+ [self.stoi[s] for s in list(x)] + [len(self.vocab) + 1]).reshape(1,len(x) + 2)
        else:
            return torch.tensor([self.stoi[s] for s in list(x)]).reshape(1,len(x))
    
    def decode(self, x):

        return ''.join([self.itos[i.item()] for i in x.reshape(x.shape[1])])
    

    

    