import torch
import numpy
import torch.nn as nn
import matplotlib.pyplot as plt


class Embeddings:
    def __init__(self, vocab_size, d_model, max_len):
        """
        Vocab size: vocabulary size
        d_model: embeddings dimentions 
        max_len: Maximum length of the sequence
        """
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Token Embeddings + He Intialization
        fan_in = d_model

        std = (2/fan_in) ** 0.5

        self.emb = torch.randn(vocab_size, d_model)
        
        self.emb = (self.emb * std).requires_grad_()



        # self.emb = torch.randn(vocab_size, d_model)

        # self.emb =  self.emb * std

        # self.emb.requires_grad_(True)
        
        self.max_len = max_len

        # Positional Encodings

        self.pos_emb = self._get_positional_encoding(max_len, d_model)

    def _get_positional_encoding(self, max_len, d_model):

        ## this function generates new encodings

        pe = torch.zeros(max_len, d_model)

        positions = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)

        dims = torch.arange(0, d_model , dtype = torch.float).reshape(1, -1)

        denom = 1/10000**(2*(dims//2)/d_model)

        pe[:, 0::2] = torch.sin(positions*denom[:, 0::2].repeat(max_len, 1))

        pe[:, 1::2] = torch.cos(positions*denom[:, 1::2].repeat(max_len, 1))

        return pe
    
    def forward(self, x):

        batch_size , seq_len = x.shape[0], x.shape[1]

        pos_emb   = self.pos_emb[:seq_len, :]

        token_emb = self.emb[x]

        return token_emb + pos_emb
    
    def parameters(self):
        return [self.emb]
    
    
    def plot_positional_encodings(self, vis_dims=0):
    
        plt.figure(figsize=(7, 4))

        positions = torch.arange(self.max_len)
        for dim in range(2):
            plt.plot(positions, self.pos_emb[:, vis_dims+dim].numpy(), label=f"Dim {dim}")

            for x, y in zip(positions, self.pos_emb[:, vis_dims+dim].numpy()):
                plt.text(x, y, f"{y:.2f}", fontsize=8, ha='center', va='bottom')

        plt.title(f"Positional Encodings (i = {vis_dims} th dims)")
        plt.xlabel("Position of Token")
        plt.ylabel("Encoding Value")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        print("a = 1, b = 2 (Token positions)")
        print("sin(a+b) = sin(a).cos(b) + cos(a).sin(b)\n")

        ## pso emb dim = num tokens, d model
        sina, cosa = self.pos_emb[1, vis_dims + 0], self.pos_emb[1, vis_dims + 1]
        sinb, cosb = self.pos_emb[2, vis_dims + 0], self.pos_emb[2, vis_dims + 1]

        print(f"sin(a) = {sina}")
        print(f"cos(a) = {cosa}")
        print(f"sin(b) = {sinb}")
        print(f"cos(b) = {cosb}")

        print(f"\nsin(1).cos(2) + cos(1).sin(2) = {sina*cosb + cosa*sinb}; \nsin(3) == {self.pos_emb[3, vis_dims+0]}")

        print("""\nExplaination: \n
              
        This property means that for any fixed offset B, the positional encoding of A+B can be obtained by applying a fixed linear transformation (a rotation) to the positional encoding of A.

        This is crucial for the self-attention mechanism. The attention score between a query at position q and a key at position k is based on a dot product. Because of this linear property, the dot product between their positional encodings, PE(q)T
        
        PE(k), depends only on the relative position q-k, not their absolute positions. This allows the model to easily learn how to attend to tokens that are a certain distance away, regardless of where they are in the sequence.
                """)
        return

    
    def __call__(self,x):

        return self.forward(x)
    

    










