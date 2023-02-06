import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, input_size, vocab_size, embed_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.word_embedding = nn.Embedding(vocab_size, embed_size)

    def forward(self, x):
        embedding = self.word_embedding(x)
        print('Word Embedding Shape:', embedding.size())
        pos_enc = self.getPositionalEncoding()
        print('Positional Encoder Shape:', pos_enc.size())
        enc_x = torch.add(embedding, pos_enc)
        print('After adding Positional Encoding:', enc_x.size())
        
    def getPositionalEncoding(self, n=10000):
        P = torch.zeros((self.input_size, self.embed_size))
        for pos in range(self.input_size):
            for i in torch.arange(int(self.embed_size/2)):
                den = torch.pow(n, 2*i/self.embed_size)
                P[pos, 2*i] = torch.sin(pos/den)
                P[pos, 2*i+1] = torch.cos(pos/den)
        return P.repeat(self.batch_size, 1, 1)