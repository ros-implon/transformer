import torch

from model import Transformer

vocab_size = 1000
embed_size = 128
batch_size = 1
input_size = 4
output_size = 1

X_train = torch.randint(0, 1000, (batch_size, input_size))
y_train = torch.randint(0, 2, (batch_size, output_size))

# print(X_train, y_train)

print('Input Shape:', X_train.size())
model = Transformer(input_size, vocab_size, embed_size, batch_size)
model(X_train)
