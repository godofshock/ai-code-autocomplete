from tokenizer.tokenizer import tokenize, build_vocab
from utils.dataset import create_sequences
from model.model import CodeModel
from trainer.train import train_model
from inference.predict import predict_next

code = """
def add(a, b):
    return a + b
"""

tokens = tokenize(code)
word2idx, idx2word = build_vocab(tokens)

X, y = create_sequences(tokens, word2idx)

model = CodeModel(len(word2idx))

train_model(model, X, y)

print("\nPrediction:")
print(predict_next(model, X[0], idx2word))
