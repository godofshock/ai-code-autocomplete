def tokenize(code):
    tokens = code.replace("(", " ( ").replace(")", " ) ").split()
    return tokens

def build_vocab(tokens):
    vocab = list(set(tokens))
    word2idx = {w:i for i,w in enumerate(vocab)}
    idx2word = {i:w for w,i in word2idx.items()}
    return word2idx, idx2word
