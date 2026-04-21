def create_sequences(tokens, word2idx, seq_len=3):
    X, y = [], []

    for i in range(len(tokens)-seq_len):
        seq = tokens[i:i+seq_len]
        target = tokens[i+seq_len]

        X.append([word2idx[w] for w in seq])
        y.append(word2idx[target])

    return X, y
