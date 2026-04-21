import torch

def predict_next(model, seq, idx2word):
    seq = torch.tensor([seq])
    output = model(seq)
    pred = torch.argmax(output, dim=1).item()
    return idx2word[pred]
