import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

def evaluate(model, loader, device):

    # print('[ # ] =====> Evaluating on validation set...')
    num_samples = 0
    num_corrections = 0

    with tqdm(total=len(loader), leave=False, desc='Evaluating...') as pbar:
        for cnt, (x1, x2, _, _, y) in enumerate(loader):
            pbar.update(1)
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.squeeze(1).to(device)

            probs = F.softmax(model(x1, x2), dim=1)
            predictions = torch.argmax(probs, dim=1)

            num_samples += len(y)
            num_corrections += torch.sum(predictions == y).item()

    acc = num_corrections / num_samples
    # print('[ ! ] =====> Evaluation Done. Accuracy {:.4f}'.format(acc))

    return acc
