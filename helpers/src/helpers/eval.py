from tqdm import tqdm
import torch
from torch import nn


def model_output_to_classes(model_output:torch.Tensor) -> torch.Tensor:
    return torch.max(model_output, 1)[1] # Indices of max values

def eval_acc(model:nn.Module, dataloader:torch.utils.data.DataLoader) -> float:
    sum = 0
    length = 0
    for (X, y) in tqdm(dataloader):
        model.eval()
        with torch.no_grad():
            y_p = model_output_to_classes(model(X))
            sum += torch.sum(y == y_p).item()
            length += len(y_p)
    return sum/length
