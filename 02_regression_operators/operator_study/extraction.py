import numpy as np
import torch


def extract_operator(model, compatibility, output_indices, device):
    free_dofs, members = compatibility.shape
    dimension = len(output_indices)
    operator = np.zeros((dimension, dimension))
    model.eval()
    with torch.no_grad():
        for k in range(dimension):
            token = np.zeros((free_dofs, members + 3), dtype=np.float64)
            token[:, :members] = compatibility
            token[output_indices[k], members] = 1.0
            token[:, members + 2] = 1.0
            batch = torch.tensor(token, dtype=torch.float64).unsqueeze(0).to(device)
            output = model(batch)
            for i in range(dimension):
                operator[i, k] = output[0, output_indices[i], members + 1].item()
    return operator
