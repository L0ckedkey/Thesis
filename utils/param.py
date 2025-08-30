import torch
import numpy as np

def get_model_params(model):
    return [val.cpu().numpy() for val in model.state_dict().values()]

def set_model_params(model, params):
    state_dict = model.state_dict()
    for key, val in zip(state_dict.keys(), params):
        state_dict[key] = torch.tensor(val)
    model.load_state_dict(state_dict)
    return model
