import torch
import fms.utils.serialization as ser

# Find the parameter that causes the mismatch
original_load = ser._load_partial_state_dict

def patched_load(model, state_dict, **kwargs):
    # Check each parameter
    model_params = dict(model.named_parameters())

    for key, tensor in state_dict.items():
        # Find matching model param
        if key in model_params:
            model_tensor = model_params[key]
            if model_tensor.shape != tensor.shape:
                print(f'SHAPE MISMATCH: {key}')
                print(f'  Checkpoint: {tensor.shape}')
                print(f'  Model:      {model_tensor.shape}')

    return original_load(model, state_dict, **kwargs)

ser._load_partial_state_dict = patched_load

from fms.models import get_model

try:
    model = get_model(
        'hf_pretrained',
        'ibm-granite/granite-speech-3.3-2b',
        data_type=torch.float32,
        device_type='cpu',
    )
except Exception as e:
    print(f'Error: {e}')
