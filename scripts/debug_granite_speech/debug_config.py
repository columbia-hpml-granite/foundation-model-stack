import torch
import fms.utils.serialization as ser

# Patch to see model config
original_load_into_model = ser.load_state_dict_into_model

def patched_load_into_model(model, state_dict, architecture, source, **kwargs):
    print('=== load_state_dict_into_model called ===')
    print(f'Model type: {type(model)}')
    if hasattr(model, 'config'):
        print(f'Model config encoder_config.output_dim: {model.config.encoder_config.output_dim}')
        print(f'Model encoder.out.weight.shape: {model.encoder.out.weight.shape}')
    return original_load_into_model(model, state_dict, architecture, source, **kwargs)

ser.load_state_dict_into_model = patched_load_into_model

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
