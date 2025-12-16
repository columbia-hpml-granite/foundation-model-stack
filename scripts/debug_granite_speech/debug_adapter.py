import torch
import fms.utils.serialization as ser
from collections import ChainMap

# Find the adapter output
original_get_adapter = ser._get_adapter
original_load = ser.load_state_dict_into_model

def patched_load_into_model(model, state_dict, architecture, source, dtype=None, **kwargs):
    print('=== load_state_dict_into_model ===')
    adapter = original_get_adapter(architecture, source)
    adapter_kwargs = {}
    if hasattr(model, "config"):
        adapter_kwargs["model_config"] = model.config
        print(f'Using model_config.encoder_config.output_dim: {model.config.encoder_config.output_dim}')

    # Find encoder.out keys
    sd_keys = set(state_dict.keys())
    for key in sd_keys:
        if key.startswith('encoder.out'):
            print(f'Raw checkpoint {key}: {state_dict[key].shape}')
            partial_sd = {key: state_dict[key]}
            converted = adapter(partial_sd, **adapter_kwargs)
            for ck, cv in converted.items():
                print(f'  -> Converted {ck}: {cv.shape}')

    # Call original
    return original_load(model, state_dict, architecture, source, dtype=dtype, **kwargs)

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
