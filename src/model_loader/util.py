import torch

MODEL_INFO_ARGS = {
    "model_name_or_path":None,
    "model_cache_dir": None,
    "tokenizer_name_or_path":None,
    "tokenizer_cache_dir": None,    
    "peft_name_or_path": None,
    "peft_cache_dir":  None,

    "model_type":  "HF", # HF, GPTQ, LLAMACPP
    "quantized": False,
    "model_quant_type": "auto", # HF_no_quant,HF_quantized, HF_win_quant,HF_unix_quant,GPTQ_quant, GPTQ_quantized, LLAMACPP
    "device":"auto", # auto, cpu, gpu, mps,
    "quant_bit":None, # None, 4, 8
    "use_fast":False, #

    "device_map": "auto", # None,"auto", "balanced", "balanced_low_0", "sequential"
    "torch_dtype":  torch.float16,

    "group_size": 128,
    "use_triton": False,
    "use_safetensors": True,
    "quantized_model_path": None,
    "ggml_file_name": None, # set if you want save your own gptq model

    "kwargs": None, # all other kwargs
}