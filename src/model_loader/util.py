MODEL_INFO_ARGS = {
    model_name_or_path:None,
    model_cache_dir:str = None,
    tokenizer_name_or_path:str=None,
    tokenizer_cache_dir:str = None,    
    peft_name_or_path:Union[str,Tuple[str],List[str]] = None,
    peft_cache_dir: str = None,

    model_type: str = "HF", # HF, GPTQ, LLAMACPP
    quantized: bool = False,
    model_quant_type: str = "auto", # HF_no_quant,HF_quantized, HF_win_quant,HF_unix_quant,GPTQ_quant, GPTQ_quantized, LLAMACPP
    device:str="auto", # auto, cpu, gpu, mps,
    quant_bit:int=None, # None, 4, 8
    use_fast:bool = False, #

    device_map: Union[str,dict] = "auto", # None,"auto", "balanced", "balanced_low_0", "sequential"
    torch_dtype: Optional[torch.dtype] = torch.float16,

    group_size:int = 128,
    use_triton: bool = False,
    use_safetensors: bool = True,
    quantized_model_path: Optional[str] = None,
    ggml_file_name: Optional[str] = None, # set if you want save your own gptq model

    kwargs: Optional[dict] = None, # all other kwargs
}