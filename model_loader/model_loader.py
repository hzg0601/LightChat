from transformers import (AutoModel,
                        AutoModelForCausalLM,
                        AutoTokenizer,
                        AutoConfig,
                        LlamaTokenizer)

import os
import sys

from typing import Union,Optional
from pathlib import Path
import torch

from .until import (logger,
                    is_ipex_available)


class ModelInfo:
    model_name_or_path:str
    tokenizer_name_or_path:str=model_name_or_path
    model_type: Optional[str] = "HF" # HF, GPTQ, LLAMACPP
    quantized: bool = False

class LoaderInfo:
    device:str="auto" # auto, cpu, gpu, mps,
    quant_bit:Optional[bool]=None # None, 4, 8
    use_fast:bool=False #
    peft_name_or_path:Optional[str] = None
    device_map: Optional[str] = "auto" # "auto", "balanced"
    torch_dtype: Optional[torch.dtype] = torch.float16


def recur_download_model(model_name_or_path:str,max_try:int=300) -> Path:
    """if given a path, return it directly; or else, call `snapshot_download`
        from `huggingface_hub` to download the repo recurrently, until the repo
        is downloaded or the time of try is greater than `max_try`.


    Args:
        model_name_or_path (str): the local_model_path or repo-id from huggingface_hub
        max_try (int, optional): max time of call snapshot_download. Defaults to 300.

    Returns:
        Path: the local path of `model_name_or_path` 
    """

    if os.path.exists(model_name_or_path):
        return model_name_or_path
    else:
        from huggingface_hub import snapshot_download
        turns = 0
        while True:
            try:
                model_name_or_path = snapshot_download(
                    model_name_or_path,resume_download=True)
                return model_name_or_path
            except:
                turns += 1
                if turns > max_try:
                    logger.error(f"""the times of calling `snapshot_download` are greater than `max_try`:{max_try},
                    please check your internet. """)
                    raise ConnectionError


class ModelLoader(object):
    def __init__(self,
                 model_info:ModelInfo,
                loader_info:Optional[LoaderInfo]=None
    ):
        assert not (self.model_info.quantized and self.loader_info.quant_bit), "can not quantize a model that has been quantized"

        if loader_info.device == "auto":
            loader_info.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        assert not(not loader_info.device.lower().startswith("cuda") and loader_info.quant_bit),"quantization can only be executed on gpu"
        
        model_info.model_name_or_path = recur_download_model(model_info.model_name_or_path)
        
        self.model_info = model_info

        if sys.platform == "windows" and loader_info.quant_bit and loader_info.device_map:
            logger.warning("Can only use compression.py to execute quantization in windows currently, however it is not supported for dispatch model")
            loader_info.device_map = None
        use_cuda = torch.cuda.is_available() and loader_info.device.lower().startswith("cuda")
        if use_cuda and (not torch.cuda.device_count() > 2) and loader_info.device_map:
            loader_info.warning("Can only execute dispatching model when `loader_info.device=cuda` and the number of gpus greater than 1 if use gpu")
            loader_info.device_map = None

        use_xpu = is_ipex_available() and torch.xpu.is_available() and loader_info.device.lower().startswith("xpu")
        if use_xpu and (not torch.xpu.device_count() > 1) and loader_info.device_map:
            loader_info.warning("Can only execute dispatching model when `loader_info.device=xpu` and the number of xpus greater than 1 if use xpu")
            loader_info.device_map = None
            
        if use_cuda and (not torch.cuda.is_bf16_supported()) and (loader_info.torch_dtype == torch.bfloat16):
            logger.warning('Your GPU does not support bfloat16, use float16 instead')
            loader_info.torch_dtype = torch.float16

        if use_xpu and loader_info.torch_dtype==torch.float16:
            logger.warning("Intel XPU does not support float16 yet, use bfloat16 instead")
            loader_info.torch_dtype = torch.bfloat16

        if loader_info.peft_name_or_path:
            loader_info.peft_name_or_path = recur_download_model(loader_info.peft_name_or_path)
        
        self.loader_info = loader_info


    def load_hf(self):
        """A function that loads a model of `transformers`
        """
        
        model_config = AutoConfig.from_pretrained(self.model_info.model_name_or_path,trust_remote_code=True)
        kwargs = {}
        if self.model_info.quant_bit and sys.platform != "windows":
            from bitsandbytes import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(load_in_8bit=self.loader_info.quant_bit=8,
                                            load_in_4bit=self.loader_info.quant_bit=4,
                                            llm_int8_threshold=6.0,
                                            llm_int8_has_fp16_weights=False
                                            )
            kwargs["quantization_config"] = quant_config
        if self.loader_info.device_map:
            kwargs["device_map"] = self.loader_info.device_map
        kwargs['torch_dtype'] = self.loader_info.torch_dtype

        # with init_empty_weights():
        try:
            model = AutoModel.from_config(self.model_info.model_name_or_path,
                                            config=model_config,
                                            trust_remote_code=True,
                                            **kwargs)
    
        except NameError:
            model = AutoModelForCausalLM.from_config(self.model_info.model_name_or_path,
                                                        config=model_config,
                                                        trust_remote_code=True,
                                                        **kwargs)
        
