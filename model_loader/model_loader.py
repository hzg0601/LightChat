from transformers import (AutoModel,
                        AutoModelForCausalLM,
                        AutoTokenizer,
                        AutoConfig,
                        LlamaTokenizer)

import os
import sys

from typing import Union,Optional,Tuple,List,Literal
from pathlib import Path
import torch

from ..utils import (logger,
                    is_ipex_available)


class ModelInfo:
    model_name_or_path:str
    tokenizer_name_or_path:str=model_name_or_path
    model_type: Optional[str] = "HF" # HF, GPTQ, LLAMACPP
    quantized: bool = False
    device:str="auto" # auto, cpu, gpu, mps,
    quant_bit:Optional[bool]=None # None, 4, 8
    use_fast:bool=False #
    peft_name_or_path:Optional[Union[str,Tuple[str],List[str]]] = None
    device_map: str = "auto" # None,"auto", "balanced", "balanced_low_0", "sequential"
    torch_dtype: Optional[torch.dtype] = torch.float16
    kwargs: Optional[dict] = None
    group_size:int = 128
    use_triton: bool = False
    use_safetensors: bool = True
    quantized_model_path: Optional[str] = None 
    ggml_file_name: Optional[str] = None


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
                 model_info:ModelInfo
    ):
        assert not (self.model_info.quantized and self.model_info.quant_bit), "can not quantize a model that has been quantized"

        if model_info.device == "auto":
            model_info.device = ("cuda" if torch.cuda.is_available() else 
                                  "xpu" if torch.xpu.is_available() and is_ipex_available() else 
                                  "mps" if torch.backends.mps.is_available() else 
                                  "cpu")
        if model_info.quant_bit:
            assert (model_info.device.lower().startswith("cuda") or
                    model_info.device.lower().startswiths("xpu"),
                    "quantization can only be executed on gpu or xpu")
        
        model_info.model_name_or_path = recur_download_model(model_info.model_name_or_path)
        if model_info.model_name_or_path != model_info.tokenizer_name_or_path:
            model_info.tokenizer_name_or_path = recur_download_model(model_info.tokenizer_name_or_path)
        
        self.model_info = model_info

        if (sys.platform == "windows") and self.model_info.quant_bit:
            self.win_quant = True

        self.use_cuda = torch.cuda.is_available() and model_info.device.lower().startswith("cuda")
        self.use_xpu = (is_ipex_available() and 
                        torch.xpu.is_available() and 
                        model_info.device.lower().startswith("xpu"))

        if self.use_cuda and (not torch.cuda.is_bf16_supported()) :
            if (model_info.torch_dtype == torch.bfloat16):
                logger.warning('Your GPU does not support bfloat16, use float16 instead')
                model_info.torch_dtype = torch.float16

        if self.use_xpu:
            if model_info.torch_dtype==torch.float16:
                logger.warning("Intel XPU does not support float16 yet, use bfloat16 instead")
                model_info.torch_dtype = torch.bfloat16

        if model_info.peft_name_or_path:
            if isinstance(model_info.peft_name_or_path,str):
                model_info.peft_name_or_path = recur_download_model(model_info.peft_name_or_path)
            elif (isinstance(model_info.peft_name_or_path,tuple) or
                  isinstance(model_info.peft_name_or_path,list)):
                temp = []
                for peft_path in model_info.peft_name_or_path:
                    peft_path = recur_download_model(peft_path)
                    temp.append(peft_path)
                model_info.peft_name_or_path = temp
            else:
                logger.error(f"unsupported type of peft_name_or_path: {model_info.peft_name_or_path}")
                raise TypeError

        
        self.model_info = model_info

    def hf_args_check(self,model_info):
        pass

    def check_device_map(self,model_info):
        if self.use_cuda:
            num_devices = torch.cuda.device_count()
        if self.use_xpu:
            num_devices = torch.xpu.device_count()
        if self.use_cuda and (num_devices < 2):
            if model_info.device_map:
            
                model_info.warning("Can only execute dispatching model when `model_info.device=cuda` and the number of gpus greater than 1 if use gpu")
                model_info.device_map = None
        
        if self.use_xpu and (num_devices < 2) :
            if model_info.device_map:
                model_info.warning("Can only execute dispatching model when `model_info.device=xpu` and the number of xpus greater than 1 if use xpu")
                model_info.device_map = None

        if not self.use_cuda and not self.use_xpu:
            if model_info.device_map:
                model_info.warning("can only dispatch model when use cuda or xpu")
                model_info.device_map = None

    def load_hf(self):
        """A function that loads a model from `transformers`
        """
        
        model_config = AutoConfig.from_pretrained(self.model_info.model_name_or_path,trust_remote_code=True)
        kwargs = {}
        if self.model_info.quant_bit and sys.platform != "windows":
            from bitsandbytes import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(load_in_8bit=self.model_info.quant_bit=8,
                                            load_in_4bit=self.model_info.quant_bit=4,
                                            llm_int8_threshold=6.0,
                                            llm_int8_has_fp16_weights=False
                                            )
            kwargs["quantization_config"] = quant_config
        if self.model_info.device_map:
            kwargs["device_map"] = self.model_info.device_map
        kwargs['torch_dtype'] = self.model_info.torch_dtype

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
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_info.tokenizer_name_or_path,
                                                      trust_remote_code=True,
                                                      use_fast=self.model_info.use_fast)
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(self.model_info.tokenizer_name_or_path,
                                                      trust_remote_code=True,
                                                      use_fast=~self.model_info.use_fast)
            
        return model, tokenizer

    def load_win_hf_compression(self) :
        from compression import load_compress_model
        model,tokenizer = load_compress_model(model_path=self.model_info.model_name_or_path,
                                                tokenizer_path=self.model_info.tokenizer_name_or_path,
                                                device=self.model_info.device,
                                                use_fast=self.model_info.use_fast)
        return model, tokenizer
            

    def load_llamacpp(self):
        from llama_cpp import Llama
        import re
        pattern = self.model_info.ggml_file_name if self.model_info.ggml_file_name else "*ggml*.bin"
        for dir_name,sub_dir, file in os.walk(self.model_info.model_name_or_path):
            
            if re.search(pattern,file):
                file_path = os.path.join(dir_name,sub_dir,file)
                break
        if self.use_cuda or self.use_xpu:
            model = Llama(model_path=file_path,n_gpu_layers=2000)
        else:
            model = Llama(model_path=file_path)
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_info.tokenizer_name_or_path,
                                                  use_fast=self.model_info.use_fast,
                                                  trust_remote_code=True)
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(self.model_info.tokenizer_name_or_path,
                                                  use_fast=~self.model_info.use_fast,
                                                  trust_remote_code=True)
        return model, tokenizer
    
    def load_quantized_gptq(self):
        assert self.use_cuda, "gptq models are only supported on cuda device"
        from auto_gptq import AutoGPTQForCausalLM
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_info.tokenizer_name_or_path,
                                                  use_fast=self.model_info.use_fast,
                                                  trust_remote_code=True)
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(self.model_info.tokenizer_name_or_path,
                                                  use_fast=~self.model_info.use_fast,
                                                  trust_remote_code=True)
        model = AutoGPTQForCausalLM.from_quantized(self.model_info.model_name_or_path,
                                                    device_map=self.model_info.device_map,
                                                    trust_remote_code=True,
                                                    device=self.model_info.device,
                                                    use_triton=False,
                                                    quantize_config=None)
        return model, tokenizer
        pass
    
    def load_and_quantize_gptq(self):
        assert self.use_cuda, "gptq models are only supported on cuda device"
        from auto_gptq import AutoGPTQForCausalLM,BaseQuantizeConfig
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_info.tokenizer_name_or_path,
                                                  use_fast=self.model_info.use_fast,
                                                  trust_remote_code=True)
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(self.model_info.tokenizer_name_or_path,
                                                  use_fast=~self.model_info.use_fast,
                                                  trust_remote_code=True)
        quantize_config = BaseQuantizeConfig(bits=self.model_info.quant_bit,
                                             group_size=self.model_info.group_size,
                                             desc_act=False)
        examples = [tokenizer(
                    "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
                    )]
        model = AutoGPTQForCausalLM.from_pretrained(self.model_info.model_name_or_path,
                                                    quantize_config)
        
        model.quantize(examples)
        if self.model_info.quantized_model_path:
            model.save_quantized(self.model_info.quantized_model_path,
                                 use_safetensors = self.model_info.use_safetensors)
        return model, tokenizer

    def load_peft(self,model):
        from peft import PeftModel
        if isinstance(self.model_info.peft_name_or_path,str):
            model = PeftModel.from_pretrained(model,
                                              self.model_info.peft_name_or_path,
                                              is_trainable=False)
        else:
            for peft_path in self.model_info.peft_name_or_path:
                model = PeftModel.from_pretrained(model,
                                                  peft_path,
                                                  is_trainable=False)

        return model

    def __call__(self):
        if self.model_info.model_type == "HF":
            pass
        elif self.model_info.model_type == "GPTQ":
            pass
        elif self.model_info.model_type == "LLAMACPP":
            pass
        else:
            print("Unspported currently!")
            raise TypeError
        

