"""
Wrapper of local models, including models that can be loaded by 
llama-cpp-python, auto-gptq, and transformers.
"""
from transformers import (AutoModel,
                        AutoModelForCausalLM,
                        AutoTokenizer,
                        AutoConfig)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Union,Optional,Tuple,List,Literal
from pathlib import Path
import torch
from utils import (logger,
                    is_ipex_available)


class ModelInfo:
    model_name_or_path:str
    model_cache_dir:str = None
    tokenizer_name_or_path:str=None
    tokenizer_cache_dir:str = None    
    peft_name_or_path:Optional[Union[str,Tuple[str],List[str]]] = None
    peft_cache_dir: str = None

    model_type: str = "HF" # HF, GPTQ, LLAMACPP
    quantized: bool = False
    model_quant_type: Optional[str] = "auto" # HF_no_quant,HF_quantized, HF_win_quant,HF_unix_quant,GPTQ_quant, GPTQ_quantized, LLAMACPP
    device:str="auto" # auto, cpu, gpu, mps,
    quant_bit:Optional[bool]=None # None, 4, 8
    use_fast:bool = False #

    device_map: str = "auto" # None,"auto", "balanced", "balanced_low_0", "sequential"
    torch_dtype: Optional[torch.dtype] = torch.float16

    group_size:int = 128
    use_triton: bool = False
    use_safetensors: bool = True
    quantized_model_path: Optional[str] = None 
    ggml_file_name: Optional[str] = None # set if you want save your own gptq model

    kwargs: Optional[dict] = None # all other kwargs

    def __init__(self,**kwargs):
        for key, value in kwargs.items():
            setattr(self,key,value)



def recur_download_model(model_name_or_path:str,max_try:int=300,cache_dir=None) -> Path:
    """if given a path, return it directly; or else, call `snapshot_download`
        from `huggingface_hub` to download the repo recurrently, until the repo
        is downloaded or the time of try is greater than `max_try`.


    Args:
        model_name_or_path (str): the local_model_path or repo-id from huggingface_hub
        max_try (int, optional): max time of call snapshot_download. Defaults to 300.

    Returns:
        Path: the local path of `model_name_or_path` 
    """
    model_path_temp = os.path.join(
            os.getenv("HOME"),
            ".cache/huggingface/hub",
            "models--" + model_name_or_path.replace("/", "--"),
            "snapshots/",
        )
    if os.path.exists(model_name_or_path):
        return model_name_or_path
    elif os.path.exists(model_path_temp):
        return model_name_or_path
    else:
        from huggingface_hub import snapshot_download
        turns = 0
        logger.info(f"Downloading {model_name_or_path}...")
        while True:
            try:
                model_name_or_path = snapshot_download(
                    model_name_or_path,
                    resume_download=True,
                    cache_dir=cache_dir)
                logger.info(f"downloading {model_name_or_path} done.")
                return model_name_or_path
            except:
                logger.info("downloading failed, retry...")
                turns += 1
                if turns > max_try:
                    logger.error(f"""the times of calling `snapshot_download` are greater than `max_try`:{max_try},
                    please check your internet. """)
                    raise ConnectionError


class ModelLoader(object):
    def __init__(self,
                 model_info:ModelInfo
    ):  
        if not model_info.tokenizer_name_or_path:
            model_info.tokenizer_name_or_path = model_info.model_name_or_path

        if model_info.quantized:
            if model_info.quant_bit:
                logger.warning("can not quantize a model that has been quantized,set `quant_bit` be `None` here.")
                model_info.quant_bit = None
        if model_info.model_type == "LLAMACPP":
            if model_info.quant_bit:
                logger.warning("llama-cpp model cannot be quantized again,set `quant_bit` be `None` here.")
                model_info.quant_bit = None

        if model_info.device == "auto":
            model_info.device = ("cuda" if torch.cuda.is_available() else 
                                  "xpu" if torch.xpu.is_available() and is_ipex_available() else 
                                  "mps" if torch.backends.mps.is_available() else 
                                  "cpu")
        if model_info.quant_bit:
           if not ((model_info.device.lower().startswith("cuda") or
                    model_info.device.lower().startswiths("xpu"))):
                logger.error("quantization can only be executed on gpu or xpu")
                raise KeyError
        
        model_info.model_name_or_path = recur_download_model(
            model_info.model_name_or_path,
            cache_dir=model_info.model_cache_dir)
        
        if model_info.model_name_or_path != model_info.tokenizer_name_or_path:
            model_info.tokenizer_name_or_path = recur_download_model(
                model_info.tokenizer_name_or_path,
                cache_dir=model_info.tokenizer_cache_dir)

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
                model_info.peft_name_or_path = recur_download_model(model_info.peft_name_or_path,
                                                                    cache_dir=model_info.peft_cache_dir)
            elif (isinstance(model_info.peft_name_or_path,tuple) or
                  isinstance(model_info.peft_name_or_path,list)):
                temp = []
                for peft_path in model_info.peft_name_or_path:
                    peft_path = recur_download_model(peft_path,cache_dir=model_info.peft_cache_dir)
                    temp.append(peft_path)
                model_info.peft_name_or_path = temp
            else:
                logger.error(f"unsupported type of peft_name_or_path: {model_info.peft_name_or_path}")
                raise TypeError

        self.model_info = model_info
        self.model_quant_check()

    def model_quant_check(self):
        if (sys.platform == "windows") and self.model_info.quant_bit and self.model_info.model_type == "HF":
            self.model_info.model_quant_type = "HF_win_quant"
        elif (sys.platform != "windows") and self.model_info.quant_bit and self.model_info.model_type == "HF":
            self.model_info.model_quant_type = "HF_unix_quant"
        elif (not self.model_info.quant_bit ) and self.model_info.model_type == "HF":
            self.model_info.model_quant_type = "HF_no_quant"
        elif self.model_info.quantized and self.model_info.model_type == "HF":
            self.model_info.model_quant_type =  "HF_quantized"
        elif self.model_info.quant_bit and self.model_info.model_type == "GPTQ":
            self.model_info.model_quant_type == "GPTQ_quant"
        elif (not self.model_info.quant_bit) and self.model_info.model_type == "GPTQ":
            self.model_info.model_quant_type == "GPTQ_quantized"
        elif self.model_info.model_type == "LLAMACPP":
            self.model_info.model_quant_type == "LLAMACPP"

    def load_tokenizer(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_info.tokenizer_name_or_path,
                                                      trust_remote_code=True,
                                                      use_fast=self.model_info.use_fast)
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(self.model_info.tokenizer_name_or_path,
                                                      trust_remote_code=True,
                                                      use_fast=~self.model_info.use_fast)
        return tokenizer
    
    def inspect_params(self,class_or_function,kwargs):
        if not kwargs:
            return {}
        
        import inspect
        class_or_func_kwargs = inspect.getfullargspec(class_or_function).args
        common_set = set(class_or_func_kwargs) & set(kwargs.keys())
        common_kwargs = {key:kwargs[key] for key in common_set}
        unuse_kwargs = set(kwargs.keys()) - common_set
        logger.warning(f"Following args are not compactible with {class_or_function.__name__} thereby unused:")
        logger.warning(unuse_kwargs)
        return common_kwargs


    def check_device_map(self):
        if self.use_cuda:
            num_devices = torch.cuda.device_count()
        if self.use_xpu:
            num_devices = torch.xpu.device_count()
        if self.use_cuda and (num_devices < 2):
            if self.model_info.device_map:
            
                logger.warning("Can only execute dispatching model when `model_info.device=cuda` and the number of gpus greater than 1 if use gpu")
                self.model_info.device_map = None
        
        if self.use_xpu and (num_devices < 2) :
            if self.model_info.device_map:
                logger.warning("Can only execute dispatching model when `model_info.device=xpu` and the number of xpus greater than 1 if use xpu")
                self.model_info.device_map = None

        if not self.use_cuda and not self.use_xpu:
            if self.model_info.device_map:
                logger.warning("can only dispatch model when use cuda or xpu")
                self.model_info.device_map = None

    def load_hf_general(self):
        """Load a model that compactible with `transformers`
        """
        
        model_config = AutoConfig.from_pretrained(self.model_info.model_name_or_path,trust_remote_code=True)
        kwargs = {}
        if self.model_info.quant_bit and sys.platform != "windows":
            from bitsandbytes import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(load_in_8bit=self.model_info.quant_bit==8,
                                            load_in_4bit=self.model_info.quant_bit==4,
                                            llm_int8_threshold=6.0,
                                            llm_int8_has_fp16_weights=False
                                            )
            kwargs["quantization_config"] = quant_config
        if self.model_info.device_map:
            kwargs["device_map"] = self.model_info.device_map
        kwargs['torch_dtype'] = self.model_info.torch_dtype
        used_kwargs = self.inspect_params(AutoModel.from_config, self.model_info.kwargs)
        kwargs = {**kwargs, **used_kwargs}
        try:
            model = AutoModel.from_pretrained(self.model_info.model_name_or_path,
                                            config=model_config,
                                            trust_remote_code=True,
                                            **kwargs)
        except NameError:
            model = AutoModelForCausalLM.from_pretrained(self.model_info.model_name_or_path,
                                                        config=model_config,
                                                        trust_remote_code=True,
                                                        **kwargs)
        except ImportError:
            model = AutoModelForCausalLM.from_pretrained(self.model_info.model_name_or_path,
                                                        config=model_config,
                                                        trust_remote_code=False,
                                                        **kwargs)
        tokenizer = self.load_tokenizer()
            
        return model, tokenizer

    def load_hf_win_quant(self) :
        from compression import load_compress_model

        model,tokenizer = load_compress_model(model_path=self.model_info.model_name_or_path,
                                                tokenizer_path=self.model_info.tokenizer_name_or_path,
                                                device=self.model_info.device,
                                                use_fast=self.model_info.use_fast)
        return model, tokenizer
            

    def load_llamacpp(self):
        """load a llama-cpp model based on llama-cpp-python library"""
        from llama_cpp import Llama,LlamaTokenizer
        import re
        pattern = self.model_info.ggml_file_name if self.model_info.ggml_file_name else "*ggml*.bin"
        for dir_name,sub_dir, file in os.walk(self.model_info.model_name_or_path):
            
            if re.search(pattern,file):
                file_path = os.path.join(dir_name,sub_dir)
                break
        used_kwargs = self.inspect_params(Llama,self.model_info.kwargs)
        used_kwargs['model_path'] = file_path

        if self.use_cuda or self.use_xpu:
            if "n_gpu_layers" not in used_kwargs:
                used_kwargs['n_gpu_layers'] = -1
        if "lora_path" not in used_kwargs:
            used_kwargs['lora_path'] = self.peft_path

        model = Llama(**used_kwargs)

        tokenizer = LlamaTokenizer(model)
        return model, tokenizer
    
    def load_gptq_quantized(self):
        """ load a quantized gptq model"""
        assert self.use_cuda, "gptq models are only supported on cuda device"
        from auto_gptq import AutoGPTQForCausalLM
        tokenizer = self.load_tokenizer()
        model = AutoGPTQForCausalLM.from_quantized(self.model_info.model_name_or_path,
                                                    device_map=self.model_info.device_map,
                                                    trust_remote_code=True,
                                                    device=self.model_info.device,
                                                    use_triton=False,
                                                    quantize_config=None)
        return model, tokenizer
    
    def load_gptq_and_quantize(self):
        """load and quantize a gptq model"""
        assert self.use_cuda, "gptq models are only supported on cuda device"
        from auto_gptq import AutoGPTQForCausalLM,BaseQuantizeConfig
        tokenizer = self.load_tokenizer()
        quantize_config = BaseQuantizeConfig(bits=self.model_info.quant_bit,
                                             group_size=self.model_info.group_size,
                                             desc_act=False)
        examples = [tokenizer(
                    "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
                    )]
        used_kwargs = self.inspect_params(AutoGPTQForCausalLM.from_pretrained, self.model_info.kwargs)
        
        model = AutoGPTQForCausalLM.from_pretrained(self.model_info.model_name_or_path,
                                                    quantize_config=quantize_config,
                                                    **used_kwargs)
        
        model.quantize(examples)
        if self.model_info.quantized_model_path:
            model.save_quantized(self.model_info.quantized_model_path,
                                 use_safetensors = self.model_info.use_safetensors)
        return model, tokenizer

    def load_peft(self,model):
        """load a peft based on peft.PeftModel"""
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
        logger.info(f"loading model {self.model_info.model_name_or_path}...")
        if self.model_info.model_quant_type in ("HF_no_quant","HF_quantized","HF_unix_quant") :
            model,tokenizer = self.load_hf_general()
        elif self.model_info.model_quant_type == "HF_win_quant":
            model,tokenizer = self.load_hf_win_quant()
        elif self.model_info.model_quant_type == "GPTQ_quantized":
            model, tokenizer = self.load_gptq_quantized()
        elif self.model_info.model_quant_type == "GPTQ_quant":
            model, tokenizer = self.load_gptq_and_quantize()
        elif self.model_info.model_quant_type == "LLAMACPP":
            model,tokenizer = self.load_llamacpp()
        else:
            print(("Unsupported model_type, quantized and quant_bits combination!,"
                   "`model_quant_type` must be one of "
                   "`HF_no_quant,HF_quantized,HF_win_quant,HF_unix_quant,GPT_quantized,GPTQ_quant,LLAMACPP` or `auto`."
                   ))
            raise TypeError
        if "HF" in self.model_info.model_quant_type and self.model_info.peft_name_or_path:
            model = self.load_peft(model)
        logger.info(f"Model {self.model_info.model_name_or_path} is loaded.") 
        return model, tokenizer
    
if __name__ == "__main__":
    model_info = ModelInfo(model_name_or_path="THUDM/chatglm2-6b")

    loader = ModelLoader(model_info)
    model,tokenizer = loader()
    print("done!")