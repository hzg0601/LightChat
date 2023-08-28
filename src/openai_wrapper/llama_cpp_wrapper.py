import os
import sys
from typing import Union, List, Tuple, Literal,Dict,Optional
from llama_cpp import Llama
from model_loader.local_model_loader import ModelInfo, ModelLoader
import torch
from utils.util import get_common_args
from util import (OPENAI_CHAT_ACREATE_ARGS,
                  OPENAI_CHAT_CREATE_ARGS,
                  OPENAI_COMPLETION_ACREATE_ARGS,
                  OPENAI_COMPLETION_ARGS)
from model_loader.util import MODEL_INFO_ARGS

class BaseClass(object):
    def __init__(self,
                **MODEL_INFO_ARGS
                 ) -> None:
        model_info = ModelInfo(**MODEL_INFO_ARGS)
        self.model_info = model_info
        self.model,self.tokenizer = ModelLoader(model_info=model_info)
        

class ChatCompletion(object):
    def __init__(self,
                **MODEL_INFO_ARGS
                 ) -> None:
        super().__init__(self,MODEL_INFO_ARGS)


    def create(self,
            **OPENAI_CHAT_CREATE_ARGS,
               ):
        pass

    async def acreate(self,               
                **OPENAI_CHAT_ACREATE_ARGS
               ):

        pass


class Completion(object):
    def __init__(self,
                **MODEL_INFO_ARGS
                 ) -> None:
        super().__init__(self,MODEL_INFO_ARGS)

    def create(self,
        **OEPNAI_COMPLETION_CREATE_ARGS):

        model = self.model_info

    async def acreate(self,
            **OPENAI_COMPLETION_ACREATE_ARGS
            ):
        pass

class Embedding(object):
    def __init__(self,
                **MODEL_INFO_ARGS
                 ) -> None:
        super().__init__(self,MODEL_INFO_ARGS)

    def create(self, model, input="Hello World"):
        pass

    async def acreate(self,model,input="Hello World"):
        pass
