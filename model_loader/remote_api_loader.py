"""
Load python library of remote APIs 
"""
import os
import sys
import subprocess
from utils import get_common_args,logger
from typing import List, Dict

class RemoteAPILoader(object):

    def chat_completion(self):
        logger.error("Chat Completion is not supported currently!")
        raise AttributeError
    
    def completion(self):
        logger.error("Completion is not supported currently!")
        raise AttributeError


    def embedding(self):
        logger.error("Embedding is not supported currently!")
        raise AttributeError

class ZhipuAPILoader(object):
    def __init__(self,
                 api_key:str=None) -> None:
        api_key = os.environ.get("ZHIPUAI_API_KEY", api_key)
        import zhipuai
        zhipuai.api_key = api_key
        logger.info("Loading ZhipuAI...")

    def chat_completion_create(self,
                    prompt: List[Dict[str,str]]=[{"role":"system","content":"你是一个人工智能助手"},
                                            {"role":"user","content":"你好。"}],
                    model:str = "chatglm_pro",
                    top_p:float=0.7,
                    temperature:float=0.9,
                    stream:bool=False,
                    **kwargs):
        if stream:
            response = zhipuai.model_api.sse_invoke(
                model = model,
                prompt = prompt,
                top_p = top_p,
                temperature = temperature,
                incremental = True
            )
            for event in response.events():
                if event.event == "add":
                    yield event.data
                elif event.event == "error" or event.event == "interrupted":
                    return event.data
                elif event.event == "finish":
                    yield event.data
                    return event.meta
                else:
                    print(event.data)

        else:
            response = zhipuai.model_api.invoke(
                model = model,
                prompt = prompt,
                top_p = top_p,
                temperature = temperature
            )
            if response["code"] == 200:
                return response["msg"]
            else:
                logger.info(f"error occurred, error code:{response['code']},error msg:{response['msg']}")
                return

    def chat_completion_acreate(self,
                    model:str="chatglm_pro",
                    top_p:float=0.7,
                    temperature:float=0.9,
                    **kwargs):
        response = zhipuai.model_api.async_invoke(
        model = model,
        prompt = prompt,
        top_p = top_p
        temperature = temperature
        )

        if response["code"] == 200:
            yield response["msg"]
        else:
            logger.info(f"error occurred, error code:{response['code']},error msg:{response['msg']}")
            return 
        
    def completion_create(self,
                    top_p=0.7,
                    temperature=0.9,
                    stream:bool=False,
                    **kwargs):
        pass

    def completion_acreate(self,
                    top_p=0.7,
                    temperature=0.9,
                    stream:bool=False,
                    **kwargs):
        pass


        
        

        

