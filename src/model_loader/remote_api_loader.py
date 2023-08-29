"""
Load python library of remote APIs 
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.util import logger
from typing import List, Dict,Union
import inspect
import asyncio


class BaseAPILoader(object):

    def prompt_collator(self,
                        content_user: str = None,
                        role_user:str = "user",
                        content_assistant: str = None,
                        role_assistant: str = "assistant",
                        meta_prompt:List[Dict[str,str]] = [{"role":"system","content":"你是一个AI工具"}],
                        use_meta_prompt:bool=False):
        prompt = []
        if use_meta_prompt:
            prompt += meta_prompt
        if content_user:
            prompt_dict = {"role": role_user, "content":content_user}
            prompt.append(prompt_dict)
        if content_assistant:
            prompt_dict = {"role": role_assistant, "content":content_assistant}
            prompt.append(prompt_dict)
        return prompt
    
    def create_chat_completion(self):
        logger.error("Chat Completion is not supported currently!")
        raise AttributeError
    
    def create_completion(self):
        logger.error("Completion is not supported currently!")
        raise AttributeError
    
    def acreate_chat_completion(self):
        logger.error("Chat Completion is not supported currently!")
        raise AttributeError
    
    def acreate_completion(self):
        logger.error("Completion is not supported currently!")
        raise AttributeError

    def embedding(self):
        logger.error("Embedding is not supported currently!")
        raise AttributeError

class ZhipuAILoader(BaseAPILoader):
    
    def __init__(self,
                 api_key:str=None) -> None:
        api_key = os.environ.get("ZHIPUAI_API_KEY", api_key)
        import zhipuai
        self.zhipuai = zhipuai
        self.zhipuai.api_key = api_key
        logger.info("Loading ZhipuAI...")
    # 函数里只要有yield就会返回一个迭代器
    # 需要额外用一个streamer才能实现，transformers.

    def create_oneshot(self,
                    message: List[Dict[str,str]]=[{"role":"user","content":"你好，你可以做什么"}],
                    model:str = "chatglm_pro",
                    top_p:float=0.7,
                    temperature:float=0.9,
                    **kwargs
                    ):
            response = self.zhipuai.model_api.invoke(
                model = model,
                prompt = message,
                top_p = top_p,
                temperature = temperature
            )
            if response["code"] == 200:
                result = response["data"]["choices"][-1]["content"]
                return result
            else:
                logger.info(f"error occurred, error code:{response['code']},error msg:{response['msg']}")
                return
            
    def create_stream(self,
                    message: List[Dict[str,str]]=[{"role":"user","content":"你好，你可以做什么"}],
                    model:str = "chatglm_pro",
                    top_p:float=0.7,
                    temperature:float=0.9,
                    **kwargs
                    ):
            response = self.zhipuai.model_api.sse_invoke(
                model = model,
                prompt = message,
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
                    logger.info(event.meta)
                else:
                    logger.error("Something get wrong with ZhipuAPILoader.create_chat_completion_stream")
                    logger.error(event.data)
               
    def create_chat_completion(self,
                               model: str = "chatglm_pro",
                               message:List[Dict[str,str]]=[{"role":"user","content":"你好，你可以做什么"}],
                                top_p:float=0.7,
                                temperature:float=0.9,
                                stream:bool=False,
                               **kwargs):

        if stream:
            return self.create_stream(model=model,
                                    message=message,
                                    top_p=top_p,
                                    temperature=temperature)
        else:
            return self.create_oneshot(model=model,
                                        message=message,
                                        top_p=top_p,
                                        temperature=temperature)
        
    async def acreate_chat_completion(self,
                    prompt: List[Dict[str,str]]=[{"role":"system","content":"你是一个人工智能助手"},
                                            {"role":"user","content":"你好。"}],
                    model:str = "chatglm_pro",
                    top_p:float=0.7,
                    temperature:float=0.9,
                    **kwargs):
        response = await self.zhipuai.model_api.async_invoke(
                    model = model,
                    prompt = prompt,
                    top_p = top_p,
                    temperature = temperature
                    )

        if response["code"] == 200:
            task_id = response['data']['task_id']
            status = "PROCESSING"
            while status != "SUCCESS":
                # await asyncio.sleep(3) # 
                resp = self.zhipuai.model_api.query_async_invoke_result(task_id)
                status = resp['data']['task_status']
            return resp['data']['choices'][-1]['content']
        else:
            logger.info(f"error occurred, error code:{response['code']},error msg:{response['msg']}")
            return 
        
    def create_completion(self,
                               prompt:str="你好",
                               model:str="chatglm_pro",
                               top_p:float=0.7,
                               temperature:float=0.9,
                               stream:bool=False,
                               **kwargs):
        message = self.prompt_collator(content_user=prompt)
        if stream:
            return self.create_stream(model=model,
                                    message=message,
                                    top_p=top_p,
                                    temperature=temperature)
        else:
            return self.create_oneshot(model=model,
                                    message=message,
                                    top_p=top_p,
                                    temperature=temperature)
    #? make it a sync function?    
    async def acreate_completion(self,
                    prompt:str="你好",
                    model:str = "chatglm_pro",
                    top_p:float=0.7,
                    temperature:float=0.9,
                    **kwargs):
        message = self.prompt_collator(content_user=prompt)
        response = self.zhipuai.model_api.async_invoke(
                    model = model,
                    prompt = message,
                    top_p = top_p,
                    temperature = temperature
                    )

        if response["code"] == 200:
            task_id = response['data']['task_id']
            status = "PROCESSING"
            while status != "SUCCESS":
                # await asyncio.sleep(3) # 
                resp = self.zhipuai.model_api.query_async_invoke_result(task_id)
                status = resp['data']['task_status']
            return resp['data']['choices'][-1]['content']
        else:
            logger.info(f"error occurred, error code:{response['code']},error msg:{response['msg']}")
            return 
    

API_TYPE_DICT = {
    "ZhipuAI": ZhipuAILoader
}

class RemoteAPILoader(object):
    def __init__(self,
                 api_type:str = "ZhipuAI",
                 api_key:str = None
                 ) -> None:
        self.api_type = api_type
        remote_api = API_TYPE_DICT[api_type](api_key=api_key)
        function_variables = [name for name, value in inspect.getmembers(remote_api)
                      if inspect.ismethod(value) or name == "__init__"]
        for _attr in function_variables:
            setattr(self,_attr,getattr(remote_api,_attr))
        logger.info(f"Loading remote api {api_type} done!")



if __name__ == "__main__":
    chatglm_pro = RemoteAPILoader(api_key="319eebee38566a54715a45018d0c8cb3.7DasTJjucxFwdwzQ")
    # res = chatglm_pro.create_completion(prompt="你能做什么",stream=False)
    # print(res)
    import time
    start = time.time()
    res = asyncio.run(chatglm_pro.acreate_completion(prompt="你能做什么"))
    end = time.time()
    print(f"edurance:{end-start}")
    print(res)
    print("done!")


        
        

        

