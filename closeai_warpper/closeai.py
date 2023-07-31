import os
import sys
import openai
from typing import Union, List, Tuple, Literal,Dict
from llama_cpp import Llama
from auto_gptq import AutoGPTQForCausalLM

def chat(model):
    if hasattr(model, "chat"):
        return model.chat
    elif hasattr(model, "generate"):
        pass
    else:
        pass

def stream_chat(model):
    if hasattr(model,"stream_chat"):
        return model.stream_chat
    elif hasattr(model,"chat"):
        pass

class LlamaCPPWarpper(Llama):
    pass

class GPTQWarpper(AutoGPTQForCausalLM):
    pass




class ChatCompletion(object):
    def __init__(self) -> None:
        pass

    def create(self,
               model,
               message:List[Dict[str,str]]=[{"role":"system","content":"你是一个人工智能助手"},
                                            {"role":"user","content":"你好。"}],
               functions: List[function] = None,
               function_call: Union[str, dict] = "auto", 
               stream: bool=True,
               max_tokens: int=2028,
               temperature:float=0,
               top_p:float=0.1,
               n:int=1,
               stop:Union[str,list,tuple]=None,
               presence_penalty:float=0,
               frequency_penalty:float=0,
               logit_bias:dict=None,
               ):
        """_summary_

        Args:
            model (_type_): the model to use
            message (_type_, optional): A list of messages comprising the conversation so far. 
                                        Defaults to [{"role":"system","content":"你是一个人工智能助手"}, {"role":"user","content":"你好。"}].
            functions (List[function], optional): A list of functions the model may generate JSON inputs for. Defaults to None.
            function_call (Union[str, dict], optional): Controls how the model responds to function calls. 
                                                        "none" means the model does not call a function, and responds to the end-user. 
                                                        "auto" means the model can pick between an end-user or calling a function. 
                                                        Specifying a particular function via {"name":\ "my_function"} forces the model 
                                                        to call that function. "none" is the default when no functions are present. 
                                                        "auto" is the default if functions are present. Defaults to "auto".
            stream (bool, optional): If set, partial message deltas will be sent, like in ChatGPT. 
                                    Tokens will be sent as data-only server-sent events as they become available, 
                                    with the stream terminated by a data: [DONE] message. 
                                    Defaults to True.
            max_tokens (int, optional): The maximum number of tokens to generate in the chat completion. Defaults to 2028.
            temperature (float, optional): What sampling temperature to use, between 0 and 2. 
                                           Higher values like 0.8 will make the output more random, 
                                           while lower values like 0.2 will make it more focused and deterministic.
                                           We generally recommend altering this or top_p but not both. 
                                           Defaults to 0.
            top_p (float, optional): An alternative to sampling with temperature, called nucleus sampling, 
                                    where the model considers the results of the tokens with top_p probability mass. 
                                    So 0.1 means only the tokens comprising the top 10% probability mass are considered. 
                                    Defaults to 0.1.
            n (int, optional): How many chat completion choices to generate for each input message. Defaults to 1.
            stop (Union[str,list,tuple], optional): Up to 4 sequences where the API will stop generating further tokens.
                                                    Defaults to None.
            presence_penalty (float, optional): Number between -2.0 and 2.0. 
                                                Positive values penalize new tokens based on whether they appear in the text so far, 
                                                increasing the model's likelihood to talk about new topics. Defaults to 0.
            frequency_penalty (float, optional): Number between -2.0 and 2.0. 
                                                 Positive values penalize new tokens based on their existing frequency in the text so far, 
                                                 decreasing the model's likelihood to repeat the same line verbatim. 
                                                 Defaults to 0.
            logit_bias (dict, optional): Modify the likelihood of specified tokens appearing in the completion.
                                         Accepts a json object that maps tokens (specified by their token ID in the tokenizer) 
                                         to an associated bias value from -100 to 100. Mathematically, 
                                         the bias is added to the logits generated by the model prior to sampling. 
                                         The exact effect will vary per model, but values between -1 and 1 should decrease or 
                                         increase likelihood of selection; values like -100 or 100 should result in a ban 
                                         or exclusive selection of the relevant token. Defaults to None.
        """
        
        pass

    async def acreate(self,model,prompt,stream=True):
        pass


class Completion(object):
    def __init__(self) -> None:
        pass

    def create(self,
               model,
               prompt="Hello World!",
               stream: bool=True,
               max_tokens: int=2028,
               temperature:float=0,
               top_p:float=0.1,
               n:int=1,
            #    logprob:int=5,
               echo:bool=False,
               stop:Union[str,list,tuple]=None,
               presence_penalty:float=0,
               frequency_penalty:float=0,
               best_of:int=1,
               logit_bias:dict=None,
               ):
        """_summary_

        Args:
            model (_type_): the model to use
            prompt (_type_, optional): The prompt(s) to generate completions for, 
                                        encoded as a string, array of strings, array of tokens, or array of token arrays.. 
            stream (bool, optional): If set, partial message deltas will be sent, like in ChatGPT. 
                                    Tokens will be sent as data-only server-sent events as they become available, 
                                    with the stream terminated by a data: [DONE] message. 
                                    Defaults to True.
            echo (bool, optional): Echo back the prompt in addition to the completion.
            max_tokens (int, optional): The maximum number of tokens to generate in the chat completion. Defaults to 2028.
            temperature (float, optional): What sampling temperature to use, between 0 and 2. 
                                           Higher values like 0.8 will make the output more random, 
                                           while lower values like 0.2 will make it more focused and deterministic.
                                           We generally recommend altering this or top_p but not both. 
                                           Defaults to 0.
            top_p (float, optional): An alternative to sampling with temperature, called nucleus sampling, 
                                    where the model considers the results of the tokens with top_p probability mass. 
                                    So 0.1 means only the tokens comprising the top 10% probability mass are considered. 
                                    Defaults to 0.1.
            n (int, optional): How many chat completion choices to generate for each input message. Defaults to 1.
            stop (Union[str,list,tuple], optional): Up to 4 sequences where the API will stop generating further tokens.
                                                    Defaults to None.
            presence_penalty (float, optional): Number between -2.0 and 2.0. 
                                                Positive values penalize new tokens based on whether they appear in the text so far, 
                                                increasing the model's likelihood to talk about new topics. Defaults to 0.
            frequency_penalty (float, optional): Number between -2.0 and 2.0. 
                                                 Positive values penalize new tokens based on their existing frequency in the text so far, 
                                                 decreasing the model's likelihood to repeat the same line verbatim. 
                                                 Defaults to 0.
            best_of (int, optional): Generates best_of completions server-side and returns the "best" 
                                     (the one with the highest log probability per token). Results cannot be streamed.
                                    when used with `n`, `best_of` controls the number of candidate completions and 
                                    `n` specifies how many to return – `best_of` must be greater than `n`. 
            logit_bias (dict, optional): Modify the likelihood of specified tokens appearing in the completion.
                                         Accepts a json object that maps tokens (specified by their token ID in the tokenizer) 
                                         to an associated bias value from -100 to 100. Mathematically, 
                                         the bias is added to the logits generated by the model prior to sampling. 
                                         The exact effect will vary per model, but values between -1 and 1 should decrease or 
                                         increase likelihood of selection; values like -100 or 100 should result in a ban 
                                         or exclusive selection of the relevant token. Defaults to None.
        """
        pass

    def acreate(self):
        pass

class Embedding(object):
    def __init__(self) -> None:
        pass

    def create(self, model, input="Hello World"):
        pass

    async def acreate(self,model,input=""):
        pass

