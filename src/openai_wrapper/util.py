"""_summary_

Args:
    # openai common args
    model (_type_): the model to use

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
    # for openai acreate method only
    timeout (float, int, optional): The timeout of asnyc_io
    # for ChatCompletion method
    message (_type_, optional): A list of messages comprising the conversation so far. 
                            Defaults to [{"role":"system","content":"你是一个人工智能助手"}, {"role":"user","content":"你好。"}].
    functions (List[function], optional): A list of functions the model may generate JSON inputs for. Defaults to None.
    function_call (Union[str, dict], optional): Controls how the model responds to function calls. 
                                            "none" means the model does not call a function, and responds to the end-user. 
                                            "auto" means the model can pick between an end-user or calling a function. 
                                            Specifying a particular function via {"name":\ "my_function"} forces the model 
                                            to call that function. "none" is the default when no functions are present. 
                                            "auto" is the default if functions are present. Defaults to "auto".

    # for Completion 
    prompt (_type_, optional): The prompt(s) to generate completions for, 
                            encoded as a string, array of strings, array of tokens, or array of token arrays.. 

    #  local_model common args

    # for llama-cpp only
          suffix: A suffix to append to the generated text. If None, no suffix is appended.
    # for hf only

    # for auto-gptq only
                                        
"""
from typing import Union, Dict, List, Tuple
OPENAI_COMMON_ARGS = {
            "model":None, #str
            "stream": True,
            "max_tokens": 2028,
            "temperature":0.1,
            "top_p":0.1,
            "n":1,
            "echo":False,
            "stop":None,
            "presence_penalty":0,
            "frequency_penalty":0,
            "best_of":1,
            "logit_bias":None,

}

OPENAI_COMPLETION_CREATE_ARGS = {
        "prompt":"Hello World!", **OPENAI_COMMON_ARGS
}

OPENAI_CHAT_CREATE_ARGS = {
            "message":[{"role":"system","content":"你是一个人工智能助手"},
                                        {"role":"user","content":"你好。"}],
            "functions": None,
            "function_call":"auto", 
            **OPENAI_COMMON_ARGS
}

OPENAI_CHAT_ACREATE_ARGS = {"timeout": 600,**OPENAI_CHAT_CREATE_ARGS}
OPENAI_COMPLETION_ACREATE_ARGS = {"timout":600, **OPENAI_COMPLETION_CREATE_ARGS}

LOCAL_MODEL_COMMON_ARGS = {
    "repeat_penalty": 1.1,
    "max_tokens": 2028,
    "temperature":0.1,
    "top_p":0.1,
    "top_k": 40,
    "stream": False,
    "echo":False,
    "stop":None,
    "presence_penalty":0,
    "frequency_penalty":0,

}

LLAMA_CPP_COMPLETION_CREATE_ARGS = {
    "suffix": None,
    "tfs_z":1.0,
    "model": None,
    "mirostat_mode":  0,
    "mirostat_tau":  5.0,
    "mirostat_eta":  0.1,
    "model": None,
    "stopping_criteria": None,
    "logits_processor": None,
    "grammar":  None,
}

