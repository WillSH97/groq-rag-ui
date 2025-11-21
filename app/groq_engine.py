from groq import Groq
import os

model_list =[
    'openai/gpt-oss-120b',
    'moonshotai/kimi-k2-instruct',
    'meta-llama/llama-4-scout-17b-16e-instruct',
    'openai/gpt-oss-20b',
    'qwen/qwen3-32b',
    'llama-3.3-70b-versatile',
    'moonshotai/kimi-k2-instruct-0905',
    'allam-2-7b',
    'meta-llama/llama-4-maverick-17b-128e-instruct',
    'llama-3.1-8b-instant',
]

reasoning_models = [
    'openai/gpt-oss-120b',
    'openai/gpt-oss-20b',
    'qwen/qwen3-32b',
]

image_models = [
    'meta-llama/llama-4-scout-17b-16e-instruct',
    'meta-llama/llama-4-maverick-17b-128e-instruct',
]

def groq_chat(client, 
              # new_message: str, 
              chat_history: list[dict], 
              model: str, 
              # system_prompt: str, 
              reasoning_level = "not-entered", #has to be weird because of qwen string
              #reasoning_format = "none"
             ):
    '''
    Groq chat API call wrapper, specifically rejigged for RAGChat.

    inputs:
        - client: Groq client (must be created user-side or whatever - in streamlit gui lol)
        - model [str]: model from model_list (cbf static typing from the list)
        - chat_history [list[dict]]: list of dictionaries of chat hist, needs "role" and "content" vars as strings
        - new_message [str]: new user input message (assuming we're only accepting user inputs) 
        - system prompt [str]: optional system prompt for whatever chat you're using

    outputs:
        - nonsys_msg_hist [list[dict]]: updated chat history
    '''
    if model not in model_list:
        raise ValueError(f"model must be in model_list: {model_list}")
        return
    
    msg_hist = [{key: x[key] for key in ["role", "content"] if key in x} for x in chat_history] #clean the chatbot bullshit out
    # for llama 4 - remove all previous images apart from newest message
    if model in image_models:
        for msg in msg_hist[:-1]:  # Exclude last message
            if isinstance(msg["content"], list):
                msg["content"] = [x for x in msg["content"] if x["type"] == "text"]
    
    if model in reasoning_models:
        if reasoning_level == "not-entered":
            reasoning_level = ("default" if model == 'qwen/qwen3-32b' else "medium")
        chat_completion = client.chat.completions.create(
            messages = msg_hist,
            model = model,
            reasoning_effort = reasoning_level,
            reasoning_format = "parsed",
            stream = True,
        )
    else:
        chat_completion = client.chat.completions.create(
            messages = msg_hist,
            model = model,
            # include_reasoning = False, #removes reasoning tokens from output because I'm lazy
            stream = True,
        )
    #toggles for reformatting streamed reasoning
    thinking = (model in reasoning_models) #bool for if thinking is on
    for chunk in chat_completion:
        content = chunk.choices[0].delta.content
        if thinking:
            thinking_tokens = chunk.choices[0].delta.reasoning
            if thinking_tokens:
                yield thinking_tokens
        if content:
            if thinking:
                yield"</think>"
                thinking = False
            yield content  # Yield content for streaming
 