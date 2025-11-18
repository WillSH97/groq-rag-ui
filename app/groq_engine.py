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

def groq_chat(client, 
              # new_message: str, 
              chat_history: list[dict], 
              model: str, 
              documents: list[dict] = [],
              # system_prompt: str, 
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

    
    if model in reasoning_models:
        chat_completion = client.chat.completions.create(
            messages = msg_hist,
            model = model,
            documents = documents,
            include_reasoning = False, #removes reasoning tokens from output because I'm lazy
            stream = True,
        )
    else:
        chat_completion = client.chat.completions.create(
            messages = msg_hist,
            model = model,
            documents = documents,
            # include_reasoning = False, #removes reasoning tokens from output because I'm lazy
            stream = True,
        )
    for chunk in chat_completion:
         content = chunk.choices[0].delta.content
         if content:
             yield content  # Yield content for streaming
 