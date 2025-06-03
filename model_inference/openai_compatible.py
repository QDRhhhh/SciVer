from utils.api_utils import generate_from_openai_chat_completion
from utils.input_processing import *
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT
from openai import AsyncOpenAI
import os
import json
from tqdm import tqdm
import time
from dotenv import load_dotenv
load_dotenv()
# import logging 
import asyncio

def generate_response(model_name: str, 
                    prompt: str,
                    queries: list,
                    output_path: str,
                    n: int=1):
    base_rate = 500
    if "gemini" in model_name:
        api_key = os.getenv("GEMINI_API_KEY")
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        if "2.0" in model_name or "2.5" in model_name:
            base_rate = 100

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url
    )

    messages = prepare_qa_inputs(model_name, queries, prompt=prompt)

    responses = asyncio.run(
        generate_from_openai_chat_completion(
            client,
            messages=messages, 
            engine_name=model_name,
            max_tokens=MAX_TOKENS if model_name != "gemini-2.0-flash-thinking-exp-1219" else 8192,
            temperature=GENERATION_TEMPERATURE,
            requests_per_minute = int(base_rate),
            n = n
        )
    )

    for query, response in zip(queries, responses):
        query["response"] = response
    
    json.dump(queries, open(output_path, "w"), indent=4, ensure_ascii=False)