from transformers import AutoTokenizer
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.utils import FlexibleArgumentParser

from argparse import Namespace
from typing import List
import torch
from transformers import AutoProcessor, AutoTokenizer

from vllm.assets.image import ImageAsset
from vllm.multimodal.utils import fetch_image
import os
import hashlib
import base64
import requests
from tqdm import tqdm
from utils.input_processing import prepare_qa_text_input, prepare_single_image_input,prepare_image_input

def prepare_qwen2_inputs(
    model_name,
    queries,
    prompt,
    temperature: float = 1,
    max_tokens: int = 1024
):
    """
    Prepare single-image QA inputs for Qwen2-VL-7B-Instruct or similar models.
    """
    inputs = []
    llm = LLM(
        model=model_name,
        tensor_parallel_size=min(torch.cuda.device_count(), 4),
        # max_model_len=49152,
        limit_mm_per_prompt={"image": 2}  # Only allow 1 image
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop_token_ids=None
    )
    
    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=256*28*28,
        max_pixels=1280*28*28
    )

    for query in tqdm(queries, desc="Prepare model inputs"):
        # 1. Prepare text QA prompt
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
        
        # 2. Retrieve single base64 image (using your custom function)
        vision_input = prepare_image_input(model_name, query)
        urls = []
        if query['claim_type']=='direct' or query['claim_type']=='analytical':
            urls.append(query['image_path'])
        else:
            urls.append(query['item1_path'])
            urls.append(query['item2_path'])

        placeholders = [{"type": "image", "image": url} for url in urls]
        messages = [
            {
                "role": "user",
                "content": [
                    *placeholders,
                    {"type": "text", "text": qa_text_prompt},
                ],
            }
        ]
        
        # Use Qwen's AutoProcessor to form a proper text input with generation prompt
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        item_input = {
            "prompt": text_input,
            "multi_modal_data": {
                "image": vision_input
            },
        }
        inputs.append(item_input)
    
    return inputs, llm, sampling_params


def prepare_phi3v_inputs(
    model_name,
    queries,
    prompt,
    temperature: float = 1,
    max_tokens: int = 1024
):
    """
    Prepare single-image QA inputs for Microsoft Phi-3.5-vision-instruct or similar models.
    """
    inputs = []
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 2},
        tensor_parallel_size=min(torch.cuda.device_count(),4),
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop_token_ids=None
    )
    
    for query in tqdm(queries, desc="Prepare model inputs"):
        # QA text
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)

        # Convert base64 to actual image object
        # single_img_b64 = prepare_single_image_input(model_name, query['image_path'])
        # vision_input = [fetch_image(f"data:image/jpeg;base64,{single_img_b64}")]
        vision_input = prepare_image_input(model_name, query)

        # Placeholders for Phi-3.5 format
        placeholders = "\n".join(f"<|image_{i}|>" for i, _ in enumerate(vision_input, start=1))
        text_input = f"<|user|>\n{placeholders}\n{qa_text_prompt}<|end|>\n<|assistant|>\n"
        
        item_input = {
            "prompt": text_input,
            "multi_modal_data": {
                "image": vision_input
            },
        }
        inputs.append(item_input)

    return inputs, llm, sampling_params


def prepare_general_vlm_inputs(
    model_name,
    queries,
    prompt,
    temperature: float = 1,
    max_tokens: int = 1024
):
    """
    Prepare single-image QA for a general VLM-based model (e.g., some HF large models).
    """
    inputs = []

    if "molmo" in model_name.lower():
        max_model_len = 4096
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=max_model_len,
            limit_mm_per_prompt={"image": 2},
            tensor_parallel_size=min(torch.cuda.device_count(),4),
        )
    else:
        # max_model_len = 65536
        max_model_len = 1024*32
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=max_model_len,
            limit_mm_per_prompt={"image": 2},
            tensor_parallel_size=min(torch.cuda.device_count(),4),
        )

    for query in tqdm(queries, desc="Prepare model inputs"):
        # QA text
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)

        # Single image
        # single_img_b64 = prepare_single_image_input(model_name, query['image_path'])
        # vision_input = [fetch_image(f"data:image/jpeg;base64,{single_img_b64}")]
        vision_input = prepare_image_input(model_name, query)
        placeholders = "\n".join(f"Image-{i}: <image>\n"
                                 for i, _ in enumerate(vision_input, start=1))
        messages = [{'role': 'user', 'content': f"{placeholders}\n{qa_text_prompt}"}]
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        item_input = {
            "prompt": text_input,
            "multi_modal_data": {
                "image": vision_input
            },
        }
        inputs.append(item_input)

    if "h2oai" in model_name:
        stop_token_ids = [tokenizer.eos_token_id]
    else:
        stop_token_ids = None

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids
    )

    return inputs, llm, sampling_params


def prepare_pixtral_inputs(
    model_name,
    queries,
    prompt,
    temperature: float = 1,
    max_tokens: int = 1024
):
    """
    Prepare single-image QA for Pixtral models.
    """
    stop_token_ids = None
    llm = LLM(
        model=model_name,
        max_model_len=8192*4*4,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": 2},
        tensor_parallel_size=min(torch.cuda.device_count(),4)
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids
    )
    
    inputs = []
    for query in tqdm(queries, desc="Prepare model inputs"):
        # QA text
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)

        # single_img_b64 = prepare_single_image_input(model_name, query['image_path'])
        vision_input = prepare_image_input(model_name, query)
        placeholders = "[IMG]" * len(vision_input)
        prompt_info = f"<s>[INST]{qa_text_message}\n{placeholders}[/INST]"
        # For Pixtral, the original logic places "[IMG]" placeholders
        # cur_input = f"<s>[INST]{qa_text_prompt}\n[IMG][/INST]"
        item_input = {
            "prompt": prompt_info,
            "multi_modal_data": {
                "image": vision_input
            },
        }
        inputs.append(item_input)
        
    return inputs, llm, sampling_params


def prepare_mllama_inputs(
    model_name,
    queries,
    prompt,
    temperature: float = 1,
    max_tokens: int = 1024
):
    """
    Prepare single-image QA for M-LLaMA or related models.
    """
    inputs = []
    llm = LLM(
        model=model_name,
        limit_mm_per_prompt={"image": 2},
        max_model_len=8192*4,
        max_num_seqs=2,
        enforce_eager=True,
        trust_remote_code=True,
        tensor_parallel_size=min(torch.cuda.device_count(),4),
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop_token_ids=[128001,128008,128009]
    )

    for query in tqdm(queries, desc="Prepare model inputs"):
        # QA text
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)

        # single_img_b64 = prepare_single_image_input(model_name, query['image_path'])
        # vision_input = [fetch_image(f"data:image/jpeg;base64,{single_img_b64}")]
        vision_input = prepare_image_input(model_name, query)

        # placeholders = "<|image|>"*len(vision_input)
        placeholders = " ".join(["<|image|>"] * len(vision_input))
        messages = [{'role': 'user', 'content': f"{placeholders}\n{qa_text_prompt}"}]
        
        # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # input_prompt = tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True
        # )
        input_prompt = f"<|begin_of_text|>{placeholders}\n{qa_text_prompt}"
        item_input = {
            "prompt": input_prompt,
            "multi_modal_data": {
                "image": vision_input
            },
        }
        inputs.append(item_input)

    return inputs, llm, sampling_params


def prepare_llava_onevision_inputs(
    model_name,
    queries,
    prompt,
    temperature: float = 1,
    max_tokens: int = 1024
):
    """
    Prepare single-image QA for LLaVA-OneVision.
    """
    inputs = []
    for query in tqdm(queries, desc="Prepare model inputs"):
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
        
        # Only <image> placeholder
        vision_input = prepare_image_input(model_name, query)
        placeholders = " ".join(["<image>"] * len(vision_input))
        input_prompt = f"<|im_start|>user {placeholders}\n{qa_text_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # single_img_b64 = prepare_single_image_input(model_name, query['image_path'])
        # vision_input = [fetch_image(f"data:image/jpeg;base64,{single_img_b64}")]
        

        item_input = {
            "prompt": input_prompt,
            "multi_modal_data": {
                "image": vision_input
            },
        }
        inputs.append(item_input)
    
    llm = LLM(
        model=model_name,
        max_model_len=32768,
        limit_mm_per_prompt={"image": 2},
        tensor_parallel_size=min(torch.cuda.device_count(),4)
    )
    
    stop_token_ids = None
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids
    )

    return inputs, llm, sampling_params
