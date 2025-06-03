from utils.vllm_input_preparation import *
import json
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT

model_example_map = {
    "Qwen/Qwen2-VL-7B-Instruct": prepare_qwen2_inputs, 
    "Qwen/Qwen2-VL-72B-Instruct": prepare_qwen2_inputs, 
    "microsoft/Phi-3.5-vision-instruct": prepare_phi3v_inputs, 
    "microsoft/Phi-4-multimodal-instruct":prepare_phi3v_inputs, 
    "OpenGVLab/InternVL2-8B": prepare_general_vlm_inputs,
    "OpenGVLab/InternVL2_5-8B": prepare_general_vlm_inputs,
    "OpenGVLab/InternVL2_5-38B":prepare_general_vlm_inputs,
    "mistral-community/pixtral-12b": prepare_pixtral_inputs,
    "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf": prepare_llava_onevision_inputs,
    "unsloth/Llama-3.2-11B-Vision-Instruct": prepare_mllama_inputs,
    "Qwen/Qwen2.5-VL-7B-Instruct": prepare_qwen2_inputs,
    "OpenGVLab/InternVL3-8B":prepare_general_vlm_inputs,
    "OpenGVLab/InternVL3-38B":prepare_general_vlm_inputs,
    "Qwen/Qwen2.5-VL-72B-Instruct": prepare_qwen2_inputs, 
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": prepare_pixtral_inputs,
}

vllm_model_list = list(model_example_map.keys())
json.dump(vllm_model_list, open("model_inference/vllm_model_list.json", "w"), indent=4, ensure_ascii=False)

def generate_vlm_response(model_name, 
                       queries, 
                       prompt=COT_PROMPT, 
                       temperature: float=1, 
                       max_tokens: int=1024):
    inputs, llm, sampling_params = model_example_map[model_name](model_name, queries, prompt, temperature, max_tokens)
    
    responses = llm.generate(inputs, sampling_params=sampling_params)
    responses = [response.outputs[0].text for response in responses]
    return responses

def generate_response(model_name: str, 
                    prompt: str,
                    queries: list,
                    output_path: str,
                    n: int=1,
                    temperature: float=GENERATION_TEMPERATURE,
                    max_tokens: int=MAX_TOKENS):
    if model_name not in model_example_map:
        raise ValueError(f"Model type {model_name} is not supported.")
    responses = generate_vlm_response(model_name, 
                                      queries, 
                                      prompt=prompt, 
                                      temperature=temperature, 
                                      max_tokens=max_tokens)
    for query, response in zip(queries, responses):
        query["response"] = response
    
    json.dump(queries, open(output_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)