import base64
from utils.constant import COT_PROMPT
import requests
import os
import time
from tqdm import tqdm
import hashlib
import base64
import json
from vllm.multimodal.utils import fetch_image
from PIL import Image
from io import BytesIO

def prepare_context(paper_path,section_list):
    with open(paper_path,'r',encoding='utf-8')as f:
        paper = json.load(f)
    top_sections = []
    for item in section_list:
        if item.split('.')[0] not in top_sections:
            top_sections.append(item.split('.')[0])
    context = ""
    for sec in paper['sections']:
        if sec['section_id'].split('.')[0] in top_sections:
            context += sec['section_name'] + ':\n' + sec['text'] + '\n'
    return context

def prepare_caption(paper_path,item_type,item_id):
    with open(paper_path,'r',encoding='utf-8')as f:
        paper = json.load(f)
    if item_type=='chart':
        caption = paper['image_paths'][item_id]['caption']
    else:
        caption = paper['tables'][item_id]['capture']
    return caption

def prepare_qa_text_input(model_name, query, prompt):
    claim_type = query['claim_type']
    if claim_type=='direct' or claim_type=='analytical':
        user_prompt = prompt[claim_type]
        paper_path = query['paper_path']
        section_list = query['section']
        context = prepare_context(paper_path,section_list)
        caption = prepare_caption(paper_path,query['type'],query['item'])
        qa_text_prompt = user_prompt.substitute(claim=query['claim'],context=context,caption=caption)
        qa_text_message = {
            "type": "text",
            "text": qa_text_prompt
        }
    else:
        user_prompt = prompt[claim_type]
        paper_path = query['paper_path']
        section_list = query['section']
        context = prepare_context(paper_path,section_list)
        caption1 = prepare_caption(paper_path,query['item1_type'],query['item1'])
        caption2 = prepare_caption(paper_path,query['item2_type'],query['item2'])
        qa_text_prompt = user_prompt.substitute(claim=query['claim'],context=context,caption1=caption1,caption2=caption2)
        qa_text_message = {
            "type": "text",
            "text": qa_text_prompt
        }
    return qa_text_message, qa_text_prompt

def prepare_single_image_input(model_name, image_source):
    """
    Prepare a single image input for different model requirements.

    This function now accepts either:
      - A local image file path (e.g., "path/to/image.jpg")
      - An image URL (e.g., "http://.../some_image.jpg" or "https://...")
    
    It then converts the image content into base64 internally before returning
    the final structure, which varies depending on 'model_name':
      - a dict with {"type": "image", "source": {...}} for Claude
      - a raw base64 string for vLLM or other HF-based models
      - a dict with {"type": "image_url", ...} otherwise

    Args:
        model_name (str): The name of the model. Used to decide the returned format.
        image_source (str): Either a local file path or an image URL.

    Returns:
        A dictionary or a base64 string, depending on the model_name.
    """

    # 1. Determine if the source is a local path or a URL
    if image_source.startswith("http://") or image_source.startswith("https://"):
        # If it's an HTTP/HTTPS URL, download the image
        response = requests.get(image_source)
        response.raise_for_status()
        image_content = response.content
    else:
        # Otherwise, assume it's a local file path
        if not os.path.exists(image_source):
            raise FileNotFoundError(f"Image file not found: {image_source}")
        with open(image_source, "rb") as f:
            image_content = f.read()
    if model_name!='Qwen/Qwen2-VL-7B-Instruct' and model_name!='microsoft/Phi-4-multimodal-instruct':
        min_size = 224
        # 2. 用 PIL 打开图像并检查大小
        image = Image.open(BytesIO(image_content)).convert("RGB")
        if image.width < min_size or image.height < min_size:
            # 比例保持不变地放大到最小边 min_size
            ratio = max(min_size / image.width, min_size / image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.BICUBIC)
    else:
        min_size = 224
        image = Image.open(BytesIO(image_content)).convert("RGB")
        width, height = image.size
        # 缩放比例，使得最小边变为 224
        scale = min_size / min(width, height)

        # 计算新尺寸
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Resize
        image = image.resize((new_width, new_height), Image.BICUBIC)

    # 3. 转回字节内容
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    image_content = buffer.getvalue()

    # 2. Convert the image content to base64
    image_base64 = base64.b64encode(image_content).decode("utf-8")
    # 3. Return the structure based on the model_name
    if "claude" in model_name:
        # Claude usually requires {"type":"image", "source":{...}}
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_base64,
            },
        }
    elif "/" in model_name:
        # For vLLM or other HF-based models, return raw base64
        return image_base64
    else:
        # Some fallback or custom format
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}",
            },
        }

def prepare_image_input(model_name, query):
    if query['claim_type']=='direct' or query['claim_type']=='analytical':
        single_img_b64 = prepare_single_image_input(model_name, query['image_path'])
        vision_input = [fetch_image(f"data:image/jpeg;base64,{single_img_b64}")]
    else:
        img1 = prepare_single_image_input(model_name, query['item1_path'])
        img2 = prepare_single_image_input(model_name, query['item2_path'])
        vision_input = [fetch_image(f"data:image/jpeg;base64,{img1}"),fetch_image(f"data:image/jpeg;base64,{img2}")]
    return vision_input

def prepare_qa_inputs(model_name, queries, prompt=COT_PROMPT):
    """
    Prepare the final list of messages that combine single image input with QA text.
    - Each query is turned into a list containing a user role with two items: image + text QA.
    """
    messages = []
    for query in tqdm(queries):
        # Create the QA text portion
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)

        # Create or retrieve the single image data structure
        if query['claim_type']=='direct' or query['claim_type']=='analytical':
            single_image_data = prepare_single_image_input(model_name, query['image_path'])

            # Combine them into a "user" content list
            prompt_message = [
                {
                    "role": "user",
                    "content": [
                        single_image_data,
                        qa_text_message
                    ],
                }
            ]
        else:
            single_image1 = prepare_single_image_input(model_name, query['item1_path'])
            single_image2 = prepare_single_image_input(model_name, query['item2_path'])
            # Combine them into a "user" content list
            prompt_message = [
                {
                    "role": "user",
                    "content": [
                        single_image1,
                        single_image2,
                        qa_text_message
                    ],
                }
            ]
        messages.append(prompt_message)
    return messages
