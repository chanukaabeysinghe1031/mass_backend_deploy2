import requests
from datetime import datetime
from fastapi import HTTPException
from utils.db import uploadImage
import random
import base64
import os
from pathlib import Path

async def process_fooocus_text_to_image(user_id: str, model_id: str, input_data: dict):
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }
    response = requests.post(
        "http://127.0.0.1:8888/v1/generation/text-to-image",
        headers=headers,
        json=input_data
    )
    if response.status_code == 200:
        response_json = response.json()
        image_url = response_json.get("url")
        uploaded_url = await uploadImage(image_url, user_id, model_id)
        return [{"url": uploaded_url, "finish_reason": "SUCCESS"}]
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)

async def process_fooocus_image_to_image(user_id: str, model_id: str, image_url: str, prompt: str, negative_prompt: str, image_strength: float, cfg_scale: float, samples: int, steps: int, init_image_mode: str):
    headers = {
        "accept": "application/json",
    }
    download_image_response = requests.get(image_url)
    if download_image_response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download image from the provided URL")

    files = {
        'input_image': ("image.jpg", download_image_response.content),
        'prompt': (None, prompt),
        'negative_prompt': (None, negative_prompt),
        'image_strength': (None, str(image_strength)),
        'cfg_scale': (None, str(cfg_scale)),
        'samples': (None, str(samples)),
        'steps': (None, str(steps)),
        'init_image_mode': (None, init_image_mode)
    }
    response = requests.post(
        "http://127.0.0.1:8888/v1/generation/image-to-image", headers=headers, files=files)

    if response.status_code == 200:
        response_json = response.json()
        image_url = response_json.get("url")
        uploaded_url = await uploadImage(image_url, user_id, model_id)
        return [{"url": uploaded_url, "finish_reason": "SUCCESS"}]
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)

async def process_fooocus_inpaint_and_outpaint(user_id: str, model_id: str, file1: str, file2: str, prompt: str, sharpness: str, cn_type1: str, base_model_name: str, style_selections: str, performance_selection: str, image_number: str, negative_prompt: str, image_strength: str, cfg_scale: str, samples: str, steps: str, init_image_mode: str, clip_guidance_preset: str, mask_source: str):
    headers = {
        "accept": "application/json",
    }
    download_image_response1 = requests.get(file1)
    download_image_response2 = requests.get(file2)

    if download_image_response1.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download image from the provided URL")
    if download_image_response2.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download image from the provided URL")

    files = {
        'sharpness': (None, sharpness),
        'input_mask': ("image1.jpg", download_image_response2.content),
        'outpaint_distance_right': (None, '0'),
        'loras': (None, '[{"model_name":"sd_xl_offset_example-lora_1.0.safetensors","weight":0.1}]'),
        'outpaint_distance_left': (None, '0'),
        'advanced_params': (None, ''),
        'guidance_scale': (None, '4'),
        'prompt': (None, prompt),
        'input_image': ("image2.jpg", download_image_response1.content),
        'outpaint_distance_bottom': (None, '0'),
        'require_base64': (None, 'false'),
        'async_process': (None, 'false'),
        'image_number': (None, image_number),
        'negative_prompt': (None, negative_prompt),
        'refiner_switch': (None, '0.5'),
        'base_model_name': (None, base_model_name),
        'image_seed': (None, '-1'),
        'style_selections': (None, style_selections),
        'inpaint_additional_prompt': (None, ''),
        'outpaint_selections': (None, ''),
        'outpaint_distance_top': (None, '0'),
        'refiner_model_name': (None, 'None'),
        'cn_stop1': (None, ''),
        'aspect_ratios_selection': (None, '1152*896'),
        'performance_selection': (None, performance_selection)
    }

    response = requests.post(
        "http://127.0.0.1:8888/v1/generation/image-inpaint-outpaint", headers=headers, files=files)

    if response.status_code == 200:
        response_json = response.json()
        image_url = response_json.get("url")
        uploaded_url = await uploadImage(image_url, user_id, model_id)
        return [{"url": uploaded_url, "finish_reason": "SUCCESS"}]
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)
