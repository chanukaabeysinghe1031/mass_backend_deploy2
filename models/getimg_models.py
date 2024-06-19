import requests
from datetime import datetime
from fastapi import HTTPException
from utils.db import uploadImage
import random
import base64
import os

GETIMG_API_KEY = os.getenv("GETIMG_API_KEY")

def remove_images(file_paths: list):
    for file_path in file_paths:
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Error removing {file_path}: {e.strerror}")

async def process_getimg_text_to_image(user_id: str, model_id: str, input_data: dict):
    url = "https://api.getimg.ai/v1/stable-diffusion/text-to-image"
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {GETIMG_API_KEY}"
    }
    response = requests.post(url, json=input_data, headers=headers)
    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))
    data = response.json()
    random_number = random.randint(10, 99999999)
    current_date = datetime.now().strftime("%Y-%m-%d")
    file_path = f"v1_txt2img_{random_number}_{current_date}.png"
    local_files = [file_path]
    image_urls = []

    with open(file_path, "wb") as f:
        f.write(base64.b64decode(data["image"]))
        uploaded_url = await uploadImage(file_path, user_id, model_id)
        image_urls.append({"url": uploaded_url, "finish_reason": "SUCCESS"})

    remove_images(local_files)

    return image_urls

async def process_getimg_image_to_image(user_id: str, model_id: str, image_url: str, prompt: str, negative_prompt: str, image_strength: float, cfg_scale: float, samples: int, steps: int, init_image_mode: str):
    url = "https://api.getimg.ai/v1/stable-diffusion/image-to-image"
    download_image_response = requests.get(image_url)
    if download_image_response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download image from the provided URL")

    base64_encoded_str = base64.b64encode(download_image_response.content).decode("utf-8")
    payload = {
        "model": "stable-diffusion-v1-5",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": base64_encoded_str,
        "strength": image_strength,
        "steps": steps,
        "guidance": cfg_scale,
        "seed": 0,
        "scheduler": "dpmsolver++",
        "output_format": "jpeg"
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {GETIMG_API_KEY}"
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))
    data = response.json()
    random_number = random.randint(10, 99999999)
    current_date = datetime.now().strftime("%Y-%m-%d")
    file_path = f"v1_txt2img_{random_number}_{current_date}.png"
    local_files = [file_path]
    image_urls = []

    with open(file_path, "wb") as f:
        f.write(base64.b64decode(data["image"]))
        uploaded_url = await uploadImage(file_path, user_id, model_id)
        image_urls.append({"url": uploaded_url, "finish_reason": "SUCCESS"})

    remove_images(local_files)

    return image_urls

async def process_getimg_inpaint_and_outpaint(user_id: str, model_id: str, file1: str, file2: str, prompt: str, sharpness: str, cn_type1: str, base_model_name: str, style_selections: str, performance_selection: str, image_number: str, negative_prompt: str, image_strength: str, cfg_scale: str, samples: str, steps: str, init_image_mode: str, clip_guidance_preset: str, mask_source: str):
    url = "https://api.getimg.ai/v1/stable-diffusion/inpaint"
    download_image_response1 = requests.get(file1)
    download_image_response2 = requests.get(file2)

    if download_image_response1.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download image from the provided URL")
    if download_image_response2.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download image from the provided URL")

    base64_encoded_str1 = base64.b64encode(download_image_response1.content).decode("utf-8")
    base64_encoded_str2 = base64.b64encode(download_image_response2.content).decode("utf-8")

    payload = {
        "model": "stable-diffusion-v1-5-inpainting",
        "prompt": prompt,
        "negative_prompt": "Disfigured, cartoon, blurry",
        "image": base64_encoded_str1,
        "mask_image": base64_encoded_str2,
        "strength": 1,
        "width": 512,
        "height": 512,
        "steps": 25,
        "guidance": 7.5,
        "seed": 0,
        "scheduler": "ddim",
        "output_format": "jpeg"
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {GETIMG_API_KEY}"
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))
    data = response.json()
    random_number = random.randint(10, 99999999)
    current_date = datetime.now().strftime("%Y-%m-%d")
    file_path = f"v1_txt2img_{random_number}_{current_date}.png"
    local_files = [file_path]
    image_urls = []

    with open(file_path, "wb") as f:
        f.write(base64.b64decode(data["image"]))
        uploaded_url = await uploadImage(file_path, user_id, model_id)
        image_urls.append({"url": uploaded_url, "finish_reason": "SUCCESS"})

    remove_images(local_files)

    return image_urls
