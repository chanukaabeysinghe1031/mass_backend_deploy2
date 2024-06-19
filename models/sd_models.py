from typing import Optional
import requests
from datetime import datetime
from fastapi import HTTPException
from utils.db import uploadImage
import random
import base64
import os

STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
STABILITY_ENGINE_ID = "stable-diffusion-v1-6"  # change this by constraint model
STABILITY_API_HOST = 'https://api.stability.ai'

# MODEL_CONSTRAINTS = {
#     "stability-ultra": {"max_resolution": (1024, 1024)},
#     "stability-core": {"max_resolution": (1280, 960)},  # Example resolution for 1.5MP
#     "stability-diffusion": {"max_resolution": (1024, 1024)},
#     "inpaint": {"max_resolution": (3072, 3072), "max_pixels": 9437184},  # 4MP limit
#     "sdxl": {"height_width_enums": [(1024, 1024), (1152, 896), (896, 1152), (1216, 832), (1344, 768), (768, 1344),
#                                     (1536, 640), (640, 1536)]},
#     "sd1-6": {"max_resolution": (1024, 1024)},  # Replace with actual max resolution if different
#     "sd1-sdai": {"max_resolution": (1024, 1024)},  # Replace with actual max resolution if different
# }

MODEL_CONSTRAINTS = {
    "stability-ultra": {"max_resolution": 1,
                        "aspect_rations": ["16:9", "1:1", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"]},
    "stability-core": {
        "max_resolution": 1.5,
        "aspect_rations": ["16:9", "1:1", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"],
        "style_preset": ["3d-model", "analog-film", "anime", "cinematic", "comic-book", "digital-art", "enhance",
                         "fantasy-art", "isometric", "line-art", "low-poly", "modeling-compound", "neon-punk",
                         "origami", "photographic", "pixel-art", "tile-texture"]
    },
    "stability-diffusion": {"max_resolution": 1,
                            "aspect_rations": ["16:9", "1:1", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"],
                            "model": ["sd3-large", "sd3-large-turbo", "sd3-medium"]},
    "inpaint": {"max_resolution": 4},
    "sdxl": {"height_width_enums": [(1024, 1024), (1152, 896), (896, 1152), (1216, 832), (1344, 768), (768, 1344),
                                    (1536, 640), (640, 1536)], "model": "stable-diffusion-v1-6"},
    # Replace with actual max resolution if different
    "sd1-sdai": {"min_dimension": 320, "max_dimension": 1536, "model": "stable-diffusion-xl-1024-v1-0"},
}


def check_constraints(model_id, width, height, aspect_ratio, style_preset=None):
    constraints = MODEL_CONSTRAINTS.get(model_id)
    if not constraints:
        raise ValueError(f"Invalid model ID: {model_id}")

    if model_id == "stability-ultra":
        # check if the aspect ratio is valid
        if aspect_ratio not in constraints["aspect_rations"]:
            raise ValueError(f"Invalid aspect ratio for {model_id}.")
        # return the aspect ratio
        return aspect_ratio

    if model_id == "stability-core":
        # check if the aspect ratio is valid
        if aspect_ratio not in constraints["aspect_rations"]:
            raise ValueError(f"Invalid aspect ratio for {model_id}.")
        if style_preset and style_preset not in constraints["style_preset"]:
            raise ValueError(f"Invalid style preset for {model_id}.")
        # # check if the style preset is valid
        # if style_preset not in constraints["style_preset"]:
        #     raise ValueError(f"Invalid style preset for {model_id}.")
        # return the aspect ratio
        return aspect_ratio, style_preset

    if model_id == "stability-diffusion":
        # check if the aspect ratio is valid
        if aspect_ratio not in constraints["aspect_rations"]:
            raise ValueError(f"Invalid aspect ratio for {model_id}.")
        return aspect_ratio, constraints["model"][0]

    if model_id == "inpaint":
        # check if the resolution is valid
        if width * height > constraints["max_resolution"]:
            raise ValueError(f"Requested resolution exceeds the maximum pixel count for {model_id}.")
        # return the width and height
        return width, height

    if model_id == "sdxl":
        # check if the resolution is valid
        if (width, height) not in constraints["height_width_enums"]:
            raise ValueError(f"Invalid resolution for {model_id}.")
        # return the resolution
        return width, height, constraints["model"]

    if model_id == "sd1-sdai":
        # check if the resolution is valid
        if width < constraints["min_dimension"] or height > constraints["max_dimension"]:
            raise ValueError(f"Invalid resolution for {model_id}.")
        # return the resolution
        return width, height, constraints["model"]

    return width, height


# def check_constraints(model_id, width=None, height=None, aspect_ratio=None, style_preset=None):
#     constraints = MODEL_CONSTRAINTS.get(model_id)
#     if not constraints:
#         raise ValueError(f"Invalid model ID: {model_id}")
#
#     # Check stability-ultra constraints
#     if model_id == "stability-ultra":
#         if aspect_ratio not in constraints["aspect_rations"]:
#             raise ValueError(f"Invalid aspect ratio for {model_id}.")
#         return aspect_ratio
#
#     # Check stability-core constraints
#     if model_id == "stability-core":
#         if aspect_ratio not in constraints["aspect_rations"]:
#             raise ValueError(f"Invalid aspect ratio for {model_id}.")
#         if style_preset and style_preset not in constraints["style_preset"]:
#             raise ValueError(f"Invalid style preset for {model_id}.")
#         return aspect_ratio, style_preset
#
#     # Check stability-diffusion constraints
#     if model_id == "stability-diffusion":
#         if aspect_ratio not in constraints["aspect_rations"]:
#             raise ValueError(f"Invalid aspect ratio for {model_id}.")
#         if style_preset and style_preset not in constraints["model"]:
#             raise ValueError(f"Invalid model for {model_id}.")
#         return aspect_ratio, style_preset
#
#     # Check inpaint constraints
#     if model_id == "inpaint":
#         if width * height > constraints["max_resolution"] * 1000000:
#             raise ValueError(f"Requested resolution exceeds the maximum pixel count for {model_id}.")
#         return width, height
#
#     # Check sdxl constraints
#     if model_id == "sdxl":
#         if (width, height) not in constraints["height_width_enums"]:
#             raise ValueError(f"Invalid resolution for {model_id}.")
#         return width, height, constraints["model"]
#
#     # Check sd1-sdai constraints
#     if model_id == "sd1-sdai":
#         if width < constraints["min_dimension"] or height > constraints["max_dimension"]:
#             raise ValueError(f"Invalid resolution for {model_id}.")
#         return width, height, constraints["model"]
#
#     return width, height


# def run_test_cases():
#     # Test Case 1: stability-ultra with valid aspect ratio
#     try:
#         result = check_constraints("stability-ultra", aspect_ratio="16:9")
#         print("Test Case 1 Passed:", result)
#     except ValueError as e:
#         print("Test Case 1 Failed:", e)
#
#     # Test Case 2: stability-core with valid aspect ratio and style preset
#     try:
#         result = check_constraints("stability-core", aspect_ratio="1:1", style_preset="cinematic")
#         print("Test Case 2 Passed:", result)
#     except ValueError as e:
#         print("Test Case 2 Failed:", e)
#
#     # Test Case 3: stability-core with valid aspect ratio and no style preset
#     try:
#         result = check_constraints("stability-core", aspect_ratio="1:1")
#         print("Test Case 3 Passed:", result)
#     except ValueError as e:
#         print("Test Case 3 Failed:", e)
#
#     # Test Case 4: inpaint with valid width and height
#     try:
#         result = check_constraints("inpaint", width=800, height=600)
#         print("Test Case 4 Passed:", result)
#     except ValueError as e:
#         print("Test Case 4 Failed:", e)
#
#     # Test Case 5: stability-ultra with invalid aspect ratio
#     try:
#         result = check_constraints("stability-ultra", aspect_ratio="10:10")
#         print("Test Case 5 Failed: Expected failure but got", result)
#     except ValueError as e:
#         print("Test Case 5 Passed:", e)
#
#     # Test Case 6: stability-core with invalid style preset
#     try:
#         result = check_constraints("stability-core", aspect_ratio="1:1", style_preset="invalid-style")
#         print("Test Case 6 Failed: Expected failure but got", result)
#     except ValueError as e:
#         print("Test Case 6 Passed:", e)
#
#     # Test Case 7: inpaint with resolution exceeding max limit
#     try:
#         result = check_constraints("inpaint", width=5000, height=5000)
#         print("Test Case 7 Failed: Expected failure but got", result)
#     except ValueError as e:
#         print("Test Case 7 Passed:", e)
#
#     # Test Case 8: sdxl with valid width and height
#     try:
#         result = check_constraints("sdxl", width=1024, height=1024)
#         print("Test Case 8 Passed:", result)
#     except ValueError as e:
#         print("Test Case 8 Failed:", e)
#
#     # Test Case 9: sdxl with invalid width and height
#     try:
#         result = check_constraints("sdxl", width=1000, height=1000)
#         print("Test Case 9 Failed: Expected failure but got", result)
#     except ValueError as e:
#         print("Test Case 9 Passed:", e)
#
#     # Test Case 10: sd1-sdai with valid dimensions
#     try:
#         result = check_constraints("sd1-sdai", width=400, height=400)
#         print("Test Case 10 Passed:", result)
#     except ValueError as e:
#         print("Test Case 10 Failed:", e)
#
#     # Test Case 11: sd1-sdai with invalid dimensions
#     try:
#         result = check_constraints("sd1-sdai", width=200, height=400)
#         print("Test Case 11 Failed: Expected failure but got", result)
#     except ValueError as e:
#         print("Test Case 11 Passed:", e)
#
# run_test_cases()


# def validate_and_adjust_dimensions(model_id, width=None, height=None, aspect_ratio=None):
#     constraints = MODEL_CONSTRAINTS.get(model_id)
#     if not constraints:
#         raise ValueError(f"Invalid model ID: {model_id}")
#
#     max_width, max_height = constraints.get("max_resolution")
#     max_pixels = constraints.get("max_pixels", max_width * max_height)
#
#     if width and height:
#         if width * height > max_pixels:
#             raise ValueError(f"Requested resolution exceeds the maximum pixel count for {model_id}.")
#         if width > max_width or height > max_height:
#             raise ValueError(f"Requested dimensions exceed the maximum resolution for {model_id}.")
#         return width, height
#
#     if aspect_ratio:
#         ar_width, ar_height = map(int, aspect_ratio.split(':'))
#         if ar_width > ar_height:
#             width = max_width
#             height = int(max_width * ar_height / ar_width)
#         else:
#             height = max_height
#             width = int(max_height * ar_width / ar_height)
#
#         if width * height > max_pixels:
#             raise ValueError(f"Calculated resolution exceeds the maximum pixel count for {model_id}.")
#         return width, height
#
#     return max_width, max_height


def remove_images(file_paths: list):
    for file_path in file_paths:
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Error removing {file_path}: {e.strerror}")


async def process_stability_ultra(user_id: str, model_id: str, input_data: dict, height: int, width: int,
                                  aspect_ratio: str):
    url = "https://api.stability.ai/v2beta/stable-image/generate/ultra"
    headers = {
        "authorization": f"Bearer {STABILITY_API_KEY}",
        "accept": "image/*"
    }

    aspect_ratio = check_constraints(model_id, width, height, aspect_ratio)
    print("Stability Model", height)
    print("Stability Model", width)

    input_data["aspect_ratio"] = aspect_ratio

    response = requests.post(url, headers=headers, files={"none": ''}, data=input_data)
    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))
    random_number = random.randint(10, 99999999)
    current_date = datetime.now().strftime("%Y-%m-%d")
    file_path = f"stability_ultra_{random_number}_{current_date}.png"
    local_files = [file_path]
    image_urls = []

    with open(file_path, "wb") as f:
        f.write(response.content)
        uploaded_url = await uploadImage(file_path, user_id, model_id)
        image_urls.append({"url": uploaded_url, "finish_reason": "SUCCESS"})

    remove_images(local_files)

    return image_urls


async def process_stability_core(user_id: str, model_id: str, input_data: dict, height: int, width: int,
                                 aspect_ratio: str, style_preset: str = None):
    url = "https://api.stability.ai/v2beta/stable-image/generate/core"
    headers = {
        "authorization": f"Bearer {STABILITY_API_KEY}",
        "accept": "image/*"
    }

    aspect_ratio, style_preset = check_constraints(model_id, width, height, aspect_ratio, style_preset)

    input_data["aspect_ratio"] = aspect_ratio
    input_data["style_preset"] = style_preset

    print("Stability Core Model", model_id)
    print(input_data)

    response = requests.post(url, headers=headers, files={"none": ''}, data=input_data)
    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))
    random_number = random.randint(10, 99999999)
    current_date = datetime.now().strftime("%Y-%m-%d")
    file_path = f"stability_core_{random_number}_{current_date}.png"
    local_files = [file_path]
    image_urls = []

    with open(file_path, "wb") as f:
        f.write(response.content)
        uploaded_url = await uploadImage(file_path, user_id, model_id)
        image_urls.append({"url": uploaded_url, "finish_reason": "SUCCESS"})

    remove_images(local_files)

    return image_urls


async def process_stability_diffusion(user_id: str, model_id: str, input_data: dict, height: int, width: int,
                                      aspect_ratio: str):
    url = "https://api.stability.ai/v2beta/stable-image/generate/sd3"
    headers = {
        "authorization": f"Bearer {STABILITY_API_KEY}",
        "accept": "image/*"
    }

    aspect_ratio, model = check_constraints(model_id, width, height, aspect_ratio)

    input_data["aspect_ratio"] = aspect_ratio
    input_data["model"] = model

    response = requests.post(url, headers=headers, files={"none": ''}, data=input_data)
    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))
    random_number = random.randint(10, 99999999)
    current_date = datetime.now().strftime("%Y-%m-%d")
    file_path = f"stability_diffusion_{random_number}_{current_date}.png"
    local_files = [file_path]
    image_urls = []

    with open(file_path, "wb") as f:
        f.write(response.content)
        uploaded_url = await uploadImage(file_path, user_id, model_id)
        image_urls.append({"url": uploaded_url, "finish_reason": "SUCCESS"})

    remove_images(local_files)

    return image_urls


async def process_sd_text_to_image(user_id: str, model_id: str, input_data: dict, height: int, width: int,
                                   aspect_ratio: str):
    api_host = STABILITY_API_HOST
    api_key = STABILITY_API_KEY
    if api_key is None:
        raise Exception("Missing Stability API key.")

    width, height, engine_id = check_constraints(model_id, width, height, aspect_ratio)

    prompt_text = input_data["prompt"]
    print("Stability Model", prompt_text)

    structured_input = {
        "text_prompts": [
            {
                "text": prompt_text
            }
        ],
        "height": height,
        "width": width
    }

    response = requests.post(
        f"{api_host}/v1/generation/{engine_id}/text-to-image",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        json=structured_input,
    )
    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))
    data = response.json()
    random_number = random.randint(10, 99999999)
    current_date = datetime.now().strftime("%Y-%m-%d")
    local_files = []
    image_urls = []

    for i, image in enumerate(data["artifacts"]):
        file_path = f"v1_txt2img_{random_number}_{current_date}_{i}.png"
        local_files.append(file_path)
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(image["base64"]))
            uploaded_url = await uploadImage(file_path, user_id, model_id)
            image_urls.append({"url": uploaded_url, "finish_reason": "SUCCESS"})

    remove_images(local_files)

    return image_urls


async def process_sd_image_to_image(user_id: str, model_id: str, image_url: str, prompt: str, negative_prompt: str,
                                    image_strength: float, cfg_scale: float, samples: int, steps: int,
                                    init_image_mode: str):
    engine_id = STABILITY_ENGINE_ID
    api_host = STABILITY_API_HOST
    api_key = STABILITY_API_KEY
    if api_key is None:
        raise Exception("Missing Stability API key.")

    download_image_response = requests.get(image_url)
    if download_image_response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download image from the provided URL")

    response = requests.post(
        f"{api_host}/v1/generation/{engine_id}/image-to-image",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        files={
            "init_image": ("image.jpg", download_image_response.content)
        },
        data={
            "text_prompts[0][text]": prompt,
            "cfg_scale": int(cfg_scale),
            "samples": int(samples),
            "steps": int(steps),
        }
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()
    random_number = random.randint(10, 99999999)
    current_date = datetime.now().strftime("%Y-%m-%d")
    local_files = []
    image_urls = []

    for i, image in enumerate(data["artifacts"]):
        file_path = f"v1_txt2img_{random_number}_{current_date}_{i}.png"
        local_files.append(file_path)
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(image["base64"]))
            uploaded_url = await uploadImage(file_path, user_id, model_id)
            image_urls.append({"url": uploaded_url, "finish_reason": "SUCCESS"})

    remove_images(local_files)

    return image_urls


async def process_sd_inpaint_and_outpaint(user_id: str, model_id: str, file1: str, file2: str, prompt: str,
                                          sharpness: str, cn_type1: str, base_model_name: str, style_selections: str,
                                          performance_selection: str, image_number: str, negative_prompt: str,
                                          image_strength: str, cfg_scale: str, samples: str, steps: str,
                                          init_image_mode: str, clip_guidance_preset: str, mask_source: str):
    engine_id = STABILITY_ENGINE_ID
    api_host = STABILITY_API_HOST
    api_key = STABILITY_API_KEY
    if api_key is None:
        raise Exception("Missing Stability API key.")

    download_image_response1 = requests.get(file1)
    download_image_response2 = requests.get(file2)

    if download_image_response1.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download image from the provided URL")
    if download_image_response2.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download image from the provided URL")

    response = requests.post(
        f"{api_host}/v1/generation/{engine_id}/image-to-image/masking",
        headers={
            "Accept": 'application/json',
            "Authorization": f"Bearer {api_key}"
        },
        files={
            'init_image': ("image1.jpg", download_image_response1.content),
            'mask_image': ("image2.jpg", download_image_response2.content)
        },
        data={
            "mask_source": "MASK_IMAGE_WHITE",
            "text_prompts[0][text]": prompt,
            "clip_guidance_preset": clip_guidance_preset,
            # "cfg_scale": int(cfg_scale),
            # "samples": int(samples),
            # "steps": int(steps),
        }
    )
    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()
    random_number = random.randint(10, 99999999)
    current_date = datetime.now().strftime("%Y-%m-%d")
    local_files = []
    image_urls = []

    for i, image in enumerate(data["artifacts"]):
        file_path = f"v1_txt2img_{random_number}_{current_date}_{i}.png"
        local_files.append(file_path)
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(image["base64"]))
            uploaded_url = await uploadImage(file_path, user_id, model_id)
            image_urls.append({"url": uploaded_url, "finish_reason": "SUCCESS"})

    remove_images(local_files)

    return image_urls


async def process_stability_inpaint(user_id: str, image: str, prompt: str, mask: Optional[str], negative_prompt: str,
                                    seed: int, output_format: str):
    url = "https://api.stability.ai/v2beta/stable-image/edit/inpaint"
    headers = {
        "authorization": f"Bearer {STABILITY_API_KEY}",
        "accept": "image/*"
    }
    files = {
        "image": open(image, "rb")
    }
    if mask:
        files["mask"] = open(mask, "rb")

    data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "output_format": output_format,
    }

    response = requests.post(url, headers=headers, files=files, data=data)
    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    random_number = random.randint(10, 99999999)
    current_date = datetime.now().strftime("%Y-%m-%d")
    file_path = f"stability_inpaint_{random_number}_{current_date}.png"
    local_files = [file_path]
    image_urls = []

    with open(file_path, "wb") as f:
        f.write(response.content)
        uploaded_url = await uploadImage(file_path, user_id, "stability-inpaint")
        image_urls.append({"url": uploaded_url, "finish_reason": "SUCCESS"})

    remove_images(local_files)

    return image_urls
