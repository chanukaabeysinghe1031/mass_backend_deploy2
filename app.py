from typing import Optional, List
from fastapi import FastAPI, HTTPException, Request
import requests
import base64
import os
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import random
from datetime import datetime
from io import BytesIO
import numpy as np
from PIL import Image
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from keras.src.applications.imagenet_utils import preprocess_input
from keras.src.applications.inception_v3 import InceptionV3
from pydantic import BaseModel

from utils.db import uploadImage
# from utils.fid_helpers import load_images_from_folder, scale_images, calculate_fid
# from utils.clip_helpers import calculate_clip_score

load_dotenv()

app = FastAPI()

origins = [
    "http://localhost:4000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FOOCUS_URL = "http://127.0.0.1:8888/v1/"
GETIMG_API_KEY = os.getenv("GETIMG_API_KEY")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
STABILITY_ENGINE_ID = "stable-diffusion-v1-6"
STABILITY_API_HOST = 'https://api.stability.ai'


def remove_images(file_paths: List[str]):
    for file_path in file_paths:
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Error removing {file_path}: {e.strerror}")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/images/{image_url}")
async def get_image(image_url: str):
    image_path = Path(image_url)
    if not image_path.is_file():
        return {"error": "Image not found on the server"}
    return FileResponse(image_path)


class TextToImageRequest(BaseModel):
    user_id: str
    model_id: str
    input: dict


class TextToImageResponse(BaseModel):
    url: str
    finish_reason: str


@app.post("/textToImage", response_model=List[TextToImageResponse])
async def text_to_image(request: TextToImageRequest):
    try:
        print(datetime.now())
        user_id = request.user_id
        model_id = request.model_id
        input_data = request.input

        image_urls = []

        if model_id == "fooocus":
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
                image_urls.append({"url": uploaded_url, "finish_reason": "SUCCESS"})
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)

        elif model_id == "sd1-sdai":
            engine_id = STABILITY_ENGINE_ID
            api_host = STABILITY_API_HOST
            api_key = STABILITY_API_KEY
            if api_key is None:
                raise Exception("Missing Stability API key.")

            prompt_text = input_data["prompt"]
            print("Stability Model", prompt_text)

            structured_input = {
                "text_prompts": [
                    {
                        "text": prompt_text
                    }
                ],
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
            for i, image in enumerate(data["artifacts"]):
                file_path = f"v1_txt2img_{random_number}_{current_date}_{i}.png"
                local_files.append(file_path)
                with open(file_path, "wb") as f:
                    f.write(base64.b64decode(image["base64"]))
                    uploaded_url = await uploadImage(file_path, user_id, model_id)
                    image_urls.append({"url": uploaded_url, "finish_reason": "SUCCESS"})

            remove_images(local_files)

        elif model_id == "sd-getai":
            print("GetImg Model", input_data["prompt"])
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
            with open(file_path, "wb") as f:
                f.write(base64.b64decode(data["image"]))
                uploaded_url = await uploadImage(file_path, user_id, model_id)
                image_urls.append({"url": uploaded_url, "finish_reason": "SUCCESS"})

            remove_images(local_files)
            print(datetime.now())
            print("Image URL", image_urls)

        return image_urls

    except Exception as e:
        print(f"Exception: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


class TextAndImageToImageRequest(BaseModel):
    user_id: str
    model_id: str
    image_url: str
    prompt: str
    negative_prompt: Optional[str] = ""
    image_strength: Optional[float] = 0.5
    cfg_scale: Optional[float] = 7.5
    samples: Optional[int] = 1
    steps: Optional[int] = 50
    init_image_mode: Optional[str] = 'image'


class TextAndImageToImageResponse(BaseModel):
    url: str
    finish_reason: str


@app.post("/textAndImageToImage", response_model=List[TextAndImageToImageResponse])
async def text_and_image_to_image(request: TextAndImageToImageRequest):
    try:
        print(datetime.now())
        user_id = request.user_id
        model_id = request.model_id
        image_url = request.image_url
        prompt = request.prompt
        negative_prompt = request.negative_prompt
        image_strength = request.image_strength
        cfg_scale = request.cfg_scale
        samples = request.samples
        steps = request.steps
        init_image_mode = request.init_image_mode

        # Download the initial image
        download_image_response = requests.get(image_url)
        image_urls = []

        if download_image_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download image from the provided URL")

        local_files = []

        if model_id == "fooocus":
            headers = {
                "accept": "application/json",
            }
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
                image_urls.append({"url": uploaded_url, "finish_reason": "SUCCESS"})
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)

        elif model_id == "sd1-sdai":
            engine_id = STABILITY_ENGINE_ID
            api_host = STABILITY_API_HOST
            api_key = STABILITY_API_KEY
            if api_key is None:
                raise Exception("Missing Stability API key.")

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
                    "image_strength": float(image_strength),
                    "init_image_mode": init_image_mode,
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

            for i, image in enumerate(data["artifacts"]):
                file_path = f"v1_txt2img_{random_number}_{current_date}_{i}.png"
                local_files.append(file_path)
                with open(file_path, "wb") as f:
                    f.write(base64.b64decode(image["base64"]))
                    uploaded_url = await uploadImage(file_path, user_id, model_id)
                    image_urls.append({"url": uploaded_url, "finish_reason": "SUCCESS"})

            remove_images(local_files)

        elif model_id == "sd-getai":
            url = "https://api.getimg.ai/v1/stable-diffusion/image-to-image"
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
            with open(file_path, "wb") as f:
                f.write(base64.b64decode(data["image"]))
                uploaded_url = await uploadImage(file_path, user_id, model_id)
                image_urls.append({"url": uploaded_url, "finish_reason": "SUCCESS"})

            remove_images(local_files)

        return image_urls

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error: " + str(e))


class InpaintAndOutpaintRequest(BaseModel):
    user_id: str
    model_id: str
    file1: str
    file2: str
    prompt: str
    sharpness: Optional[str] = None
    cn_type1: Optional[str] = None
    base_model_name: Optional[str] = None
    style_selections: Optional[str] = None
    performance_selection: Optional[str] = None
    image_number: Optional[str] = None
    negative_prompt: Optional[str] = None
    image_strength: Optional[str] = None
    cfg_scale: Optional[str] = None
    samples: Optional[str] = None
    steps: Optional[str] = None
    init_image_mode: Optional[str] = None
    clip_guidance_preset: Optional[str] = None
    mask_source: Optional[str] = None
    model: Optional[str] = None


class InpaintAndOutpaintResponse(BaseModel):
    url: str
    finish_reason: str


@app.post("/inpaintAndOutpaint", response_model=List[InpaintAndOutpaintResponse])
async def inpaintAndOutpaint(request: InpaintAndOutpaintRequest):
    try:
        print(datetime.now())
        user_id = request.user_id
        model_id = request.model_id
        file1 = request.file1
        file2 = request.file2
        prompt = request.prompt
        sharpness = request.sharpness
        cn_type1 = request.cn_type1
        base_model_name = request.base_model_name
        style_selections = request.style_selections
        performance_selection = request.performance_selection
        image_number = request.image_number
        negative_prompt = request.negative_prompt
        image_strength = request.image_strength
        cfg_scale = request.cfg_scale
        samples = request.samples
        steps = request.steps
        init_image_mode = request.init_image_mode
        clip_guidance_preset = request.clip_guidance_preset
        mask_source = request.mask_source
        model = request.model

        download_image_response1 = requests.get(file1)
        download_image_response2 = requests.get(file2)

        if download_image_response1.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download image from the provided URL")
        if download_image_response2.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download image from the provided URL")

        image_urls = []
        local_files = []

        if model == "fooocus":
            headers = {
                "accept": "application/json",
            }
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
                image_urls.append({"url": uploaded_url, "finish_reason": "SUCCESS"})
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)

        elif model_id == "sd1-sdai":
            engine_id = STABILITY_ENGINE_ID
            api_host = STABILITY_API_HOST
            api_key = STABILITY_API_KEY
            if api_key is None:
                raise Exception("Missing Stability API key.")

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
                    "mask_source": mask_source,
                    "text_prompts[0][text]": prompt,
                    "clip_guidance_preset": clip_guidance_preset,
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

            for i, image in enumerate(data["artifacts"]):
                file_path = f"v1_txt2img_{random_number}_{current_date}_{i}.png"
                local_files.append(file_path)
                with open(file_path, "wb") as f:
                    f.write(base64.b64decode(image["base64"]))
                    image_url = await uploadImage(f"http://127.0.0.1:8005/images/{file_path}", user_id, model_id)
                    image_urls.append({"url": image_url, "finish_reason": "SUCCESS"})

            remove_images(local_files)

        elif model_id == "sd-getai":
            url = "https://api.getimg.ai/v1/stable-diffusion/inpaint"
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
            with open(file_path, "wb") as f:
                f.write(base64.b64decode(data["image"]))
                uploaded_url = await uploadImage(file_path, user_id, model_id)
                image_urls.append({"url": uploaded_url, "finish_reason": "SUCCESS"})

            remove_images(local_files)

        return image_urls

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error: " + str(e))


# class CalculateFidRequest(BaseModel):
#     stability_image: str
#     getimg_image: str


# class CalculateFidResponse(BaseModel):
#     stability_image_fid_score: float
#     getimg_image_fid_score: float
#     result: str


# @app.post("/calculate_fid", response_model=CalculateFidResponse)
# async def calculate_fid_endpoint(request: CalculateFidRequest):
#     try:
#         file1 = request.stability_image
#         file2 = request.getimg_image

#         download_image_response1 = requests.get(file1)
#         download_image_response2 = requests.get(file2)

#         if download_image_response1.status_code != 200:
#             raise HTTPException(status_code=400, detail="Failed to download image from the provided URL")
#         if download_image_response2.status_code != 200:
#             raise HTTPException(status_code=400, detail="Failed to download image from the provided URL")

#         model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

#         reference_image_folder = "images_training/converted"
#         reference_images = load_images_from_folder(reference_image_folder)
#         num_files = len(os.listdir(reference_image_folder))
#         reference_images = reference_images.reshape((num_files, 32, 32, 3))

#         uploaded_image = Image.open(BytesIO(download_image_response1.content)).resize((32, 32))
#         uploaded_image = np.array(uploaded_image)
#         uploaded_image = np.expand_dims(uploaded_image, axis=0)
#         uploaded_image = preprocess_input(uploaded_image)

#         uploaded_image2 = Image.open(BytesIO(download_image_response2.content)).resize((32, 32))
#         uploaded_image2 = np.array(uploaded_image2)
#         uploaded_image2 = np.expand_dims(uploaded_image2, axis=0)
#         uploaded_image2 = preprocess_input(uploaded_image2)

#         images1 = np.repeat(uploaded_image, num_files, axis=0)
#         images2 = np.repeat(uploaded_image2, num_files, axis=0)

#         reference_images = reference_images.astype('float32')
#         images1 = images1.astype('float32')
#         images2 = images2.astype('float32')

#         reference_images = scale_images(reference_images, (299, 299, 3))
#         images1 = scale_images(images1, (299, 299, 3))
#         images2 = scale_images(images2, (299, 299, 3))

#         images1 = preprocess_input(images1)
#         images2 = preprocess_input(images2)

#         fid_score1 = calculate_fid(model, reference_images, images1)
#         fid_score2 = calculate_fid(model, reference_images, images2)

#         result = ""
#         if fid_score1 < fid_score2:
#             result = "Stability model has displayed best results according to the FID score."
#         elif fid_score1 > fid_score2:
#             result = "GetImg model has displayed best results according to the FID score."
#         else:
#             result = "Both models have displayed similar results according to the FID score."

#         return {
#             "stability_image_fid_score": fid_score1,
#             "getimg_image_fid_score": fid_score2,
#             "result": result
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail="Internal server error")


# class CalculateClipScoreRequest(BaseModel):
#     prompt: str
#     stability_image: str
#     getimg_image: str


# class CalculateClipScoreResponse(BaseModel):
#     stability_image_clip_score: float
#     getimg_image_clip_score: float
#     result: str


# @app.post("/calculate_clip_score", response_model=CalculateClipScoreResponse)
# async def calculate_clip_score_endpoint(request: CalculateClipScoreRequest):
#     try:
#         prompt = request.prompt
#         file1 = request.stability_image
#         file2 = request.getimg_image

#         download_image_response1 = requests.get(file1)
#         download_image_response2 = requests.get(file2)

#         if download_image_response1.status_code != 200:
#             raise HTTPException(status_code=400, detail="Failed to download image from the provided URL")
#         if download_image_response2.status_code != 200:
#             raise HTTPException(status_code=400, detail="Failed to download image from the provided URL")

#         uploaded_image1 = Image.open(BytesIO(download_image_response1.content)).resize((32, 32))
#         uploaded_image1 = np.array(uploaded_image1)
#         uploaded_image1 = np.expand_dims(uploaded_image1, axis=0)
#         uploaded_image1 = preprocess_input(uploaded_image1)

#         uploaded_image2 = Image.open(BytesIO(download_image_response2.content)).resize((32, 32))
#         uploaded_image2 = np.array(uploaded_image2)
#         uploaded_image2 = np.expand_dims(uploaded_image2, axis=0)
#         uploaded_image2 = preprocess_input(uploaded_image2)

#         images1 = np.repeat(uploaded_image1, 1, axis=0)
#         images2 = np.repeat(uploaded_image2, 1, axis=0)

#         images1 = images1.reshape((1, 32, 32, 3))
#         images2 = images2.reshape((1, 32, 32, 3))

#         prompts = [prompt]

#         sd_clip_score1 = calculate_clip_score(images1, prompts)
#         sd_clip_score2 = calculate_clip_score(images2, prompts)

#         result = ""
#         if sd_clip_score1 < sd_clip_score2:
#             result = "GetImg model has displayed best results according to the Clip Score."
#         elif sd_clip_score1 > sd_clip_score2:
#             result = "Stability model has displayed best results according to the Clip Score"
#         else:
#             result = "Both models have displayed best results according to the clip score."

#         return {
#             "stability_image_clip_score": sd_clip_score1,
#             "getimg_image_clip_score": sd_clip_score2,
#             "result": result
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
