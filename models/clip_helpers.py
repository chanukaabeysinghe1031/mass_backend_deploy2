import numpy as np
from PIL import Image
from io import BytesIO
from fastapi import HTTPException
import requests

from utils.clip_helpers import get_clip_score


async def calculate_clip_score(prompt: str, stability_image: str, getimg_image: str):
    download_image_response1 = requests.get(stability_image)
    download_image_response2 = requests.get(getimg_image)

    if download_image_response1.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download image from the provided URL")
    if download_image_response2.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download image from the provided URL")

    uploaded_image1 = Image.open(BytesIO(download_image_response1.content)).resize((32, 32))
    uploaded_image1 = np.array(uploaded_image1)
    uploaded_image1 = np.expand_dims(uploaded_image1, axis=0)

    uploaded_image2 = Image.open(BytesIO(download_image_response2.content)).resize((32, 32))
    uploaded_image2 = np.array(uploaded_image2)
    uploaded_image2 = np.expand_dims(uploaded_image2, axis=0)

    images1 = np.repeat(uploaded_image1, 1, axis=0)
    images2 = np.repeat(uploaded_image2, 1, axis=0)

    images1 = images1.reshape((1, 32, 32, 3))
    images2 = images2.reshape((1, 32, 32, 3))

    prompts = [prompt]

    sd_clip_score1 = get_clip_score(images1, prompts)
    sd_clip_score2 = get_clip_score(images2, prompts)

    result = ""
    if sd_clip_score1 < sd_clip_score2:
        result = "GetImg model has displayed best results according to the Clip Score."
    elif sd_clip_score1 > sd_clip_score2:
        result = "Stability model has displayed best results according to the Clip Score"
    else:
        result = "Both models have displayed best results according to the clip score."

    return {
        "stability_image_clip_score": sd_clip_score1,
        "getimg_image_clip_score": sd_clip_score2,
        "result": result
    }


async def calculate_single_clip_score(prompt: str, image_url: str):
    download_image_response = requests.get(image_url)

    if download_image_response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download image from the provided URL")

    uploaded_image = Image.open(BytesIO(download_image_response.content)).resize((32, 32))
    uploaded_image = np.array(uploaded_image)
    uploaded_image = np.expand_dims(uploaded_image, axis=0)

    images = np.repeat(uploaded_image, 1, axis=0)
    images = images.reshape((1, 32, 32, 3))

    prompts = [prompt]

    clip_score_value = get_clip_score(images, prompts)

    return {
        "clip_score": clip_score_value
    }