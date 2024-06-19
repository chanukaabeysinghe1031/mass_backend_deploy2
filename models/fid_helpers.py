import os
import numpy as np
from PIL import Image
from fastapi import HTTPException
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from io import BytesIO
import requests
from scipy.linalg import sqrtm

def load_images_from_folder(folder):
    images = []
    if not os.path.exists(folder):
        raise HTTPException(status_code=400, detail="Folder does not exist.")
    try:
        for filename in os.listdir(folder):
            img = Image.open(os.path.join(folder, filename)).convert('RGB')
            if img is not None:
                images.append(np.array(img))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred: {e}")
    return images

def scale_images(images, new_shape):
    images_list = []
    for image in images:
        new_image = Image.fromarray(image).resize(new_shape[:2])
        images_list.append(np.array(new_image))
    return np.array(images_list)

def calculate_fid(model, act1, act2):
    try:
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred: {e}")

async def calculate_fid_score(stability_image: str, getimg_image: str):
    download_image_response1 = requests.get(stability_image)
    download_image_response2 = requests.get(getimg_image)

    if download_image_response1.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download image from the provided URL")
    if download_image_response2.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download image from the provided URL")

    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    reference_image_folder = "images_training/converted"
    reference_images = load_images_from_folder(reference_image_folder)
    num_files = len(reference_images)

    uploaded_image1 = Image.open(BytesIO(download_image_response1.content)).convert('RGB').resize((299, 299))
    uploaded_image2 = Image.open(BytesIO(download_image_response2.content)).convert('RGB').resize((299, 299))

    uploaded_image1 = np.expand_dims(np.array(uploaded_image1), axis=0)
    uploaded_image2 = np.expand_dims(np.array(uploaded_image2), axis=0)

    reference_images = scale_images(reference_images, (299, 299, 3))

    reference_images = preprocess_input(reference_images)
    uploaded_image1 = preprocess_input(uploaded_image1)
    uploaded_image2 = preprocess_input(uploaded_image2)

    act_ref = model.predict(reference_images)
    act_img1 = model.predict(uploaded_image1)
    act_img2 = model.predict(uploaded_image2)

    fid_score1 = calculate_fid(model, act_ref, act_img1)
    fid_score2 = calculate_fid(model, act_ref, act_img2)

    result = ""
    if fid_score1 < fid_score2:
        result = "Stability model has displayed the best results according to the FID score."
    elif fid_score1 > fid_score2:
        result = "GetImg model has displayed the best results according to the FID score."
    else:
        result = "Both models have displayed similar results according to the FID score."

    return {
        "stability_image_fid_score": fid_score1,
        "getimg_image_fid_score": fid_score2,
        "result": result
    }
