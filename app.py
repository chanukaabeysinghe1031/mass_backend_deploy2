from typing import Optional, List

import uvicorn
from fastapi import FastAPI, HTTPException
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from models.clip_helpers import calculate_clip_score, calculate_single_clip_score
from models.fid_helpers import calculate_fid_score
from models.fooocus_model import process_fooocus_text_to_image, process_fooocus_image_to_image, \
    process_fooocus_inpaint_and_outpaint
from models.getimg_models import process_getimg_text_to_image, process_getimg_image_to_image, \
    process_getimg_inpaint_and_outpaint
from models.sd_models import process_stability_ultra, process_stability_core, process_stability_diffusion, \
    process_sd_text_to_image, process_sd_image_to_image, process_sd_inpaint_and_outpaint, process_stability_inpaint

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


class TextToImageRequest(BaseModel):
    user_id: str
    model_id: str
    input: dict
    height: Optional[int] = 1024
    width: Optional[int] = 1024
    aspect_ratio: Optional[str] = "1:1"


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
        height = request.height
        width = request.width
        aspect_ratio = request.aspect_ratio

        if model_id == "fooocus":
            return await process_fooocus_text_to_image(user_id, model_id, input_data)
        elif model_id == "sd1-sdai":
            return await process_sd_text_to_image(user_id, model_id, input_data, height, width, aspect_ratio)
        elif model_id == "sd-getai":
            return await process_getimg_text_to_image(user_id, model_id, input_data)
        elif model_id == "stability-ultra":
            return await process_stability_ultra(user_id, model_id, input_data, height, width, aspect_ratio)
        elif model_id == "stability-core":
            return await process_stability_core(user_id, model_id, input_data, height, width, aspect_ratio)
        elif model_id == "stability-diffusion":
            return await process_stability_diffusion(user_id, model_id, input_data, height, width, aspect_ratio)
        else:
            raise HTTPException(status_code=400, detail="Invalid model ID")

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

        if model_id == "fooocus":
            return await process_fooocus_image_to_image(user_id, model_id, image_url, prompt, negative_prompt,
                                                        image_strength, cfg_scale, samples, steps, init_image_mode)
        elif model_id == "sd1-sdai":
            return await process_sd_image_to_image(user_id, model_id, image_url, prompt, negative_prompt,
                                                   image_strength, cfg_scale, samples, steps, init_image_mode)
        elif model_id == "sd-getai":
            return await process_getimg_image_to_image(user_id, model_id, image_url, prompt, negative_prompt,
                                                       image_strength, cfg_scale, samples, steps, init_image_mode)
        else:
            raise HTTPException(status_code=400, detail="Invalid model ID")

    except Exception as e:
        print(f"Exception: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


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

        if model == "fooocus":
            return await process_fooocus_inpaint_and_outpaint(user_id, model_id, file1, file2, prompt, sharpness,
                                                              cn_type1, base_model_name, style_selections,
                                                              performance_selection, image_number, negative_prompt,
                                                              image_strength, cfg_scale, samples, steps,
                                                              init_image_mode, clip_guidance_preset, mask_source)
        elif model_id == "sd1-sdai":
            return await process_sd_inpaint_and_outpaint(user_id, model_id, file1, file2, prompt, sharpness, cn_type1,
                                                         base_model_name, style_selections, performance_selection,
                                                         image_number, negative_prompt, image_strength, cfg_scale,
                                                         samples, steps, init_image_mode, clip_guidance_preset,
                                                         mask_source)
        elif model_id == "sd-getai":
            return await process_getimg_inpaint_and_outpaint(user_id, model_id, file1, file2, prompt, sharpness,
                                                             cn_type1, base_model_name, style_selections,
                                                             performance_selection, image_number, negative_prompt,
                                                             image_strength, cfg_scale, samples, steps, init_image_mode,
                                                             clip_guidance_preset, mask_source)
        else:
            raise HTTPException(status_code=400, detail="Invalid model ID")

    except Exception as e:
        print(f"Exception: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


class StabilityInpaintRequest(BaseModel):
    user_id: str
    image: str
    prompt: str
    mask: Optional[str] = None
    negative_prompt: Optional[str] = ""
    seed: Optional[int] = 0
    output_format: Optional[str] = "png"


class StabilityInpaintResponse(BaseModel):
    url: str
    finish_reason: str


@app.post("/stabilityInpaint", response_model=List[StabilityInpaintResponse])
async def stability_inpaint(request: StabilityInpaintRequest):
    try:
        print(datetime.now())
        user_id = request.user_id
        image = request.image
        prompt = request.prompt
        mask = request.mask
        negative_prompt = request.negative_prompt
        seed = request.seed
        output_format = request.output_format

        return await process_stability_inpaint(user_id, image, prompt, mask, negative_prompt, seed, output_format)

    except Exception as e:
        print(f"Exception: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


class CalculateFidRequest(BaseModel):
    stability_image: str
    getimg_image: str


class CalculateFidResponse(BaseModel):
    stability_image_fid_score: float
    getimg_image_fid_score: float
    result: str


@app.post("/calculate_fid", response_model=CalculateFidResponse)
async def calculate_fid_endpoint(request: CalculateFidRequest):
    try:
        return await calculate_fid_score(request.stability_image, request.getimg_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


class CalculateClipScoreRequest(BaseModel):
    prompt: str
    stability_image: str
    getimg_image: str


class CalculateClipScoreResponse(BaseModel):
    stability_image_clip_score: float
    getimg_image_clip_score: float
    result: str


@app.post("/calculate_clip_score", response_model=CalculateClipScoreResponse)
async def calculate_clip_score_endpoint(request: CalculateClipScoreRequest):
    try:
        return await calculate_clip_score(request.prompt, request.stability_image, request.getimg_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


class CalculateSingleClipScoreRequest(BaseModel):
    prompt: str
    image_url: str


class CalculateSingleClipScoreResponse(BaseModel):
    clip_score: float


@app.post("/calculate_single_clip_score", response_model=CalculateSingleClipScoreResponse)
async def calculate_single_clip_score_endpoint(request: CalculateSingleClipScoreRequest):
    try:
        return await calculate_single_clip_score(request.prompt, request.image_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
