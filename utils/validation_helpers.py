import re
from fastapi import HTTPException


def validate_resolution(resolution, max_pixels):
    width, height = resolution
    total_pixels = width * height
    if total_pixels > max_pixels:
        raise HTTPException(
            status_code=400,
            detail=f"Resolution exceeds the maximum allowed pixel count of {max_pixels}. Given resolution: {total_pixels} pixels."
        )
    return True


def validate_aspect_ratio(aspect_ratio):
    pattern = re.compile(r'^\d+:\d+$')
    if not pattern.match(aspect_ratio):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid aspect ratio format. Expected format is 'width:height'. Given: {aspect_ratio}."
        )
    return True


def validate_prompt_length(prompt, max_length=10000):
    if len(prompt) > max_length:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt exceeds the maximum allowed length of {max_length} characters. Given length: {len(prompt)}."
        )
    return True


def validate_image_dimensions(image_size, min_size=64, max_pixels=9437184):
    width, height = image_size
    if width < min_size or height < min_size:
        raise HTTPException(
            status_code=400,
            detail=f"Image dimensions are too small. Minimum size: {min_size} pixels. Given: {width}x{height}."
        )
    if width * height > max_pixels:
        raise HTTPException(
            status_code=400,
            detail=f"Image exceeds the maximum allowed pixel count of {max_pixels}. Given: {width * height} pixels."
        )
    return True
