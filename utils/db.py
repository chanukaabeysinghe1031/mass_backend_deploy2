import cloudinary
import cloudinary.uploader
import cloudinary.api

from dotenv import load_dotenv

from setup_env import supabase

load_dotenv()

# config = cloudinary.config(secure=True)

cloudinary.config(
    cloud_name='det0mvsek',
    api_key='746385323859142',
    api_secret='XELzrPuxJoqUe9QrQ1l1WU3cu9Q'
)


async def uploadImage(image_path, user_id, model):
    try:
        # Upload the image to Cloudinary
        response = cloudinary.uploader.upload(image_path)
        srcURL = response.get('secure_url')

        print("****2. Upload an image****\nDelivery URL: ", srcURL, "\n")

        # Insert the image URL into Supabase
        data, count = supabase.table("images").insert({
            "user_id": user_id,
            "url": srcURL,
            "type": "gen",
            "model": model,
            "public": False
        }).execute()

        print(data, count)
        print("srcUrl", srcURL)

        return srcURL

    except Exception as e:
        print(e)
        return str(e)

# uploadImage("256ddc39-d8ea-45db-a01d-5834b54da3a7", "d21badef-bcec-4d1c-891f-7e474ae9a7fa")
