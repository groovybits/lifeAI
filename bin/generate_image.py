#!/usr/bin/env python
from openai import OpenAI
import base64

openai_client = OpenAI()

def save_image(data, file_path):
    # Strip out the header of the base64 string if present
    if ',' in data:
        header, data = data.split(',', 1)

    image = base64.b64decode(data)
    
    with open(file_path, "wb") as fh:
        fh.write(image)

    return image

def generate_openai(prompt, username="lifeai"):
    response = openai_client.images.generate(
    model="dall-e-3",
    prompt=prompt,
    size="1024x1024",
    quality="standard",
    style="natural",
    response_format="b64_json",
    user=username,
    n=1,
    )

    print(f"{response.data[0]}")

    image_url = response.data[0].url
    b64_json = response.data[0].b64_json

    revised_prompt = response.data[0].revised_prompt
    print(f"OpenAI revised prompt: {revised_prompt}")

    image = save_image(b64_json, "out.png")
    print(f"got url: {image_url}")
    
    return image

prompt = "Tibetan mountainside with temples and prayerflags colorful and a big blue sky and white clouds, fish eye view."

image = generate_openai(prompt)

