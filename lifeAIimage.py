#!/usr/bin/env python

## Life AI Stable Diffusion module
#
# Chris Kennedy 2023 (C) GPL
#
# Free to use for any use as in truly free software
# as Richard Stallman intended it to be.
#

import zmq
import argparse
from transformers import VitsModel, AutoTokenizer
from PIL import Image, ImageDraw, ImageFont
import textwrap
import io

from diffusers import StableDiffusionPipeline
import torch


model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")


model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("mps")

def image_to_ascii(image, width):
    image = image.resize((width, int((image.height/image.width) * width * 0.55)), Image.LANCZOS)
    image = image.convert('L')  # Convert to grayscale

    pixels = list(image.getdata())
    ascii_chars = ["@", "#", "S", "%", "?", "*", "+", ";", ":", ",", "."]
    ascii_image = [ascii_chars[pixel//25] for pixel in pixels]
    ascii_image = ''.join([''.join(ascii_image[i:i+width]) + '\n' for i in range(0, len(ascii_image), width)])
    return ascii_image

def main(input_port, output_port):
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind(f"tcp://*:{input_port}")

    sender = context.socket(zmq.PUSH)
    sender.bind(f"tcp://*:{output_port}")

    segment_number = 1

    while True:
        message = receiver.recv_string()
        _, text = message.split(":", 1)

        image = pipe(text).images[0]

        if args.output_file:
            if args.image_format == "pil":
                image.save(args.output_file)
                print(f"Image saved to {args.output_file} as PIL\n")
            else:
                with open(args.output_file, 'wb') as f:
                    f.write(output)
                print(f"Payload written to {args.output_file}\n")
        else:
            payload_hex = image_to_ascii(image, 80)
            print(f"Image Payload (Hex):\n{payload_hex}\n")

        sender.send_string(str(segment_number), zmq.SNDMORE)
        sender.send(audiobuf.getvalue())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, required=True, help="Port for receiving text input")
    parser.add_argument("--output_port", type=int, required=True, help="Port for sending audio output")
    parser.add_argument("--output_file", type=str, default="", help="Output payloads to a file for analysis")
    parser.add_argument("--video_format", choices=["pil", "raw"], default="pil", help="Video format to save as. Choices are 'pil' or 'raw'.")

    args = parser.parse_args()
    main(args.input_port, args.output_port)

