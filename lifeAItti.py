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
from transformers import logging as trlogging
import warnings
import urllib3

warnings.simplefilter(action='ignore', category=Warning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
from urllib3.exceptions import NotOpenSSLWarning
warnings.simplefilter(action='ignore', category=NotOpenSSLWarning)
trlogging.set_verbosity_error()
model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
model_id = "runwayml/stable-diffusion-v1-5"

## Disable NSFW filters
pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                                torch_dtype=torch.float16,
                                                safety_checker = None,
                                                requires_safety_checker = False)

## Offload to GPU Metal
pipe = pipe.to("mps")

def main(input_port, output_port):
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind(f"tcp://*:{input_port}")

    sender = context.socket(zmq.PUSH)
    sender.bind(f"tcp://*:{output_port}")

    segment_number = 1

    while True:
        message = receiver.recv_string()
        ts, text = message.split(":", 1)

        image = pipe(text).images[0]

        print("Text to Image: recieved image #%s" % ts)
        if args.output_file:
            if args.image_format == "pil":
                image.save(args.output_file)
                print(f"Image saved to {args.output_file} as PIL\n")
            else:
                with open(args.output_file, 'wb') as f:
                    f.write(output)
                print(f"Payload written to {args.output_file}\n")

        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')  # Save it as PNG or JPEG depending on your preference
        img_byte_arr = img_byte_arr.getvalue()

        sender.send_string(str(segment_number), zmq.SNDMORE)
        sender.send(img_byte_arr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, required=True, help="Port for receiving text input")
    parser.add_argument("--output_port", type=int, required=True, help="Port for sending audio output")
    parser.add_argument("--output_file", type=str, default="", help="Output payloads to a file for analysis")
    parser.add_argument("--image_format", choices=["pil", "raw"], default="pil", help="Image format to save as. Choices are 'pil' or 'raw'.")

    args = parser.parse_args()
    main(args.input_port, args.output_port)

