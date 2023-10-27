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

def main():
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    print("connected to ZMQ in: %s:%d" % (args.input_host, args.input_port))
    receiver.connect(f"tcp://{args.input_host}:{args.input_port}")
    #receiver.setsockopt_string(zmq.SUBSCRIBE, "")

    sender = context.socket(zmq.PUSH)
    print("binded to ZMQ out: %s:%d" % (args.output_host, args.output_port))
    sender.bind(f"tcp://{args.output_host}:{args.output_port}")

    while True:
        try:
            segment_number = receiver.recv_string()
            prompt = receiver.recv_string()
            text = receiver.recv_string()

            print(f"Text to Image recieved image #%s {prompt}." % segment_number)

            image = pipe(prompt).images[0]

            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')  # Save it as PNG or JPEG depending on your preference
            image = img_byte_arr.getvalue()

            sender.send_string(str(segment_number), zmq.SNDMORE)
            sender.send_string(prompt, zmq.SNDMORE)
            sender.send_string(text, zmq.SNDMORE)
            sender.send(image)

            print(f"Text to Image sent image #%s:\n - {text}" % segment_number)
        except Exception as e:
            print("Error in Text to Image: %s" % str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, default=3001, required=False, help="Port for receiving text input")
    parser.add_argument("--output_port", type=int, default=3002, required=False, help="Port for sending image output")
    parser.add_argument("--input_host", type=str, default="127.0.0.1", required=False, help="Port for receiving text input")
    parser.add_argument("--output_host", type=str, default="127.0.0.1", required=False, help="Port for sending image output")
    parser.add_argument("--width", type=int, default=1024, help="Width of the output image")
    parser.add_argument("--height", type=int, default=1024, help="Height of the output image")

    args = parser.parse_args()
    main()

