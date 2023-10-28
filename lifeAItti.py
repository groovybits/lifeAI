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

"""
warnings.simplefilter(action='ignore', category=Warning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
from urllib3.exceptions import NotOpenSSLWarning
warnings.simplefilter(action='ignore', category=NotOpenSSLWarning)
trlogging.set_verbosity_error()
"""

def main():
    while True:
        segment_number = receiver.recv_string()
        id = receiver.recv_string()
        type = receiver.recv_string()
        username = receiver.recv_string()
        source = receiver.recv_string()
        message = receiver.recv_string()
        prompt = receiver.recv_string()
        text = receiver.recv_string()

        print(f"Text to Image recieved text #{segment_number}\n - {prompt}.")

        image = pipe(prompt).images[0]

        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')  # Save it as PNG or JPEG depending on your preference
        image = img_byte_arr.getvalue()

        sender.send_string(str(segment_number), zmq.SNDMORE)
        sender.send_string(id, zmq.SNDMORE)
        sender.send_string(type, zmq.SNDMORE)
        sender.send_string(username, zmq.SNDMORE)
        sender.send_string(source, zmq.SNDMORE)
        sender.send_string(message, zmq.SNDMORE)
        sender.send_string(prompt, zmq.SNDMORE)
        sender.send_string(text, zmq.SNDMORE)
        sender.send(image)

        print(f"Text to Image sent image #{segment_number}:\n - {prompt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, default=3001, required=False, help="Port for receiving text input")
    parser.add_argument("--output_port", type=int, default=3002, required=False, help="Port for sending image output")
    parser.add_argument("--input_host", type=str, default="127.0.0.1", required=False, help="Port for receiving text input")
    parser.add_argument("--output_host", type=str, default="127.0.0.1", required=False, help="Port for sending image output")

    args = parser.parse_args()

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

    context = zmq.Context()
    receiver = context.socket(zmq.SUB)
    print("connected to ZMQ in: %s:%d" % (args.input_host, args.input_port))
    receiver.connect(f"tcp://{args.input_host}:{args.input_port}")
    receiver.setsockopt_string(zmq.SUBSCRIBE, "")

    sender = context.socket(zmq.PUB)
    print("binded to ZMQ out: %s:%d" % (args.output_host, args.output_port))
    sender.bind(f"tcp://{args.output_host}:{args.output_port}")

    main()

