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
import inflect
import re

warnings.simplefilter(action='ignore', category=Warning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
from urllib3.exceptions import NotOpenSSLWarning
warnings.simplefilter(action='ignore', category=NotOpenSSLWarning)
trlogging.set_verbosity_error()

def clean_text(text):
    p = inflect.engine()

    def num_to_words(match):
        number = match.group()
        try:
            words = p.number_to_words(number)
        except inflect.NumOutOfRangeError:
            words = "[number too large]"
        return words

    text = re.sub(r'\b\d+(\.\d+)?\b', num_to_words, text)

    # Add a pause after punctuation
    text = text.replace('.', '. ')
    text = text.replace(',', ', ')
    text = text.replace('?', '? ')
    text = text.replace('!', '! ')

    return text[:300]

def main():
    while True:
        """ 
          header_message = {
            "segment_number": segment_number,
            "mediaid": mediaid,
            "mediatype": mediatype,
            "username": username,
            "source": source,
            "message": message,
            "text": text,
            "optimized_text": optimized_text,
        }"""
        # Receive a message
        header_message = receiver.recv_json()

        # get variables from header
        segment_number = header_message["segment_number"]
        optimized_prompt = ""
        if "optimized_text" in header_message:
            optimized_prompt = header_message["optimized_text"]
        else:
            optimized_prompt = header_message["text"]
            print(f"TTI: No optimized text, using original text.")

        print(f"Text to Image recieved optimized prompt:\n{header_message}.")

        image = pipe(clean_text(optimized_prompt)).images[0]

        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')  # Save it as PNG or JPEG depending on your preference
        image = img_byte_arr.getvalue()

        sender.send_json(header_message, zmq.SNDMORE)
        sender.send(image)

        print(f"Text to Image sent image #{segment_number}:\n - {optimized_prompt}")

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

