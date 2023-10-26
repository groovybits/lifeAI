#!/usr/bin/env python

## Life AI Prompt optimizer
#
# Chris Kennedy 2023 (C) GPL
#
# Free to use for any use as in truly free software
# as Richard Stallman intended it to be.
#

import zmq
import argparse
import textwrap
import io
import json

import torch
from transformers import logging as trlogging
import warnings
import urllib3

from llama_cpp import Llama

warnings.simplefilter(action='ignore', category=Warning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
from urllib3.exceptions import NotOpenSSLWarning
warnings.simplefilter(action='ignore', category=NotOpenSSLWarning)
trlogging.set_verbosity_error()

def main(input_host, input_port, output_host, output_port):
    context = zmq.Context()

    # Set up the subscriber
    receiver = context.socket(zmq.PULL)
    print(f"Connecting to input on {input_host}:{input_port}")
    receiver.connect(f"tcp://{input_host}:{input_port}")
    #receiver.setsockopt_string(zmq.SUBSCRIBE, "")

    # Set up the publisher
    sender = context.socket(zmq.PUSH)
    print(f"Binding to output on {output_host}:{output_port}")
    sender.bind(f"tcp://{output_host}:{output_port}")

    # LLM Model for image prompt generation
    llm_image = Llama(model_path=args.model,
                      n_ctx=args.context, verbose=args.debug, n_gpu_layers=args.gpulayers)

    while True:
        # Receive a message
        segment_number = receiver.recv_string()
        text = receiver.recv_string()
        print(f"Prompt optimizer: received text #{text}")
        optimized_prompt = ""

        image_prompt_data = None
        try:
            image_prompt_data = llm_image(
                f"{args.prompt}\n\nDescription: {text}\nImage:",
                max_tokens=args.maxtokens,
                temperature=args.temperature,
                stream=False,
                stop=["Description:"]
            )

            # Confirm we have an image prompt
            if 'choices' in image_prompt_data and len(image_prompt_data["choices"]) > 0 and 'text' in image_prompt_data["choices"][0]:
                optimized_prompt = image_prompt_data["choices"][0]['text']
                print(f"Got Image Prompt: {optimized_prompt}")

            if not optimized_prompt.strip():
                print(f"Image prompt generation failed, using original prompt: {json.dumps(image_prompt_data)}")
                optimized_prompt = text
        except Exception as e:
            print(f"Image prompt generation llm didn't get any result: {json.dumps(e)}")
            optimized_prompt = text

        # Send the processed message
        sender.send_string(str(segment_number), zmq.SNDMORE)
        sender.send_string(optimized_prompt)

if __name__ == "__main__":
    model = "models/mistral.7b.mistral-openorca.gguf_v2.q4_k_m.gguf"
    prompt = "You are an image prompt generator. Take the text and summarize it into a description for image generation."
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_host", type=str, default="127.0.0.1")
    parser.add_argument("--input_port", type=int, default=3000)
    parser.add_argument("--output_host", type=str, default="127.0.0.1")
    parser.add_argument("--output_port", type=int, default=3001)
    parser.add_argument("--prompt", type=str, default=prompt)
    parser.add_argument("--maxtokens", type=int, default=0)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--gpulayers", type=int, default=0)
    parser.add_argument("--model", type=str, default=model)
    parser.add_argument("-d", "--debug", action="store_true", default=False)

    args = parser.parse_args()
    main(args.input_host, args.input_port, args.output_host, args.output_port)

