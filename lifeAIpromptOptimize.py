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
import json

from transformers import logging as trlogging
import warnings
import urllib3

from llama_cpp import Llama

"""
warnings.simplefilter(action='ignore', category=Warning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
from urllib3.exceptions import NotOpenSSLWarning
warnings.simplefilter(action='ignore', category=NotOpenSSLWarning)
trlogging.set_verbosity_error()
"""

def main():
    while True:
        # Receive a message
        segment_number = receiver.recv_string()
        id = receiver.recv_string()
        type = receiver.recv_string()
        username = receiver.recv_string()
        source = receiver.recv_string()
        message = receiver.recv_string()
        text = receiver.recv_string()

        print(f"\n---\nPrompt optimizer: received text #{text}\n")
        optimized_prompt = ""

        image_prompt_data = None
        prompt = f"{args.prompt}\n\nImageDescription: {text}\nImagePrompt:"
        print(f"Prompt optimizer: sending text to LLM:\n - {text}\n")
        try:
            image_prompt_data = llm_image(
                prompt,
                max_tokens=args.maxtokens,
                temperature=args.temperature,
                stream=False,
                stop=["ImageDescription:"]
            )

            # Confirm we have an image prompt
            if 'choices' in image_prompt_data and len(image_prompt_data["choices"]) > 0 and 'text' in image_prompt_data["choices"][0]:
                optimized_prompt = image_prompt_data["choices"][0]['text']
                print(f"\nSuccess! Generated image prompt:\n - {optimized_prompt}")

            if not optimized_prompt.strip():
                print(f"\nError! Image prompt generation failed, using original prompt:\n - {json.dumps(image_prompt_data)}")
                optimized_prompt = text
        except Exception as e:
            print(f"\nError! Image prompt generation llm didn't get any result:\n{str(e)}")
            optimized_prompt = text

        # Send the processed message
        sender.send_string(str(segment_number), zmq.SNDMORE)
        sender.send_string(id, zmq.SNDMORE)
        sender.send_string(type, zmq.SNDMORE)
        sender.send_string(username, zmq.SNDMORE)
        sender.send_string(source, zmq.SNDMORE)
        sender.send_string(message, zmq.SNDMORE)
        sender.send_string(optimized_prompt, zmq.SNDMORE)
        sender.send_string(text)

        print(f"\nPrompt optimizer: sent optimized prompt:\n - {optimized_prompt}\n")

if __name__ == "__main__":
    model = "models/zephyr-7b-alpha.Q2_K.gguf"
    prompt = "Take the ImageDescription and summarize it into a short 2 sentence description of under 200 tokens that explains the condensed picture visualized from the ImageDescription to use for image generation."
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_host", type=str, default="127.0.0.1")
    parser.add_argument("--input_port", type=int, default=2000)
    parser.add_argument("--output_host", type=str, default="127.0.0.1")
    parser.add_argument("--output_port", type=int, default=3001)
    parser.add_argument("--prompt", type=str, default=prompt)
    parser.add_argument("--maxtokens", type=int, default=200)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--gpulayers", type=int, default=0)
    parser.add_argument("--model", type=str, default=model)
    parser.add_argument("-d", "--debug", action="store_true", default=False)

    args = parser.parse_args()

    context = zmq.Context()

    # Set up the subscriber
    receiver = context.socket(zmq.SUB)
    print(f"connected to ZMQ in {args.input_host}:{args.input_port}")
    receiver.connect(f"tcp://{args.input_host}:{args.input_port}")
    receiver.setsockopt_string(zmq.SUBSCRIBE, "")

    # Set up the publisher
    sender = context.socket(zmq.PUB)
    print(f"binded to ZMQ out {args.output_host}:{args.output_port}")
    sender.bind(f"tcp://{args.output_host}:{args.output_port}")

    # LLM Model for image prompt generation
    llm_image = Llama(model_path=args.model,
                      n_ctx=args.context, verbose=args.debug, n_gpu_layers=args.gpulayers)

    main()

