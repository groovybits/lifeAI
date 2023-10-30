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

warnings.simplefilter(action='ignore', category=Warning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
from urllib3.exceptions import NotOpenSSLWarning
warnings.simplefilter(action='ignore', category=NotOpenSSLWarning)
trlogging.set_verbosity_error()

def main():
    while True:
        """ From LLM Source
          header_message = {
            "segment_number": segment_number,
            "mediaid": mediaid,
            "mediatype": mediatype,
            "username": username,
            "source": source,
            "message": message,
            "text": "",
        }
        """
        # Receive a message
        header_message = receiver.recv_json()
        text = header_message["text"]

        print(f"\n---\nPrompt optimizer received {header_message}\n")
        optimized_prompt = ""

        image_prompt_data = None
        full_prompt = f"{prompt}\n\n{args.qprompt}: {text}\n{args.aprompt}:"
        print(f"Prompt optimizer: sending text to LLM:\n - {text}\n")
        try:
            image_prompt_data = llm_image(
                full_prompt,
                max_tokens=args.maxtokens,
                temperature=args.temperature,
                stream=False,
                stop=[f"{args.qprompt}:"]
            )

            # Confirm we have an image prompt
            if 'choices' in image_prompt_data and len(image_prompt_data["choices"]) > 0 and 'text' in image_prompt_data["choices"][0]:
                optimized_prompt = image_prompt_data["choices"][0]['text']
                print(f"\nSuccess! Generated image prompt:\n - {optimized_prompt}")

            if not optimized_prompt.strip():
                print(f"\nError! Image prompt generation failed, using original prompt:\n - {json.dumps(image_prompt_data)}")
                optimized_prompt = None
        except Exception as e:
            print(f"\nError! Image prompt generation llm didn't get any result:\n{str(e)}")
            optimized_prompt = None

        # Add optimized prompt
        if optimized_prompt:
            header_message["optimized_text"] = optimized_prompt

        # Send the processed message
        sender.send_json(header_message)

        print(f"\nPrompt optimizer generated optimized text:\n - {text}\nprompt:\n - {optimized_prompt}\n")

if __name__ == "__main__":
    model = "models/zephyr-7b-alpha.Q2_K.gguf"
    prompt = "Take the {qprompt} and summarize it into a short 2 sentence description of under 200 tokens for {topic} from the {aprompt}."
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_host", type=str, default="127.0.0.1")
    parser.add_argument("--input_port", type=int, default=2000)
    parser.add_argument("--output_host", type=str, default="127.0.0.1")
    parser.add_argument("--output_port", type=int, default=3001)
    parser.add_argument("--topic", type=str, default="image generation", 
                        help="Topic to use for image generation, default 'image generation'")
    parser.add_argument("--maxtokens", type=int, default=200)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--model", type=str, default=model)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("--qprompt", type=str, default="ImageDescription", 
                        help="Prompt to use for image generation, default ImageDescription")
    parser.add_argument("--aprompt", type=str, default="ImagePrompt", 
                        help="Prompt to use for image generation, default ImagePrompt")
    parser.add_argument("--metal", action="store_true", default=False, help="offload to metal mps GPU")
    parser.add_argument("--cuda", action="store_true", default=False, help="offload to metal cuda GPU")
    args = parser.parse_args()

    prompt = prompt.format(qprompt=args.qprompt, aprompt=args.aprompt, topic=args.topic)

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
    gpulayers = 0
    if args.metal or args.cuda:
       gpulayers = -1 
    llm_image = Llama(model_path=args.model,
                      n_ctx=args.context, verbose=args.debug, n_gpu_layers=gpulayers)

    main()

