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
import logging
import time

warnings.simplefilter(action='ignore', category=Warning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
from urllib3.exceptions import NotOpenSSLWarning
warnings.simplefilter(action='ignore', category=NotOpenSSLWarning)
trlogging.set_verbosity_error()

def clean_text(text):
    # clean text so it works in JSON
    text = text.replace('\\', '\\\\')  # Escape backslashes first
    text = text.replace('"', '\\"')    # Escape double quotes
    text = text.replace('\n', '\\n')   # Escape newlines
    text = text.replace('\r', '\\r')   # Escape carriage returns
    text = text.replace('\t', '\\t')   # Escape tabs
    text = text.replace('/', '\\/')    # usually, this isn't necessary.

    # Truncate the text to 300 characters if needed
    return text[:args.maxtokens]

def main():
    prompt_template = "Take the <title> - <summary> title and summary listed and transform it into a short summarized description to be used for {topic}."
    prompt = prompt_template.format(topic=args.topic)

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
        header_message = receiver.recv_json()
        # Receive a message
        if not header_message:
            logger.error("Error! No message received.")
            time.sleep(1)
            continue

        text = ""
        message = ""

        if "text" in header_message:
            text = header_message["text"]

        if "message" in header_message:
            message = header_message["message"][:80]

        logger.debug(f"\n---\nPrompt optimizer received {header_message}\n")

        logger.info(f"Message: - {message}\nText: - {text}")

        full_prompt = f"{prompt}\n\n{args.qprompt}: {message} - {text}\n{args.aprompt}:"

        optimized_prompt = ""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_host", type=str, default="127.0.0.1")
    parser.add_argument("--input_port", type=int, default=2000)
    parser.add_argument("--output_host", type=str, default="127.0.0.1")
    parser.add_argument("--output_port", type=int, default=3001)
    parser.add_argument("--topic", type=str, default="image generation", 
                        help="Topic to use for image generation, default 'image generation'")
    parser.add_argument("--maxtokens", type=int, default=120)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--model", type=str, default=model)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("--qprompt", type=str, default="Question:", 
                        help="Prompt to use for image generation, default ImageDescription")
    parser.add_argument("--aprompt", type=str, default="Answer:", 
                        help="Prompt to use for image generation, default ImagePrompt")
    parser.add_argument("--metal", action="store_true", default=False, help="offload to metal mps GPU")
    parser.add_argument("--cuda", action="store_true", default=False, help="offload to metal cuda GPU")
    parser.add_argument("-ll", "--loglevel", type=str, default="info", help="Logging level: debug, info...")
 
    args = parser.parse_args()

    LOGLEVEL = logging.INFO

    if args.loglevel == "info":
        LOGLEVEL = logging.INFO
    elif args.loglevel == "debug":
        LOGLEVEL = logging.DEBUG
    elif args.loglevel == "warning":
        LOGLEVEL = logging.WARNING
    else:
        LOGLEVEL = logging.INFO

    log_id = time.strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(filename=f"logs/promptOptimize-{log_id}.log", level=LOGLEVEL)
    logger = logging.getLogger('GAIB')

    ch = logging.StreamHandler()
    ch.setLevel(LOGLEVEL)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

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

