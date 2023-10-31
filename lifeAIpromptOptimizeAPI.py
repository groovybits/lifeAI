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
import time
import warnings
import urllib3
import traceback
import logging
import requests

warnings.simplefilter(action='ignore', category=Warning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
from urllib3.exceptions import NotOpenSSLWarning
warnings.simplefilter(action='ignore', category=NotOpenSSLWarning)

def get_api_response(api_url, completion_params):
    logger.debug(f"\n--- stream_api_response(): Sending POST request to {api_url} with completion_params: {completion_params}")
    response = requests.post(api_url, json=completion_params)
    
    print(f"Response status code: {response.status_code}")
    print(f"Response text: {response.text}")

    if response.status_code != 200:
        logger.error(f"Request failed with status code {response.status_code}: {response.text}")
        return None
    
    return response.json()  # Assuming the server returns JSON response

def run_llm(prompt, api_url, args):
    print(f"\n--- run_llm(): chat LLM generating text from request message.")
    logger.info(f"--- run_llm(): chat LLM generating text from request message.")

    try:
        completion_params = {
            'prompt': prompt,
            'temperature': args.temperature,
            'top_k': 40,
            'top_p': 0.9,
            'n_keep': args.n_keep,
            'cache_prompt': not args.no_cache_prompt,
            'slot_id': -1,
            'stop': args.stoptokens.split(','),
            'stream': False,
        }

        if args.maxtokens:
            completion_params['n_predict'] = args.maxtokens
        
        return get_api_response(api_url, completion_params)
    except Exception as e:
        logger.error(f"--- run_llm(): LLM exception: {e}")
        traceback.print_exc()

    return None

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

        logger.debug(f"\n---\nPrompt optimizer received {header_message}\n")
        optimized_prompt = ""

        image_prompt_data = None
        full_prompt = f"{prompt}\n\n{args.qprompt}: {text}\n{args.aprompt}:"
        logger.info(f"Prompt optimizer: sending text to LLM:\n - {text}\n")
        print(f"Prompt optimizer: sending text to LLM:\n - {text}\n")
        try:
            image_prompt_data = run_llm(full_prompt, args.api_endpoint, args)

            # image_prompt_data["choices"][0]["text"]

            # Confirm we have an image prompt
            if 'choices' in image_prompt_data and len(image_prompt_data["choices"]) > 0 and 'text' in image_prompt_data["choices"][0]:
                optimized_prompt = image_prompt_data["choices"][0]["text"]
                logger.info(f"\nSuccess! Generated image prompt:\n - {optimized_prompt}")
                print(f"\nSuccess! Generated image prompt:\n - {optimized_prompt}")
            else:
                logger.error(f"\nError! Image prompt generation failed, using original prompt:\n - {image_prompt_data}")
                optimized_prompt = None

            if not optimized_prompt.strip():
                logger.error(f"\nError! Image prompt generation empty, using original prompt:\n - {image_prompt_data}")
                optimized_prompt = None
        except Exception as e:
            logger.error(f"\nError! Image prompt generation llm didn't get any result:\n{str(e)}")
            optimized_prompt = None

        # Add optimized prompt
        if optimized_prompt:
            header_message["optimized_text"] = optimized_prompt

        # Send the processed message
        sender.send_json(header_message)

        logger.info(f"\nPrompt optimizer input text:\n - {text}\ngenerated optimized prompt:\n - {optimized_prompt}\n")
        print(f"\nPrompt optimizer input text:\n - {text}\ngenerated optimized prompt:\n - {optimized_prompt}\n")

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
    parser.add_argument("--maxtokens", type=int, default=120)
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
    parser.add_argument("-ll", "--loglevel", type=str, default="info", help="Logging level: debug, info...")
    parser.add_argument("--api_endpoint", type=str, default="http://127.0.0.1:8080/completion", help="API endpoint for LLM completion.")
    parser.add_argument("--n_keep", type=int, default=0, help="Number of tokens to keep for the context.")
    parser.add_argument("-sts", "--stoptokens", type=str, default="Question:", help="Stop tokens to use, do not change unless you know what you are doing!")
    parser.add_argument("--no_cache_prompt", action='store_true', help="Flag to disable caching of prompts.")

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
    logging.basicConfig(filename=f"logs/promptOptimizeAPI-{log_id}.log", level=LOGLEVEL)
    logger = logging.getLogger('GAIB')

    ch = logging.StreamHandler()
    ch.setLevel(LOGLEVEL)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

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

    main()

