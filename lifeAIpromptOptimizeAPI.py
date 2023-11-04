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
import traceback
import logging
import requests
import json
import re

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Remove image tags or Markdown image syntax
    text = re.sub(r'\!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'<img.*?>', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove any inline code blocks
    text = re.sub(r'`.*?`', '', text)
    
    # Remove any block code segments
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    
    # Remove special characters and digits (optional, be cautious)
    text = re.sub(r'[^a-zA-Z0-9\s.?,!\n]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def get_api_response(api_url, completion_params):
    logger.debug(f"promptOptimizerAPI LLM: POST to {api_url} with parameters {completion_params}")

    response = requests.request("POST", api_url, data=json.dumps(completion_params))

    logger.debug(f"LLM: Response status code: {response.status_code}")
    logger.debug(f"LLM: Response text: {response.text}")

    if response.status_code != 200:
        logger.error(f"Request failed with status code {response.status_code}: {response.text}")
        return None
    
    return response.json()

def run_llm(prompt, api_url, args):
    optimized_prompt = ""
    try:
        prompt = clean_text(prompt)
        completion_params = {
            'prompt': prompt[:args.context - len(prompt)],
            'temperature': args.temperature,
            'stop': args.stoptokens.split(','),
            'max_tokens': args.maxtokens,
            'stream': False,
        }

        if args.maxtokens:
            completion_params['n_predict'] = args.maxtokens
        
        response = None
        try:
            response = get_api_response(api_url, completion_params)
            #response = json.loads(response)
        except Exception as e:
            logger.error(f"{traceback.print_exc()}")
            logger.error(f"LLM exception: {str(e)}")

        """
        Response status code: 200
        Response text: {"content":"Generate according to: The output from the LLM\n\n
        Short Description: A result produced by a language learning model, as requested.",
        "generation_settings":{"frequency_penalty":0.0,"grammar":"","ignore_eos":false,
        "logit_bias":[],"mirostat":0,"mirostat_eta":0.10000000149011612,"mirostat_tau":5.0,
        "model":"/Volumes/BrahmaSSD/LLM/models/GGUF/zephyr-7b-beta.Q8_0.gguf","n_ctx":32768,"n_keep":0,
        "n_predict":120,"n_probs":0,"penalize_nl":true,"presence_penalty":0.0,"repeat_last_n":64,
        "repeat_penalty":1.100000023841858,"seed":4294967295,"stop":["Question:"],"stream":false,
        "temp":0.4000000059604645,"tfs_z":1.0,"top_k":40,"top_p":0.8999999761581421,"typical_p":1.0},
        "model":"/Volumes/BrahmaSSD/LLM/models/GGUF/zephyr-7b-beta.Q8_0.gguf",
        "prompt":"Take the ImageDescription and summarize it into a short 2 sentence description of under 200 tokens for image generation from the ImagePrompt.\n\nImageDescription: The output from the LLM\nImagePrompt:",
        "slot_id":0,"stop":true,"stopped_eos":true,"stopped_limit":false,"stopped_word":false,"stopping_word":"",
        "timings":{"predicted_ms":961.919,"predicted_n":27,"predicted_per_second":28.068891455517566,
        "predicted_per_token_ms":35.626629629629626,"prompt_ms":64.421,"prompt_n":0,"prompt_per_second":0.0,
        "prompt_per_token_ms":null},"tokens_cached":75,"tokens_evaluated":48,"tokens_predicted":27,"truncated":false}
        """
        # Confirm we have an image prompt
        if response and 'content' in response:
            optimized_prompt = clean_text(response["content"])
            logger.info(f"promptOptimizeAPI: LLM response: '{optimized_prompt}'")
        else:
            logger.error(f"Error! LLM prompt generation failed: '{response}'")
            optimized_prompt = ""
    except Exception as e:
        logger.error(f"{traceback.print_exc()}")
        logger.error(f"LLM exception: {str(e)}")

    return optimized_prompt

def main():
    prompt_template = "Use the following Text to create a short and summarized short Description for the {topic} as a summary in words."
    prompt = prompt_template.format(topic=args.topic)

    current_text_array = []
    combined_header_message = None
    in_combine = False

    while True:
        # Receive a message
        header_message = receiver.recv_json()
        if not header_message:
            logger.error("Error! No message received.")
            time.sleep(1)
            continue

        text = ""
        message = ""

        if "text" in header_message:
            text = header_message["text"]
        else:
            logger.error(f"Error! No text in message: {header_message}")
            continue

        if "message" in header_message:
            message = header_message["message"][:80]

        mediaid = header_message["mediaid"]
        timestamp = header_message["timestamp"]
        segment_number = header_message["segment_number"]
        md5sum = header_message["md5sum"]

        logger.debug(f"Prompt optimizer received header: {header_message}")
        logger.info(f" Prompt optimizer for {mediaid} #{segment_number} {timestamp} {md5sum} '{message}' - {text}")
        # check if enabled and combine prompts, once we have enough then we send them combined
        if args.combine_count > 1:
            current_text_array.append(text)
            if len(current_text_array) < args.combine_count:
                if not in_combine:
                    combined_header_message = header_message.copy()
                else:
                    if 'merged_packets' in combined_header_message:
                        combined_header_message["merged_count"] += 1
                        combined_header_message["merged_packets"].append(header_message)
                    else:
                        combined_header_message["merged_count"] = 0
                        combined_header_message["merged_packets"] = [header_message]
                in_combine = True
                continue
            else:
                text = " ".join(current_text_array)
                current_text_array = []

        full_prompt = f"Request: {prompt}\n\n{args.qprompt}: {message[:120]} - {text[:300]}\n{args.aprompt}:"

        optimized_prompt = ""
        try:
            logger.info(f"Prompt optimizer: sending text to LLM - {full_prompt}")
            optimized_prompt = run_llm(full_prompt, api_endpoint, args)

            if not optimized_prompt.strip():
                logger.error(f"Error! prompt generation generated an empty prompt, using original.")
                optimized_prompt = ""
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error! prompt generation llm failed with exception: %s" % str(e))
            optimized_prompt = ""

        # Add optimized prompt
        if optimized_prompt:
            header_message["optimized_text"] = clean_text(optimized_prompt)

        # if we combined text, then we need to send the original text as well
        if args.combine_count > 1:
            header_message["text"] = text

        # Send the processed message
        sender.send_json(header_message)

        logger.info(f"Optimized: {mediaid} #{segment_number} {timestamp} {md5sum} - {optimized_prompt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_port", type=int, default=8080)
    parser.add_argument("--llm_host", type=str, default="127.0.0.1")
    parser.add_argument("--input_host", type=str, default="127.0.0.1")
    parser.add_argument("--input_port", type=int, default=2000)
    parser.add_argument("--output_host", type=str, default="127.0.0.1")
    parser.add_argument("--output_port", type=int, default=3001)
    parser.add_argument("--topic", type=str, default="picture", 
                        help="Topic to use for image generation, default 'image generation'")
    parser.add_argument("--maxtokens", type=int, default=77)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("--qprompt", type=str, default="Text", 
                        help="Prompt to use for image generation, default Text")
    parser.add_argument("--aprompt", type=str, default="Description", 
                        help="Prompt to use for image generation, default Description")
    parser.add_argument("--metal", action="store_true", default=False, help="offload to metal mps GPU")
    parser.add_argument("--cuda", action="store_true", default=False, help="offload to metal cuda GPU")
    parser.add_argument("-ll", "--loglevel", type=str, default="info", help="Logging level: debug, info...")
    parser.add_argument("--n_keep", type=int, default=0, help="Number of tokens to keep for the context.")
    parser.add_argument("-sts", "--stoptokens", type=str, default="Text:,Request:", help="Stop tokens to use, do not change unless you know what you are doing!")
    parser.add_argument("--no_cache_prompt", action='store_true', help="Flag to disable caching of prompts.")
    parser.add_argument("--sub", action="store_true", default=False, help="Publish to a topic")
    parser.add_argument("--pub", action="store_true", default=False, help="Publish to a topic")
    parser.add_argument("--bind_output", action="store_true", default=False, help="Bind to a topic")
    parser.add_argument("--bind_input", action="store_true", default=False, help="Bind to a topic")
    parser.add_argument("--combine_count", type=int, default=0, help="Number of messages to combine into one prompt.")

    args = parser.parse_args()

    api_endpoint = f"http://{args.llm_host}:{args.llm_port}/completion"

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
    logger = logging.getLogger('promptOptimizeAPI')

    ch = logging.StreamHandler()
    ch.setLevel(LOGLEVEL)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    context = zmq.Context()
    receiver = None
    sender = None

    # Set up the subscriber
    if args.sub:
        receiver = context.socket(zmq.SUB)
        print(f"Setup ZMQ in {args.input_host}:{args.input_port}")
    else:
        receiver = context.socket(zmq.PULL)
        print(f"Setup ZMQ in {args.input_host}:{args.input_port}")

    if args.bind_input:
        receiver.bind(f"tcp://{args.input_host}:{args.input_port}")
    else:
        receiver.connect(f"tcp://{args.input_host}:{args.input_port}")

    if args.sub:
        receiver.setsockopt_string(zmq.SUBSCRIBE, "")

    # Set up the publisher
    if args.pub:
        sender = context.socket(zmq.PUB)
        print(f"binded to ZMQ out {args.output_host}:{args.output_port}")
    else:
        sender = context.socket(zmq.PUSH)
        print(f"binded to ZMQ out {args.output_host}:{args.output_port}")
        
    if args.bind_output:
        sender.bind(f"tcp://{args.output_host}:{args.output_port}")
    else:    
        sender.connect(f"tcp://{args.output_host}:{args.output_port}")

    main()

