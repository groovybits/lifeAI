#!/usr/bin/env python

## Life AI LLM 
#
# Chris Kennedy 2023 (C) GPL
#
# Free to use for any use as in truly free software
# as Richard Stallman intended it to be.
#

import zmq
import argparse
import json
import traceback

import warnings
import urllib3
import signal
import time
import requests
import logging
import uuid

import nltk  # Import nltk for sentence tokenization
from threading import Thread

# Download the Punkt tokenizer models (only needed once)
nltk.download('punkt')

warnings.simplefilter(action='ignore', category=Warning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
from urllib3.exceptions import InsecureRequestWarning
warnings.simplefilter(action='ignore', category=InsecureRequestWarning)

# Function to group the text into subtitle groups
def get_subtitle_groups(text, sentence_count):
    sentences = nltk.sent_tokenize(text)  # Split text into sentences
    groups = []
    group = []
    for sentence in sentences:
        if len(group) <= sentence_count:  # Limit of N lines per group
            group.append(sentence)
        else:
            groups.append(group)
            group = [sentence]
    if group:  # Don't forget the last group
        groups.append(group)
    return groups

def clean_text(text):
    # Placeholder for text cleaning logic, which might include removing unwanted characters, etc.
    # For now, it simply returns the input text.
    return text

def send_data(zmq_sender, message):
    # Placeholder for the ZMQ send function, which should be defined to match your ZMQ setup.
    zmq_sender.send_json(message)

def stream_api_response(api_url, completion_params, zmq_sender, header_message, characters_per_line, sentence_count, segment_number):
    accumulated_text = ""

    logger.info(f"--- stream_api_response(): chat LLM streaming API response. to {api_url} with completion_params: {completion_params}")

    tokens = 0
    current_tokens = 0
    characters = 0
    with requests.post(api_url, json=completion_params, stream=True) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                logger.debug(f"--- stream_api_response(): chat LLM streaming API response: {json.dumps(decoded_line)}")
                if decoded_line.startswith('data: '):
                    message = json.loads(decoded_line[6:])
                    content = message.get('content', '')

                    if content:  # Only add to accumulated text if there is content
                        print(content, end="")
                        tokens += 1
                        current_tokens += 1
                        characters += len(content)
                        accumulated_text += content
                        header_message["tokens"] = current_tokens

                        if len(accumulated_text) >= characters_per_line:
                            segment_number = clean_and_send_group(accumulated_text, zmq_sender, header_message, segment_number, sentence_count)
                            accumulated_text = ""  # Reset the accumulator
                            current_tokens = 0
                            header_message["tokens"] = 0

    # If there's any remaining text after the loop, send it as well
    if accumulated_text:
        segment_number = clean_and_send_group(accumulated_text, zmq_sender, header_message, segment_number, sentence_count)

    logger.info(f"--- stream_api_response(): chat LLM streaming API response: {tokens} tokens, {characters} characters.")

def clean_and_send_group(text, zmq_sender, header_message, segment_number, sentence_count):
    # Clean the text
    cleaned_text = clean_text(text)

    # Split into sentences and group them
    groups = get_subtitle_groups(cleaned_text, sentence_count)

    # Send each group via ZMQ
    for group in groups:
        combined_lines = "\n".join(group)
        if combined_lines.strip():  # Ensure the group is not just whitespace
            header_message["text"] = combined_lines
            header_message["segment_number"] = segment_number
            header_message["timestamp"] = time.time()
            send_data(zmq_sender, header_message)
            segment_number += 1

    return segment_number

def run_llm(header_message, zmq_sender, api_url, characters_per_line, sentence_count, args, segment_number):
    logger.info(f"--- run_llm(): chat LLM generating text from request message.")

    # Prepare the message to send to the LLM
    header_message["text"] = f"User {header_message['username']} asked: {header_message['message'][:200]}...."
    header_message["segment_number"] = segment_number
    header_message["timestamp"] = time.time()
    send_data(zmq_sender, header_message.copy())
    segment_number += 1

    # Add llm info to header
    header_message["llm_info"] = {
        "maxtokens": args.maxtokens,
        "temperature": args.temperature,
        "stoptokens": args.stoptokens,
        "characters_per_line": characters_per_line,
        "sentence_count": sentence_count,
        "simplesplit": args.simplesplit,
        "autogenerate": args.autogenerate,
    }

    try:
        completion_params = {
            'prompt': header_message["llm_prompt"],
            'temperature': args.temperature,
            'top_k': 40,
            'top_p': 0.9,
            'n_keep': args.n_keep,
            'cache_prompt': not args.no_cache_prompt,
            'slot_id': -1,
            'stop': args.stoptokens.split(','),
            'stream': True,
        }

        if args.maxtokens:
            completion_params['n_predict'] = args.maxtokens
        
        # Start a new thread to stream the API response and send it back to the client
        streaming_thread = Thread(target=stream_api_response, args=(api_url, completion_params, zmq_sender, header_message, characters_per_line, sentence_count, segment_number))
        streaming_thread.start()
        streaming_thread.join()
    except Exception as e:
        logger.error(f"--- run_llm(): LLM exception: {e}")
        logger.error(f"{traceback.print_exc()}")
        return header_message.copy()

    return header_message.copy()

def create_prompt(header_message, args):
    prompt_context = ""
    if "context" in header_message:
        prompt_context = "\nContext:%s\n" % json.dumps(header_message["context"]).replace('"', '').replace('[', '').replace(']', '').replace(',', '\n').strip()

    instructions = args.systemprompt
    if args.episode:
        instructions = "Format the output like a TV episode script using markdown."

    prompt = (f"Personality: As {header_message['ainame']} You are {header_message['aipersonality']} "
              f"{instructions}\n\n"
              f"{args.roleenforcer.replace('{user}', header_message['username']).replace('{assistant}', header_message['ainame'])}\n"
              f"{args.promptcompletion.replace('{user_question}', header_message['message'])}")
    
    return prompt

def main(args):
    zmq_context = zmq.Context()
    receiver = None
    sender = None
    segment_number = 0

    # Set up the ZMQ receiver
    receiver = zmq_context.socket(zmq.PULL)
    logger.info(f"Connected to ZMQ at {args.input_host}:{args.input_port}")
    receiver.bind(f"tcp://{args.input_host}:{args.input_port}")

    # Set up the ZMQ sender
    sender = zmq_context.socket(zmq.PUB)
    logger.info(f"Bound to ZMQ out at {args.output_host}:{args.output_port}")
    sender.bind(f"tcp://{args.output_host}:{args.output_port}")

    while True:
        try:
            # Receive a message
            client_request = receiver.recv_json()

            # Extract information from client request
            header_message = {
                "segment_number": segment_number,
                "mediaid": client_request["mediaid"],
                "mediatype": client_request["mediatype"],
                "username": client_request["username"],
                "source": client_request["source"],
                "message": client_request["message"],
                "ainame": client_request.get("ainame", args.ai_name),
                "aipersonality": client_request.get("aipersonality", args.personality),
                "context": client_request.get("history", []),
                "tokens": 0,
                "timestamp": time.time(),
                "text": ""
            }
            
            logger.debug(f"LLM: received message:\n - {json.dumps(header_message, indent=2)}\n")

            # Generate the prompt
            prompt = create_prompt(header_message, args)
            header_message["llm_prompt"] = prompt

            # Call LLM function to process the request
            header_message = run_llm(header_message, sender, api_endpoint, args.characters_per_line, args.sentence_count, args, segment_number)

            # If run_llm returned None, we should handle this case.
            if header_message is None:
                logger.error(f"\nLLM: Failed to generate a response, trying again.")
                continue  # Continue to the next iteration of the loop.

        except Exception as e:
            logger.error(f"Exception occurred: {e}")
            logger.error(f"{traceback.print_exc()}")
            # Add some sleep time to prevent a tight loop in case of a recurring error
            time.sleep(1)

if __name__ == "__main__":
    model = "models/zephyr-7b-alpha.Q2_K.gguf"
    prompt_template = "Question: {user_question}\nAnswer: "
    role_enforcer = ("Give an Answer the message from {user} listed as a Question at the prompt below. "
                     "Stay in the role of {assistant} using the Context if listed to help generate a response.\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_host", type=str, default="127.0.0.1")
    parser.add_argument("--input_port", type=int, default=1500)
    parser.add_argument("--output_host", type=str, default="127.0.0.1")
    parser.add_argument("--output_port", type=int, default=2000)
    parser.add_argument("--maxtokens", type=int, default=0)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--model", type=str, default=model)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("--ai_name", type=str, default="GAIB")
    parser.add_argument("--systemprompt", type=str, default="The Groovy AI Bot that is here to help you find enlightenment and learn about technology of the future.")
    parser.add_argument("-e", "--episode", action="store_true", default=False, help="Episode mode, Output a TV Episode format script.")
    parser.add_argument("-pc", "--promptcompletion", type=str, default=prompt_template, help="Prompt completion like... `Question: {user_question}\nAnswer:`")
    parser.add_argument("-re", "--roleenforcer", type=str, default=role_enforcer, help="Role enforcer statement with {user} and {assistant} template names replaced by the actual ones in use.")
    parser.add_argument("-p", "--personality", type=str, default="friendly helpful compassionate bodhisattva guru.", help="Personality of the AI, choices are 'friendly' or 'mean'.")
    parser.add_argument("-sts", "--stoptokens", type=str, default="Question:", help="Stop tokens to use, do not change unless you know what you are doing!")
    parser.add_argument("-tp", "--characters_per_line", type=int, default=100, help="Minimum number of characters per buffer, buffer window before output.")
    parser.add_argument("-sc", "--sentence_count", type=int, default=2, help="Number of sentences per line.")
    parser.add_argument("-ag", "--autogenerate", action="store_true", default=False, help="Carry on long conversations, remove stop tokens.")
    parser.add_argument("--simplesplit", action="store_true", default=False, help="Simple split of text into lines, no sentence tokenization.")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds for LLM to respond.")
    parser.add_argument("--metal", action="store_true", default=False, help="Offload to metal MPS GPU")
    parser.add_argument("--cuda", action="store_true", default=False, help="Offload to CUDA GPU")
    parser.add_argument("--purgecontext", action="store_true", default=False, help="Purge context if it gets too large")
    parser.add_argument("--n_keep", type=int, default=0, help="Number of tokens to keep for the context.")
    parser.add_argument("--no_cache_prompt", action='store_true', help="Flag to disable caching of prompts.")
    parser.add_argument("-ll", "--loglevel", type=str, default="info", help="Logging level: debug, info...")
    parser.add_argument("--sub", action="store_true", default=False, help="Publish to a topic")
    parser.add_argument("--pub", action="store_true", default=False, help="Publish to a topic")
    parser.add_argument("--llm_port", type=int, default=8080)
    parser.add_argument("--llm_host", type=str, default="127.0.0.1")

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
    logging.basicConfig(filename=f"logs/llmAPI-{log_id}.log", level=LOGLEVEL)
    logger = logging.getLogger('GAIB')

    ch = logging.StreamHandler()
    ch.setLevel(LOGLEVEL)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Setup episode mode if enabled
    if args.episode:
        args.roleenforcer = args.roleenforcer.replace('Answer the question asked by', 'Create a story from the plotline given by')
        args.promptcompletion = args.promptcompletion.replace('Question:', 'Plotline:')
        args.stoptokens = ["Plotline:"]

    # Call the main function with the parsed arguments
    main(args)
