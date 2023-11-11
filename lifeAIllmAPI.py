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
import time
import requests
import logging
import hashlib
import re

import nltk  # Import nltk for sentence tokenization

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
    # Remove URLs
    #text = re.sub(r'http[s]?://\S+', '', text)
    
    # Remove image tags or Markdown image syntax
    #text = re.sub(r'\!\[.*?\]\(.*?\)', '', text)
    #text = re.sub(r'<img.*?>', '', text)
    
    # Remove HTML tags
    #text = re.sub(r'<.*?>', '', text)
    
    # Remove any inline code blocks
    #text = re.sub(r'`.*?`', '', text)
    
    # Remove any block code segments
    #text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    
    # Remove special characters and digits (optional, be cautious)
    #text = re.sub(r'[^a-zA-Z0-9\s.?,!\n:\'\"\-\t]<>', '', text)
    
    # Remove extra whitespace
    #text = ' '.join(text.split())

    return text

def send_data(zmq_sender, message):
    # Placeholder for the ZMQ send function, which should be defined to match your ZMQ setup.
    zmq_sender.send_json(message)

def stream_api_response(header_message, api_url, completion_params, zmq_sender, characters_per_line, sentence_count):
    accumulated_text = ""
    logger.info(f"LLM streaming API response to {api_url} with completion_params: {completion_params}")

    tokens = 0
    current_tokens = 0
    characters = 0
    with requests.post(api_url, json=completion_params, stream=True) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                logger.debug(f"LLM streaming API response: {json.dumps(decoded_line)}")
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

                        cleaned_accumulated_text = clean_text(accumulated_text)

                        # When checking for the break point, make sure to use the same text cleaning method for consistency
                        if len(cleaned_accumulated_text) >= characters_per_line and ('.' in content or '?' in content or '!' in content or '\n' in content):
                            header_message = clean_and_send_group(cleaned_accumulated_text, zmq_sender, header_message.copy(), sentence_count)
                            current_tokens = 0
                            header_message["tokens"] = 0
                            header_message["text"] = ""
                            accumulated_text = ""

    # If there's any remaining text after the loop, send it as well
    if accumulated_text:
        header_message = clean_and_send_group(accumulated_text, zmq_sender, header_message.copy(), sentence_count)

    logger.info(f"LLM streamed API response: {tokens} tokens, {characters} characters.")
    return header_message

def clean_and_send_group(text, zmq_sender, header_message, sentence_count):
    # Clean the text
    cleaned_text = clean_text(text)

    # Split into sentences and group them
    groups = get_subtitle_groups(cleaned_text, sentence_count)

    # Send each group via ZMQ
    for group in groups:
        combined_lines = "\n".join(group)
        if combined_lines.strip():  # Ensure the group is not just whitespace
            md5text = hashlib.md5(text.encode('utf-8')).hexdigest()
            header_message["md5sum"] = md5text
            header_message["text"] = combined_lines
            header_message["timestamp"] = int(round(time.time() * 1000))
            send_data(zmq_sender, header_message.copy())
            logger.info(f"LLM: sent text #{header_message['segment_number']} {header_message['timestamp']} {header_message['md5sum']}:\n - {combined_lines[:30]}...")
            header_message["segment_number"] += 1
            header_message["text"] = ""

    return header_message

def run_llm(header_message, zmq_sender, api_url, characters_per_line, sentence_count, args):
    logger.info(f"LLM generating text for media id {header_message['mediaid']}.")

    # Prepare the message to send to the LLM
    header_message["text"] = f"{header_message['mediatype']} {header_message['username']}: {header_message['message'][:500]}...."

    # Send initial question
    header_message["timestamp"] = int(round(time.time() * 1000))
    send_data(zmq_sender, header_message.copy())
    logger.info(f"LLM: sent Question #{header_message['segment_number']} {header_message['timestamp']} {header_message['md5sum']}: - {header_message['text'][:30]}")
    header_message["segment_number"] += 1
    header_message["text"] = ""

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
            #'stop': args.stoptokens.split(','),
            'max_tokens': header_message["maxtokens"],
            'stream': True,
        }

        if header_message["maxtokens"] > 0:
            completion_params['n_predict'] = header_message["maxtokens"]
        
        # Start a new thread to stream the API response and send it back to the client
        header_message = stream_api_response(header_message.copy(), 
                                             api_url, 
                                             completion_params, 
                                             zmq_sender, 
                                             characters_per_line, 
                                             sentence_count)
        
        # Send end frame
        # Prepare the message to send to the LLM
        header_message["text"] = f"{args.end_message}"
        header_message["timestamp"] = int(round(time.time() * 1000))
        send_data(zmq_sender, header_message.copy())

    except Exception as e:
        logger.error(f"LLM exception: {e}")
        logger.error(f"{traceback.print_exc()}")
        return header_message.copy()

    return header_message.copy()

def create_prompt(header_message, args):
    prompt_context = ""
    
    if "context" in header_message and header_message["context"]:
        prompt_context = "Context: %s\n" % clean_text(header_message["context"])
    
    if prompt_context == "Context: \n":
        prompt_context = ""
        
    qprompt_l = qprompt
    aprompt_l = aprompt
    oprompt_l = oprompt
    iprompt_l = iprompt

    # Setup episode mode if enabled
    if header_message["episode"] == "true":
        qprompt_l = "Plotline"
        aprompt_l = "Episode"
        oprompt_l = "episode"
        iprompt_l = "Output as a TV episode full length character speaker parts with name[gender]: lines and a plotline it follows and a surprise ending. use [m], [f], and [n] as gender markers for speakers intended genders."

    args.stoptokens = f"{qprompt_l}:,Context:,Personality:,Question:"

    prompt = args.promptcompletion.format(question = header_message["message"],
                                          context = prompt_context, 
                                          instructions = iprompt_l, 
                                          user = header_message["username"], 
                                          assistant = header_message["ainame"],
                                          personality = header_message["aipersonality"],
                                          output = oprompt_l,
                                          Q = qprompt_l,
                                          A = aprompt_l)
    
    logger.info(f"LLM: prompt: - {prompt}")
    
    return prompt

def main(args):
    zmq_context = zmq.Context()
    receiver = None
    sender = None

    ## Continuity Counter initial value
    segment_number = 0

    # Set up the ZMQ receiver
    receiver = zmq_context.socket(zmq.PULL)
    logger.info(f"Connected to ZMQ at {args.input_host}:{args.input_port}")
    receiver.connect(f"tcp://{args.input_host}:{args.input_port}")
    #receiver.subscribe("")

    # Set up the ZMQ sender
    sender = zmq_context.socket(zmq.PUB)
    logger.info(f"Bound to ZMQ out at {args.output_host}:{args.output_port}")
    sender.bind(f"tcp://{args.output_host}:{args.output_port}")

    jobs = []
    while True:
        try:
            # Receive a message
            client_request = None
            if len(jobs) == 0:
                new_job = receiver.recv_json()
                jobs.append(new_job)
                while receiver.get(zmq.RCVMORE):
                    jobs.append(receiver.recv_json())

                # Define a custom sort key function
                def sort_key(job):
                    # Give priority to 'Twitch' by returning a tuple with the first element as boolean
                    return (job['source'] != 'Twitch', job['source'])

                # Sort jobs with Twitch as the priority
                jobs = sorted(jobs, key=sort_key)

            # get the current client request
            client_request = jobs.pop(0)

            is_episode = "false"
            if args.episode:
                is_episode = "true"

            # Extract information from client request
            header_message = {
                "segment_number": segment_number,
                "start_time": int(round(time.time() * 1000)),
                "timestamp": int(round(time.time() * 1000)),
                "mediaid": client_request["mediaid"],
                "mediatype": client_request["mediatype"],
                "username": client_request["username"],
                "source": client_request["source"],
                "message": client_request["message"],
                "episode": client_request.get("episode", is_episode),
                "ainame": client_request.get("ainame", args.ai_name),
                "aipersonality": client_request.get("aipersonality", args.personality),
                "context": client_request.get("history", ""),
                "tokens": 0,
                "md5sum": "",
                "index": 0,
                "text": "",
                "maxtokens": client_request.get("maxtokens", args.maxtokens),
                "voice_model": client_request.get("voice_model", "mimic3:en_US/cmu-arctic_low#eey:1.2")
            }
            
            logger.debug(f"LLM: received message: - {json.dumps(header_message)}\n")
            logger.info(f"LLM: received message: - {header_message['message'][:30]}...")

            # Generate the prompt
            prompt = create_prompt(header_message, args)
            header_message["llm_prompt"] = prompt

            logger.info(f"LLM: generated prompt: - {prompt}")

            # Call LLM function to process the request
            header_message = run_llm(header_message, sender, api_endpoint, args.characters_per_line, args.sentence_count, args)

            segment_number = header_message["segment_number"]
            timestamp = header_message["timestamp"]
            mediaid = header_message["mediaid"]
            header_message['client_request'] = client_request
            logger.info(f"LLM: job #{mediaid} completed at {timestamp} segment number #{segment_number}.")

            logger.debug(f"LLM: completed with response: - {json.dumps(header_message)}\n")

        except Exception as e:
            logger.error(f"Exception occurred: {e}")
            logger.error(f"{traceback.print_exc()}")
            # Add some sleep time to prevent a tight loop in case of a recurring error
            time.sleep(1)

if __name__ == "__main__":
    role_enforcer = ("Give an {A} for the message from {user} listed as a {Q} at the prompt below. "
                     "Stay in the role of {assistant} using the Context if present to help generate the {output}.\n")
    prompt_template = "<s>[INST]<<SYS>>Personality: As {assistant} {personality}{instructions}%s<</SYS>>\n\n{context}\n{Q}: {question}[/INST]\n{A}:" % role_enforcer
    qprompt = "Question"
    aprompt = "Answer"
    oprompt = "response"
    iprompt = "You are The Groovy AI Bot GAIB from Life AI who is here to help people find joy in learning about the technology of the future."

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_host", type=str, default="127.0.0.1")
    parser.add_argument("--input_port", type=int, default=1500)
    parser.add_argument("--output_host", type=str, default="127.0.0.1")
    parser.add_argument("--output_port", type=int, default=2000)
    parser.add_argument("--maxtokens", type=int, default=2000)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("--ai_name", type=str, default="GAIB")
    parser.add_argument("--systemprompt", type=str, default=f"System prompt, default is: {iprompt}")
    parser.add_argument("-pc", "--promptcompletion", type=str, default=prompt_template, help=f"Prompt Template, default is {prompt_template}.")
    parser.add_argument("-e", "--episode", action="store_true", default=False, help="Episode mode, Output a TV Episode format script.")
    parser.add_argument("-p", "--personality", type=str, default="friendly helpful compassionate bodhisattva guru.", help="Personality of the AI, choices are 'friendly' or 'mean'.")
    parser.add_argument("-sts", "--stoptokens", type=str, default="Question:,Context:,Personality:", help="Stop tokens to use, do not change unless you know what you are doing!")
    parser.add_argument("-tp", "--characters_per_line", type=int, default=80, help="Minimum number of characters per buffer, buffer window before output.")
    parser.add_argument("-sc", "--sentence_count", type=int, default=1, help="Number of sentences per line.")
    parser.add_argument("-ag", "--autogenerate", action="store_true", default=False, help="Carry on long conversations, remove stop tokens.")
    parser.add_argument("--simplesplit", action="store_true", default=False, help="Simple split of text into lines, no sentence tokenization.")
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
    parser.add_argument("--end_message", type=str, default="The Groovy Life AI - www.groovylife.ai", help="End message to send to the client.")

    args = parser.parse_args()

    if args.systemprompt:
        iprompt = args.systemprompt

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
    logger = logging.getLogger('liveAIllmAPI')

    ch = logging.StreamHandler()
    ch.setLevel(LOGLEVEL)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Call the main function with the parsed arguments
    main(args)
