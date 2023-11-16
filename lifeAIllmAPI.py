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
import spacy ## python -m spacy download en_core_web_sm

# Download the Punkt tokenizer models (only needed once)
nltk.download('punkt')

warnings.simplefilter(action='ignore', category=Warning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
from urllib3.exceptions import InsecureRequestWarning
warnings.simplefilter(action='ignore', category=InsecureRequestWarning)

def extract_sensible_sentences(text):
    # Load the spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Process the text with spaCy
    doc = nlp(text)

    # Filter sentences based on some criteria (e.g., length, structure)
    sensible_sentences = [sent.text for sent in doc.sents if len(sent.text.split()) > 3 and is_sensible(sent.text)]

    return sensible_sentences

def is_sensible(sentence):
    # Implement a basic check for sentence sensibility
    # This is a placeholder - you'd need a more sophisticated method for real use
    return not bool(re.search(r'\b[a-zA-Z]{20,}\b', sentence))

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

def send_data(zmq_sender, message):
    # Placeholder for the ZMQ send function, which should be defined to match your ZMQ setup.
    zmq_sender.send_json(message)

def stream_api_response(header_message, api_url, completion_params, zmq_sender, characters_per_line, sentence_count):
    accumulated_text = ""
    logger.info(f"LLM streaming API response to {api_url} with completion_params: {completion_params}")

    tokens = 0
    current_tokens = 0
    characters = 0
    all_output = ""
    usermatch = re.search(r"(\.|\!|\?|\])\s*\b\w+:", accumulated_text)
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
                        all_output += content

                        # When checking for the break point, make sure to use the same text cleaning method for consistency
                        if (content.endswith("]\n")) or (len(accumulated_text) >= characters_per_line and ('.' in content or '?' in content or '!' in content or '\n' in content)):
                            remaining_text = ""
                            remaining_text_tokens = 0

                            header_message = send_group(accumulated_text, zmq_sender, header_message.copy(), sentence_count)
                            current_tokens = 0
                            header_message["tokens"] = remaining_text_tokens
                            header_message["text"] = remaining_text
                            accumulated_text = remaining_text
                        # check for a stop token like .,!?] and a following name without spaces and then a colon like . username:
                        elif (len(accumulated_text.split(" ")) > 3) and accumulated_text.endswith(":") and (accumulated_text.split(" ")[-2].endswith(".") or accumulated_text.split(" ")[-2].endswith("!") or accumulated_text.split(" ")[-2].endswith("?") or accumulated_text.split(" ")[-2].endswith("]")) and len(accumulated_text.split(" ")[-1]) > 1:
                            remaining_text = ""
                            remaining_text_tokens = 0
                            remaining_text = accumulated_text.split(" ")[-1]
                            remaining_text_tokens = len(remaining_text.split())
                            header_message = send_group(accumulated_text, zmq_sender, header_message.copy(), sentence_count)
                            current_tokens = 0
                            header_message["tokens"] = remaining_text_tokens
                            header_message["text"] = remaining_text
                            accumulated_text = remaining_text
                        elif len(accumulated_text) >= (characters_per_line * 1.5) and (content.endswith(" ") or content.endswith(",") or content.startswith(" ")):
                            remaining_text = ""
                            remaining_text_tokens = 0
                            if content.startswith(" ") and len(content) > 1:
                                remaining_text = content[1:]
                                remaining_text_tokens = len(remaining_text.split())
                                # remove the duplicated end of accumulated text that contains the content token
                                accumulated_text = accumulated_text[:-(len(content)-1)]
                            header_message = send_group(accumulated_text, zmq_sender, header_message.copy(), sentence_count)
                            current_tokens = 0
                            header_message["tokens"] = remaining_text_tokens
                            header_message["text"] = remaining_text
                            accumulated_text = remaining_text

    # If there's any remaining text after the loop, send it as well
    if accumulated_text:
        header_message = send_group(accumulated_text, zmq_sender, header_message.copy(), sentence_count)

    logger.info(f"LLM streamed API response: {tokens} tokens, {characters} characters.")
    header_message['text'] = all_output
    return header_message

def send_group(text, zmq_sender, header_message, sentence_count):
    #sensible_sentences = extract_sensible_sentences(text)
    #text = ' '.join(sensible_sentences)

    # clean text of [INST], [/INST], <<SYS>>, <</SYS>>, <s>, </s> tags
    exclusions = ["[INST]", "[/INST]", "<<SYS>>", "<</SYS>>", "<s>", "</s>"]
    for exclusion in exclusions:
        text = text.replace(exclusion, "")

    # Clean the text of any special tokens [\S+]
    text = re.sub(r"\[\\[A-Z]+\]", "", text)
    # CLean the [speaker:] strings to just the speaker name and a colon
    text = re.sub(r"\[?(.+?):\]?", r"\1:", text)

    # Split into sentences and group them
    groups = get_subtitle_groups(text, sentence_count)

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
    header_message["text"] = f"{header_message['mediatype']} message from {header_message['username']}: {header_message['message'][:300]}..."

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
        "characters_per_line": characters_per_line,
        "sentence_count": sentence_count,
    }

    try:
        completion_params = {
            'prompt': header_message["llm_prompt"],
            'temperature': args.temperature,
            'max_tokens': header_message["maxtokens"],
            'stream': True,
        }

        if int(header_message["maxtokens"]) > 0:
            completion_params['n_predict'] = int(header_message["maxtokens"])
        
        # Start a new thread to stream the API response and send it back to the client
        header_message = stream_api_response(header_message.copy(), 
                                             api_url, 
                                             completion_params, 
                                             zmq_sender, 
                                             characters_per_line, 
                                             sentence_count)
        
        # Send end frame
        # Prepare the message to send to the LLM
        end_header = header_message.copy()
        end_header["text"] = f"{args.end_message}"
        end_header["timestamp"] = int(round(time.time() * 1000))
        send_data(zmq_sender, end_header.copy())

    except Exception as e:
        logger.error(f"LLM exception: {e}")
        logger.error(f"{traceback.print_exc()}")
        return header_message.copy()

    return header_message.copy()

def main(args):
    zmq_context = zmq.Context()
    receiver = None
    sender = None

    history = []

    ## Continuity Counter initial value
    segment_number = 0

    # Set up the ZMQ receiver
    receiver = zmq_context.socket(zmq.PULL)
    logger.info(f"Connected to ZMQ at {args.input_host}:{args.input_port}")
    receiver.connect(f"tcp://{args.input_host}:{args.input_port}")

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
            header_message['client_request'] = client_request
            
            logger.debug(f"LLM: received message: - {json.dumps(header_message)}\n")
            logger.info(f"LLM: received message: - {header_message['message'][:30]}...")

            ## keep history arrays members total bytes under the args.context size
            # read through history array from newest member and count bytes, once they equal or are more than args.context size, remove the oldest member
            if args.n_keep > 0:
                while len(history) > args.n_keep:
                    history = history[1:]
            
            if not args.nopurgecontext:
                history_bytes = 0
                for i in range(len(history)-1, -1, -1):
                    history_bytes += len(history[i])
                    if history_bytes >= (args.context * args.contextpct): # purge the history if it is over the context size percentage
                        history = history[i+1:]
                        break

            prompt_context = ""
    
            if "context" in header_message and header_message["context"]:
                prompt_context = "Context: %s\n" % header_message["context"].replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("  ", " ")
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
                iprompt_l = ("Output as a full length TV episode formatted as character speaker parts with the syntax of 'name: lines' "
                             " where the speaker name has a colon after it then the speaker lines separated by new lines per speakers. "
                             "create a unique plotline and a surprise ending. Do not use spaces in character names. "
                             "keep speakers in separate paragraphs from one another always starting with the speaker name followed by a colon, "
                             "always break lines with 2 line breaks before changing speakers. Do not speak in run on sentences, "
                             "make sure they all are less than 120 lines before a period. Use the name 'narrator:' for any meta talk. "
                             "Make it like a transcript easy to automate reading and speaking.")

            # create a history of the conversation with system prompt at the start
            tmp_history = []
            current_system_prompt = system_prompt.format( # add the system prompt
                assistant = header_message["ainame"], 
                personality = header_message["aipersonality"], 
                instructions = iprompt_l, 
                output = oprompt_l)
            
            tmp_history.append("<s>[INST]<<SYS>>%s<</SYS>>[/INST]</s>" % current_system_prompt)
            tmp_history.extend(history) # add the history of the conversation
            tmp_history.append("<s>[INST]<<SYS>>%s<</SYS>>\n%s%s\n\n%s: %s[/INST]\n%s:" % (current_system_prompt,
                                                                            prompt_context, 
                                                                            user_prompt.format(user=header_message["username"], 
                                                                                Q=qprompt_l, 
                                                                                A=aprompt_l), 
                                                                                    qprompt_l, 
                                                                                     header_message["message"],
                                                                                     aprompt_l)) # add the question
            
            header_message["llm_prompt"] = "\n".join(tmp_history) # create the prompt
            logger.info(f"LLM: generated prompt: - {header_message['llm_prompt']}")

            # Call LLM function to process the request
            header_message = run_llm(header_message, sender, api_endpoint, args.characters_per_line, args.sentence_count, args)

            # store the history
            history.append(f"<s>[INST]{qprompt_l}: {header_message['message']}[/INST]\n{aprompt_l}: {header_message['text']}</s>")

            segment_number = header_message["segment_number"]
            timestamp = header_message["timestamp"]
            mediaid = header_message["mediaid"]
            logger.info(f"LLM: job #{mediaid} completed at {timestamp} segment number #{segment_number}.")

            logger.debug(f"LLM: completed with response: - {json.dumps(header_message)}\n")

        except Exception as e:
            logger.error(f"Exception occurred: {e}")
            logger.error(f"{traceback.print_exc()}")
            # Add some sleep time to prevent a tight loop in case of a recurring error
            time.sleep(1)

if __name__ == "__main__":

    ## Prompt setup
    qprompt = "Question"
    aprompt = "Answer"
    oprompt = "response"
    iprompt = ("Conversate and answer the message whether a question or generic comment, request given. "
               "Play your Personality role, do not break from it. Do not use spaces in character names. "
               "Do not output run on sentences, make sure they all are less than 120 lines before a period. "
               "Use the the format of 'Yourname:' for speaking lines always starting with your speaker name and your lines after, "
               "you are the sole speaker unless there is a guest brought in for you to talk to. do not use the name 'narrator:' or any meta talk. "
               "Speak in first person and conversate with the user. Talk about your previous conversations if any are listed in the Context, "
               "otherwise use the Contexxt as reference but do not regurgitate it.")

    system_prompt = ("Personality: As {assistant} {personality}{instructions}"
                     "Stay in the role of {assistant} using the Context and chat history if present to help generate the {output} as a continuation of the same sentiment and topics.\n")
    user_prompt = "Give an {A} for the message from {user} listed as a {Q} below. "

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_host", type=str, default="127.0.0.1")
    parser.add_argument("--input_port", type=int, default=1500)
    parser.add_argument("--output_host", type=str, default="127.0.0.1")
    parser.add_argument("--output_port", type=int, default=2000)
    parser.add_argument("--maxtokens", type=int, default=2000)
    parser.add_argument("--context", type=int, default=32768, help="Size of context for LLM so we can measure history fill.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("--ai_name", type=str, default="GAIB")
    parser.add_argument("-e", "--episode", action="store_true", default=False, help="Episode mode, Output a TV Episode format script.")
    parser.add_argument("-p", "--personality", type=str, default="friendly helpful compassionate bodhisattva guru.", help="Personality of the AI, choices are 'friendly' or 'mean'.")
    parser.add_argument("-tp", "--characters_per_line", type=int, default=120, help="Minimum number of characters per buffer, buffer window before output. default 100")
    parser.add_argument("-sc", "--sentence_count", type=int, default=1, help="Number of sentences per line.")
    parser.add_argument("--nopurgecontext", action="store_true", default=False, help="Don't Purge context, warning this will cause memory issues!")
    parser.add_argument("--n_keep", type=int, default=0, help="Number of messages to keep for the context.")
    parser.add_argument("--no_cache_prompt", action='store_true', help="Flag to disable caching of prompts.")
    parser.add_argument("--contextpct", type=float, default=0.75, help="Percentage of context to use for history.")
    parser.add_argument("-ll", "--loglevel", type=str, default="info", help="Logging level: debug, info...")
    parser.add_argument("--llm_port", type=int, default=8080)
    parser.add_argument("--llm_host", type=str, default="127.0.0.1")
    parser.add_argument("--end_message", type=str, default="The Groovy Life AI - groovylife.ai", help="End message to send to the client.")

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
    logger = logging.getLogger('liveAIllmAPI')

    ch = logging.StreamHandler()
    ch.setLevel(LOGLEVEL)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Call the main function with the parsed arguments
    main(args)
