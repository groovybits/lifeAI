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

from transformers import logging as trlogging
import warnings
import urllib3

from llama_cpp import Llama, ChatCompletionMessage
import nltk  # Import nltk for sentence tokenization

# Download the Punkt tokenizer models (only needed once)
nltk.download('punkt')

warnings.simplefilter(action='ignore', category=Warning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
from urllib3.exceptions import NotOpenSSLWarning
warnings.simplefilter(action='ignore', category=NotOpenSSLWarning)
trlogging.set_verbosity_error()

# Function to group the text into subtitle groups
def get_subtitle_groups(text):
    sentences = nltk.sent_tokenize(text)  # Split text into sentences
    groups = []
    group = []
    for sentence in sentences:
        if len(group) <= args.sentence_count:  # Limit of N lines per group
            group.append(sentence)
        else:
            groups.append(group)
            group = [sentence]
    if group:  # Don't forget the last group
        groups.append(group)
    return groups

def clean_text(text):
    cleaned_text = text.encode('ascii', 'ignore').decode('ascii')
    cleaned_text = cleaned_text.replace('<|assistant|>',"")
    cleaned_text = cleaned_text.replace('<|/assistant|>',"")
    cleaned_text = cleaned_text.replace('<|>',"")
    cleaned_text = cleaned_text.replace('</|>',"")
    cleaned_text = cleaned_text.replace('<|user|>',"")
    cleaned_text = cleaned_text.replace('<|/user|>',"")
    cleaned_text = cleaned_text.replace('<<SYS>>',"")
    cleaned_text = cleaned_text.replace('<</SYS>>',"")
    cleaned_text = cleaned_text.replace('[INST]',"")
    cleaned_text = cleaned_text.replace('[/INST]',"")
    return cleaned_text

def prepare_send_data(header_message, combined_lines):
    header_message["text"] = combined_lines
    data = header_message
    return data

def send_data(data):
    sender.send_json(data)

def decide_and_send(accumulator, header_message):
    combined_lines = "".join([line for line in accumulator if line.strip()])
    combined_lines = clean_text(combined_lines)
    if combined_lines:
        data = prepare_send_data(header_message, combined_lines)
        send_data(data)
        return True  # Indicates that data was sent
    return False  # Indicates that no data was sent

def run_llm(header_message, user_messages):
    segment_number = int(header_message["segment_number"])
    response_text = ""
    print(f"\n--- run_llm(): chat LLM generating text from request message:\n - {header_message}\n")

    # collect llm info in header
    header_message["llm_info"] = {
        "maxtokens": args.maxtokens,
        "temperature": args.temperature,
        "stoptokens": args.stoptokens,
        "characters_per_line": args.characters_per_line,
        "sentence_count": args.sentence_count,
        "simplesplit": args.simplesplit,
        "autogenerate": args.autogenerate,
    }

    output = llm.create_chat_completion(
        messages=user_messages,
        max_tokens=args.maxtokens,
        temperature=args.temperature,
        stream=True,
        stop=args.stoptokens.split(',') if args.stoptokens else []
    )

    accumulator = []
    token_count = 0
    total_tokens = 0
    found_question = True
    for item in output:
        delta = item["choices"][0]['delta']
        header_message["llm_choices"] = item["choices"]

        # End of the output
        if 'content' not in delta:
            if 'finish_reason' in item["choices"][0] and item["choices"][0]['finish_reason'] == "stop":
                print(f"--- run_llm(): LLM response token stop: {json.dumps(item)}")
                # store stop message from LLM
                header_message["llm_stop"] = item
                break
            print(f"--- Skipping LLM response token lack of content: {json.dumps(item)}")
            continue

        token = delta['content']
        print(token, end='', flush=True)
        total_tokens += 1
        response_text += token

        accumulator.append(token)
        token_count += len(token)

        # Convert accumulator list to a string
        accumulator_str = ''.join(accumulator)

        if 'Question: ' in accumulator_str:
            # remove everything before Question: including Question: in accumulator array
            found_question = True

        # Check if it's time to send data
        if found_question and token_count >= args.characters_per_line:
            split_index = -1
            # Find the last occurrence of punctuation followed by a space or a newline
            for punct in ['.\s', '!\s', '?\s', '\n']:
                index = accumulator_str.rfind(punct)
                if index > split_index:
                    split_index = index + 1  # Include punctuation

            # Ensure we have enough characters to split
            if split_index >= 0 and split_index <= len(accumulator_str) - args.characters_per_line:
                pre_split = accumulator_str[:split_index]
                post_split = accumulator_str[split_index:]

                # Send subtitle groups for pre_split text
                groups = get_subtitle_groups(pre_split)
                for group in groups:
                    combined_lines = "\n".join(group)
                    combined_lines = clean_text(combined_lines)
                    if combined_lines:
                        header_message["text"] = combined_lines
                        header_message["segment_number"] = segment_number
                        send_data(header_message.copy())  # Send the data with the current segment number
                        segment_number += 1  # Increment for the next round

                # Clear accumulator and update token_count for the next round
                accumulator = [post_split]
                token_count = len(post_split)
            else:
                # If there's no suitable split point, wait for more content
                continue

        # Send any remaining tokens in accumulator
    if accumulator:
        remaining_text = ''.join(accumulator)
        if len(remaining_text) >= args.characters_per_line:
            groups = get_subtitle_groups(remaining_text)
            for group in groups:
                combined_lines = "\n".join(group)
                combined_lines = clean_text(combined_lines)
                if combined_lines:
                    header_message["text"] = combined_lines
                    header_message["segment_number"] = segment_number
                    send_data(header_message.copy())  # Send the data with the current segment number
                    segment_number += 1  # Increment for the next round


    print(f"\n--- run_llm(): finished generating text with {total_tokens} tokens and {segment_number + 1} segments for request:\n - {header_message}\n")
    header_message['text'] = response_text
    header_message['segment_number'] = segment_number  # Ensure the final segment number is correct
    return header_message.copy()

def create_prompt(header_message):
    ## Context inclusion if we have vectorDB results
    prompt_context = ""
    if "context" in header_message:
        # join array of history messsges for context
        prompt_context = "\nContext:%s\n" % " ".join(header_message["context"])
    username = header_message["username"]
    ainame = header_message["ainame"]
    aipersonality = header_message["aipersonality"]
    question = header_message["message"]

    ## Prompt parts
    instructions = "Use the context as inspiration and references for your answers the questions or requests asked from various sources like Twitch chat or a news feed."
    if args.episode:
        # instructions altered for generating an episode script
        instructions = "Use the context as inspiration and references for requests with a plotline for a story from various sources like Twitch chat or a news feed. Format the output like a TV episode script using markdown."
    ## Build prompt
    prompt = f"<<SYS>>{context}Personality: As {ainame} You are {aipersonality} {instructions}<</SYS>>\n\n%s%s" % (
            args.roleenforcer.replace('{user}', username).replace('{assistant}', ainame),
            args.promptcompletion.replace('{user_question}', question))
    
    return prompt

def main():
    messages = [
        ChatCompletionMessage(
            role="system",
            content=f"<<SYS>>{args.systemprompt}<</SYS>>"
        ),
    ]

    while True:
        # Receive a message
        client_request = receiver.recv_json()

        # fill in variables from client request
        segment_number = client_request["segment_number"]
        mediaid = client_request["mediaid"]
        mediatype = client_request["mediatype"]
        username = client_request["username"]
        source = client_request["source"]
        message = client_request["message"]
        ainame = args.ai_name
        if "ainame" in client_request and client_request["ainame"] != "":
            ainame = client_request["ainame"]
        aipersonality = args.personality
        if "aipersonality" in client_request and client_request["aipersonality"] != "":
            aipersonality = client_request["aipersonality"]
        user_history = []
        # confirm we have a list for history
        if "history" in client_request and isinstance(client_request["history"], list):
            user_history = client_request["history"]

        header_message = {
            "segment_number": segment_number,
            "mediaid": mediaid,
            "mediatype": mediatype,
            "username": username,
            "source": source,
            "message": message,
            "ainame": ainame,
            "aipersonality": aipersonality,
            "context": user_history,
            "tokens": 0,
            "text": ""
        }
        
        print(f"\n---\nLLM: received message:\n - {header_message}\n")
        response = ""

        prompt = create_prompt(header_message)

        header_message["llm_prompt"] = prompt

        llm_output = None
        response = None
        if not args.analysis:
            # Calculate the total length of all messages in history
            total_length = sum([len(msg['content']) for msg in messages])
            # keep history within context size
            while total_length > (args.context/2)+len(prompt):
                # Remove the oldest message after the system prompt
                if len(messages) > 2:
                    total_length -= len(messages[1]['content'])
                    del messages[1]

            messages.append(ChatCompletionMessage(
                role="user",
                content=prompt,
            ))

            header_message = run_llm(header_message.copy(), messages)
            response = header_message["text"]

            messages.append(ChatCompletionMessage(
                role="assistant",
                content=response,
            ))
        else:
            # store llm information in header
            header_message["llm_info"] = {
                "maxtokens": args.maxtokens,
                "temperature": args.temperature,
                "stoptokens": args.stoptokens,
                "characters_per_line": args.characters_per_line,
                "sentence_count": args.sentence_count,
                "simplesplit": args.simplesplit,
                "autogenerate": args.autogenerate,
            }

            llm_output = llm_image(
                prompt,
                max_tokens=args.maxtokens,
                temperature=args.temperature,
                stream=False,
                stop=args.stoptokens.split(',') if args.stoptokens else [],
            )

            # Confirm we have a proper output
            if 'choices' in llm_output and len(llm_output["choices"]) > 0 and 'text' in llm_output["choices"][0]:
                response = llm_output["choices"][0]['text']

            if not response.strip():
                print(f"\nLLM: Reverting to original message, got back empty answer\n - {json.dumps(llm_output)}")
                response = message

            header_message["text"] = response

            sender.send_json(header_message)

            print(f"\nLLM: sent response - {response}.\n")
        
        print(f"\nLLM: finished generating and sending text for job request:\n - {header_message}\n")

if __name__ == "__main__":
    model = "models/zephyr-7b-alpha.Q2_K.gguf"
    prompt_template = "Question: {user_question}\nAnswer: "
    role_enforcer = "Give an Answer the message from {user} listed as a Question at the prompt below. Stay in the role of {assistant} using the Context if listed to help generate a response.\n"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_host", type=str, default="127.0.0.1")
    parser.add_argument("--input_port", type=int, default=1500)
    parser.add_argument("--output_host", type=str, default="127.0.0.1")
    parser.add_argument("--output_port", type=int, default=2000)
    parser.add_argument("--maxtokens", type=int, default=0)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--gpulayers", type=int, default=0)
    parser.add_argument("--model", type=str, default=model)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("--ai_name", type=str, default="GAIB")
    parser.add_argument("--systemprompt", type=str, default="The Groovy AI Bot that is here to help you find enlightenment and learn about technology of the future.")
    parser.add_argument("-e", "--episode", action="store_true", default=False, help="Episode mode, Output an TV Episode format script.")
    parser.add_argument("-pc", "--promptcompletion", type=str, default=prompt_template,
                        help="Prompt completion like...\n\nQuestion: {user_question}\nAnswer:")
    parser.add_argument("-re", "--roleenforcer",
                        type=str, default=role_enforcer,
                        help="Role enforcer statement with {user} and {assistant} template names replaced by the actual ones in use.")
    parser.add_argument("-p", "--personality", type=str, default="friendly helpful compassionate boddisatvva guru.", help="Personality of the AI, choices are 'friendly' or 'mean'.")
    parser.add_argument("-analysis", "--analysis", action="store_true", default=False, help="Instruction mode, no history and focused on solving problems.")
    parser.add_argument("-sts", "--stoptokens", type=str, default="Question:,Answer:,Context:,Episode:,Plotline Description:,Personality:,User:",
        help="Stop tokens to use, do not change unless you know what you are doing!")
    parser.add_argument("-tp", "--characters_per_line", type=int, default=100, help="Minimum umber of characters per buffer, buffer window before output.")
    parser.add_argument("-sc", "--sentence_count", type=int, default=1, help="Number of sentences per line.")
    parser.add_argument("-ag", "--autogenerate", action="store_true", default=False, help="Carry on long conversations, remove stop tokens.")
    parser.add_argument("--simplesplit", action="store_true", default=False, help="Simple split of text into lines, no sentence tokenization.")
    args = parser.parse_args()

    ## setup episode mode
    if args.episode:
        args.roleenforcer.replace('Answer the question asked by', 'Create a story from the plotline given by')
        args.promptcompletion.replace('Answer:', 'Episode:')
        args.promptcompletion.replace('Question:', 'Plotline Description:')

    context = ""
    llm = Llama(model_path=args.model, n_ctx=args.context, verbose=False, n_gpu_layers=args.gpulayers, rope_freq_base=0, rope_freq_scale=0)
    # LLM Model for image prompt generation
    llm_image = Llama(model_path=args.model,
                      n_ctx=args.context, verbose=False, n_gpu_layers=args.gpulayers, rope_freq_base=0, rope_freq_scale=0)

    if args.autogenerate:
        args.stoptokens = []
    
    zmq_context = zmq.Context()

    # Set up the subscriber
    receiver = zmq_context.socket(zmq.PULL)
    print(f"connected to ZMQ in {args.input_host}:{args.input_port}")
    receiver.bind(f"tcp://{args.input_host}:{args.input_port}")
    #receiver.setsockopt_string(zmq.SUBSCRIBE, "")

    # Set up the publisher
    sender = zmq_context.socket(zmq.PUB)
    print(f"binded to ZMQ out {args.output_host}:{args.output_port}")
    sender.bind(f"tcp://{args.output_host}:{args.output_port}")
    
    main()

