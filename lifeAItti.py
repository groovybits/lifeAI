#!/usr/bin/env python

## Life AI Stable Diffusion module
#
# Chris Kennedy 2023 (C) GPL
#
# Free to use for any use as in truly free software
# as Richard Stallman intended it to be.
#

import zmq
import argparse
from transformers import VitsModel, AutoTokenizer
import io

from diffusers import StableDiffusionPipeline
import torch
from transformers import logging as trlogging
import warnings
import urllib3
import inflect
import re
import logging
import time

warnings.simplefilter(action='ignore', category=Warning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
from urllib3.exceptions import NotOpenSSLWarning
warnings.simplefilter(action='ignore', category=NotOpenSSLWarning)
trlogging.set_verbosity_error()

def clean_text(text):
    p = inflect.engine()

    def num_to_words(match):
        number = match.group()
        try:
            words = p.number_to_words(number)
        except inflect.NumOutOfRangeError:
            words = "[number too large]"
        return words

    text = re.sub(r'\b\d+(\.\d+)?\b', num_to_words, text)

    # Add a pause after punctuation
    text = text.replace('.', '. ')
    text = text.replace(',', ', ')
    text = text.replace('?', '? ')
    text = text.replace('!', '! ')

    return text[:300]

def main():
    last_image = None
    last_image_timestamp = 0
    last_image_walltime = 0
    text_cache = []
    while True:
        """ 
          header_message = {
            "segment_number": segment_number,
            "mediaid": mediaid,
            "mediatype": mediatype,
            "username": username,
            "source": source,
            "message": message,
            "text": text,
            "optimized_text": optimized_text,
        }"""
        # Receive a message
        header_message = receiver.recv_json()

        # get variables from header
        segment_number = header_message["segment_number"]
        optimized_prompt = ""
        if "optimized_text" in header_message:
            optimized_prompt = header_message["optimized_text"]
        else:
            optimized_prompt = header_message["text"]
            logger.error(f"TTI: No optimized text, using original text.")

        logger.debug(f"Text to Image recieved optimized prompt:\n{header_message}.")

        image = None
        if last_image == None or time.time() - header_message["timestamp"] < args.latency:
            # 2. Forward embeddings and negative embeddings through text encoder
            max_length = pipe.tokenizer.model_max_length

            # 3. Forward
            input_ids = pipe.tokenizer(optimized_prompt, return_tensors="pt").input_ids
            input_ids = input_ids.to("mps")

            negative_ids = pipe.tokenizer("", truncation=False, padding="max_length", max_length=input_ids.shape[-1], return_tensors="pt").input_ids                                                                                                     
            negative_ids = negative_ids.to("mps")

            concat_embeds = []
            neg_embeds = []
            for i in range(0, input_ids.shape[-1], max_length):
                concat_embeds.append(pipe.text_encoder(input_ids[:, i: i + max_length])[0])
                neg_embeds.append(pipe.text_encoder(negative_ids[:, i: i + max_length])[0])

            prompt_embeds = torch.cat(concat_embeds, dim=1)
            negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

            image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds).images[0]

            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')  # Save it as PNG or JPEG depending on your preference
            image = img_byte_arr.getvalue()

            last_image = image

        header_message["stream"] = "image"

        sender.send_json(header_message, zmq.SNDMORE)
        sender.send(last_image)

        logger.info(f"Text to Image sent image #{segment_number}:\n - {optimized_prompt}")
        last_image_walltime = time.time()
        last_image_timestamp = header_message["timestamp"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, default=3001, required=False, help="Port for receiving text input")
    parser.add_argument("--output_port", type=int, default=3002, required=False, help="Port for sending image output")
    parser.add_argument("--input_host", type=str, default="127.0.0.1", required=False, help="Port for receiving text input")
    parser.add_argument("--output_host", type=str, default="127.0.0.1", required=False, help="Port for sending image output")
    parser.add_argument("--nsfw", action="store_true", default=False, help="Disable NSFW filters, caution!!!")
    parser.add_argument("--metal", action="store_true", default=False, help="offload to metal mps GPU")
    parser.add_argument("--cuda", action="store_true", default=False, help="offload to metal cuda GPU")
    parser.add_argument("-ll", "--loglevel", type=str, default="info", help="Logging level: debug, info...")
    parser.add_argument("-m", "--model", type=str, default="runwayml/stable-diffusion-v1-5", help="Model ID to use")
    parser.add_argument("--latency", type=int, default=60, help="Latency in seconds to wait for a message")
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
    logging.basicConfig(filename=f"logs/tti-{log_id}.log", level=LOGLEVEL)
    logger = logging.getLogger('GAIB')

    ch = logging.StreamHandler()
    ch.setLevel(LOGLEVEL)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    model_id = args.model

    ## Disable NSFW filters
    pipe = None
    if args.nsfw:
        pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                                        torch_dtype=torch.float16,
                                                        safety_checker = None,
                                                        requires_safety_checker = False)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

    ## Offload to GPU Metal
    if args.metal:
        pipe = pipe.to("mps")
    elif args.cuda:
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("mps")

    context = zmq.Context()
    receiver = context.socket(zmq.SUB)
    logger.info("connected to ZMQ in: %s:%d" % (args.input_host, args.input_port))
    receiver.connect(f"tcp://{args.input_host}:{args.input_port}")
    receiver.setsockopt_string(zmq.SUBSCRIBE, "")

    sender = context.socket(zmq.PUB)
    logger.info("binded to ZMQ out: %s:%d" % (args.output_host, args.output_port))
    sender.bind(f"tcp://{args.output_host}:{args.output_port}")

    main()

