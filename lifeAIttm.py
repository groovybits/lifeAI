#!/usr/bin/env python

## Life AI Text to Music module
#
# Chris Kennedy 2023 (C) GPL
#
# Free to use for any use as in truly free software
# as Richard Stallman intended it to be.
#

import zmq
import argparse
import io
import soundfile as sf
from transformers import logging as trlogging
from transformers import AutoProcessor, MusicgenForConditionalGeneration, set_seed
import torch
import logging
import time
import numpy as np

trlogging.set_verbosity_error()

def generate_audio(prompt, negative_prompt, guidance_scale=3, audio_length_in_s=10, seed=0):
    inputs = processor(
        text=[prompt, negative_prompt],
        padding=True,
        return_tensors="pt",
        ).to("cpu")

    with torch.no_grad():
        encoder_outputs = text_encoder(**inputs)

    max_new_tokens = int(frame_rate * audio_length_in_s)

    set_seed(seed)
    audio_values = model.generate(inputs.input_ids[0][None, :], attention_mask=inputs.attention_mask, encoder_outputs=encoder_outputs, do_sample=True, guidance_scale=guidance_scale, max_new_tokens=max_new_tokens)

    audio_values = (audio_values.cpu().numpy() * 32767).astype(np.int16)
    return (sampling_rate, audio_values)

def main():
    latency = 0
    max_latency = args.max_latency
    throttle = False
    while True:
        messages_buffered = ""
        if throttle:
            start = time.time()
            combine_time = latency / 1000

            # read and combine the messages for 60 seconds into a single message
            while time.time() - start < combine_time:
                message = receiver.recv_json()
                if 'text' in message:
                    messages_buffered += message["text"] + " "
                elif 'optimized_prompt' in message:
                    messages_buffered += message["optimized_prompt"] + " "
            logger.info(f"TTM: Throttling for {combine_time} seconds.")

        # read the message
        header_message = receiver.recv_json()
       
        # fill in the variables form the header_message
        optimized_prompt = ""
        if "optimized_prompt" in header_message:
            optimized_prompt = header_message["optimized_prompt"]
        else:
            optimized_prompt = header_message["text"]
            logger.error(f"TTM: No optimized prompt, using original text.")

        optimized_prompt = f"{messages_buffered} {optimized_prompt}"

        logger.debug(f"Text to Music Recieved:\n{header_message}")
        logger.info(f"Text to Music Recieved:\n{optimized_prompt}")

        prompt = f"music like {args.genre} {optimized_prompt}"

        sampling_rate, audio_values = generate_audio(prompt, 
                                                     "noise, static, banging and clanging",
                                                     args.guidance_scale,
                                                     args.seconds,
                                                     args.seed)

        # This is assuming audio_values is meant to be mono; if it's stereo, it should be shaped to (frames, 2)
        audio_values = audio_values.squeeze()  # This will convert (1, 1, 318080) to (318080,)

        audiobuf = io.BytesIO()
        sf.write(audiobuf, audio_values, sampling_rate, format='WAV')
        audiobuf.seek(0)

        duration = len(audio_values) / sampling_rate
        header_message["duration"] = duration
        header_message["stream"] = "music"
        sender.send_json(header_message, zmq.SNDMORE)
        sender.send(audiobuf.getvalue())

        # measure latency and see if we need to throttle output
        latency = round(time.time() * 1000) - header_message['timestamp']
        if latency > (max_latency * 1000):
            logger.error(f"TTM: Message is too old {latency/1000}, throttling for the next 60 seconds.")
            throttle = True
        
        logger.debug(f"Text to Music Sent:\n{header_message}")
        logger.info(f"Text to Music of {duration} duration Sent:\n{optimized_prompt}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, default=4001, required=False, help="Port for receiving text input")
    parser.add_argument("--output_port", type=int, default=6002, required=False, help="Port for sending audio output")
    parser.add_argument("--target_lang", type=str, default="eng", help="Target language")
    parser.add_argument("--source_lang", type=str, default="eng", help="Source language")
    parser.add_argument("--audio_format", choices=["wav", "raw"], default="raw", help="Audio format to save as. Choices are 'wav' or 'raw'.")
    parser.add_argument("--input_host", type=str, default="127.0.0.1", required=False, help="Port for receiving text input")
    parser.add_argument("--output_host", type=str, default="127.0.0.1", required=False, help="Port for sending audio output")
    parser.add_argument("--model", type=str, required=False, default="facebook/musicgen-small", help="Text to music model to use")
    parser.add_argument("--seconds", type=int, default=10, required=False, help="Seconds to create, default is 30")
    parser.add_argument("--metal", action="store_true", default=False, help="offload to metal mps GPU")
    parser.add_argument("--cuda", action="store_true", default=False, help="offload to metal cuda GPU")
    parser.add_argument("-ll", "--loglevel", type=str, default="info", help="Logging level: debug, info...")
    parser.add_argument("--guidance_scale", type=float, default=3.0, help="Guidance scale for the model")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the model")
    parser.add_argument("--genre", type=str, default="Groovy 70s soul music", help="Genre for the model")
    parser.add_argument("--max_latency", type=int, default=10, help="Max latency for messages before they are throttled / combined")
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
    logging.basicConfig(filename=f"logs/ttm-{log_id}.log", level=LOGLEVEL)
    logger = logging.getLogger('TTM')

    ch = logging.StreamHandler()
    ch.setLevel(LOGLEVEL)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    context = zmq.Context()
    receiver = context.socket(zmq.SUB)
    logger.info("connected to ZMQ in: %s:%d" % (args.input_host, args.input_port))
    receiver.connect(f"tcp://{args.input_host}:{args.input_port}")
    receiver.setsockopt_string(zmq.SUBSCRIBE, "")

    # Set up the publisher
    sender = context.socket(zmq.PUSH)
    sender.connect(f"tcp://{args.output_host}:{args.output_port}")
    logger.info("connected to ZMQ out: %s:%d" % (args.output_host, args.output_port))

    processor = AutoProcessor.from_pretrained(args.model)
    model = MusicgenForConditionalGeneration.from_pretrained(args.model)

    sampling_rate = model.audio_encoder.config.sampling_rate
    frame_rate = model.audio_encoder.config.frame_rate
    text_encoder = model.get_text_encoder()

    if args.metal:
        model = model.to("mps")
    elif args.cuda:
        model = model.to("cuda")
    else:
        model = model.to("cpu")

    main()

