#!/usr/bin/env python

# Life AI Text to Speech module using a new TTS API
# Adapted from original script by Chris Kennedy 2023 (C) GPL

import zmq
import argparse
import requests
import io
import warnings
import re
import logging
import time

# Suppress warnings
warnings.simplefilter(action='ignore', category=Warning)

def get_tts_audio(text, voice=None, noise_scale=None, noise_w=None, length_scale=None, ssml=None, audio_target=None):
    params = {
        'text': text,
        'voice': voice or 'en_US/cmu-arctic_low#slt',
        'noiseScale': noise_scale or '0.333',
        'noiseW': noise_w or '0.333',
        'lengthScale': length_scale or '1',
        'ssml': ssml or 'false',
        'audioTarget': audio_target or 'client'
    }

    response = requests.get('http://earth:59125/api/tts', params=params)
    response.raise_for_status()
    return response.content


def main():
    while True:
        header_message = receiver.recv_json()
        segment_number = header_message["segment_number"]
        text = header_message["text"]

        # remove new lines
        #text = text.replace('\n', ' ')
        # reduce multiple spaces to single space
        #text = re.sub(r'\s+', ' ', text)

         # clean text of end of line spaces after punctuation
        text = re.sub(r'([.,!?;:])\s+', r'\1', text)

        # add ssml tags
        if args.ssml == 'true':
            text = f"<speak><prosody pitch=\"{args.pitch}\" range=\"{args.range}\" rate=\"{args.rate}\">" + text + f"</prosody></speak>"

        logger.debug("Text to Speech received request:\n%s" % header_message)
        logger.info(f"Text to Speech received request #{segment_number}:\n{text}")

        try:
            audio_blob = get_tts_audio(
                text,
                voice=args.voice,
                noise_scale=args.noise_scale,
                noise_w=args.noise_w,
                length_scale=args.length_scale,
                ssml=args.ssml,
                audio_target=args.audio_target
            )
            duration = len(audio_blob) / (22050 * 2)  # Assuming 22.5kHz 16-bit audio for duration calculation
        except Exception as e:
            logger.error(f"Exception: ERROR TTS error with API request for text: {text}")
            logger.error(e)
            continue

        audiobuf = io.BytesIO(audio_blob)
        audiobuf.seek(0)

        # Fill in the header
        header_message["duration"] = duration
        header_message["stream"] = "speech"

        # Send the header and the audio
        sender.send_json(header_message, zmq.SNDMORE)
        sender.send(audiobuf.getvalue())

        logger.debug(f"Text to Speech: sent audio #{segment_number}\n{header_message}")
        logger.info(f"Text to Speech: sent audio #{segment_number} of {duration} duration.\n{text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, default=2000, required=False, help="Port for receiving text input")
    parser.add_argument("--output_port", type=int, default=6002, required=False, help="Port for sending audio output")
    parser.add_argument("--input_host", type=str, default="127.0.0.1", required=False, help="Host for receiving text input")
    parser.add_argument("--output_host", type=str, default="127.0.0.1", required=False, help="Host for sending audio output")
    parser.add_argument("--voice", type=str, default='en_US/cmu-arctic_low#ljm', help="Voice parameter for TTS API")
    parser.add_argument("--noise_scale", type=str, default='0.333', help="Noise scale parameter for TTS API")
    parser.add_argument("--noise_w", type=str, default='0.333', help="Noise weight parameter for TTS API")
    parser.add_argument("--length_scale", type=str, default='1.5', help="Length scale parameter for TTS API")
    parser.add_argument("--ssml", type=str, default='false', help="SSML parameter for TTS API")
    parser.add_argument("--audio_target", type=str, default='client', help="Audio target parameter for TTS API")
    parser.add_argument("-ll", "--loglevel", type=str, default="info", help="Logging level: debug, info...")
    parser.add_argument("--sub", action="store_true", default=False, help="Publish to a topic")
    parser.add_argument("--pub", action="store_true", default=False, help="Publish to a topic")
    parser.add_argument("--bind_output", action="store_true", default=False, help="Bind to a topic")
    parser.add_argument("--bind_input", action="store_true", default=False, help="Bind to a topic")
    parser.add_argument("--rate", type=str, default="default", help="Speech rate, slow, medium, fast")
    parser.add_argument("--range", type=str, default="high", help="Speech range, low, medium, high")
    parser.add_argument("--pitch", type=str, default="high", help="Speech pitch, low, medium, high")
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
    logging.basicConfig(filename=f"logs/ttsMimic3-{log_id}.log", level=LOGLEVEL)
    logger = logging.getLogger('GAIB')

    ch = logging.StreamHandler()
    ch.setLevel(LOGLEVEL)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    context = zmq.Context()
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
    sender = context.socket(zmq.PUSH)
    print(f"binded to ZMQ out {args.output_host}:{args.output_port}")
    sender.connect(f"tcp://{args.output_host}:{args.output_port}")

    main()
