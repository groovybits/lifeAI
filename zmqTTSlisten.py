#!/usr/bin/env python

## Life AI Text to Speech listener ZMQ client
#
# Chris Kennedy 2023 (C) GPL
#
# Free to use for any use as in truly free software
# as Richard Stallman intended it to be.
#

import io
import zmq
import argparse
import textwrap
import soundfile as sf
import pygame
import re
import os
import sys
import logging
import time

def play_audio(audio_samples):
    pygame.mixer.init(frequency=16000, size=-16, channels=1, buffer=1024)
    pygame.init()
     
    audiobuf = io.BytesIO(audio_samples)
    if audiobuf:
        ## Speak WAV TTS Output using pygame
        pygame.mixer.music.load(audiobuf)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

def main():
    while True:
        try:
            """ From LLM Source
            header_message = {
            "segment_number": segment_number,
            "mediaid": mediaid,
            "mediatype": mediatype,
            "username": username,
            "source": source,
            "message": message,
            "text": text,
            "duration": duration,
            }
            audio_blob = io.BytesIO()
            """
            header_message = socket.recv_json()
            # get variable from header message
            segment_number = header_message['segment_number']
            mediaid = header_message['mediaid']
           
            # Now, receive the binary audio data
            audio_samples = socket.recv()

            logger.debug(f"Received audio segment {header_message}\n")

            # Check if we need to output to a file
            if args.save_file:
                audio_file = f"{args.output_directory}/{mediaid}/{segment_number}.wav"
                ## create directory recrusively for the file if it doesn't exist
                os.makedirs(os.path.dirname(audio_file), exist_ok=True)
                if args.audio_format == "wav":
                    with open(audio_file, 'wb') as f:
                        f.write(audio_samples)
                    logger.info(f"Audio saved to {audio_file} as WAV")
                else:
                    with open(audio_file, 'wb') as f:
                        f.write(audio_samples)
                    logger.info(f"Payload written to {audio_file}\n")

            # Convert the payload to its hex representation and display
            if args.show_hex:
                payload_hex = audio_samples.hex()
                print(f"Payload (Hex): {textwrap.fill(payload_hex, width=80)}\n", flush=True)

            play_audio(audio_samples)

        except Exception as e:
            logger.error(f"Error: %s" % str(e))
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, default=2001, required=False, help="Port for receiving audio numpy arrays")
    parser.add_argument("--input_host", type=str, default="127.0.0.1", required=False, help="Host for receiving audio input")
    parser.add_argument("--output_directory", default="audio", type=str, help="Directory path to save the received wave files in")
    parser.add_argument("--save_file", action="store_true", help="Save the received audio as WAV files")
    parser.add_argument("--show_hex", action="store_true", help="Show the hex representation of the audio payload")
    parser.add_argument("--audio_format", type=str, choices=["wav", "raw"], default="wav", help="Audio format to save as. Choices are 'wav' or 'raw'. Default is 'wav'.")
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
    logging.basicConfig(filename=f"logs/zmqTTSlisten-{log_id}.log", level=LOGLEVEL)
    logger = logging.getLogger('GAIB')

    ch = logging.StreamHandler()
    ch.setLevel(LOGLEVEL)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    logger.info("connected to ZMQ in: %s:%d" % (args.input_host, args.input_port))
    socket.connect(f"tcp://{args.input_host}:{args.input_port}")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    main()

