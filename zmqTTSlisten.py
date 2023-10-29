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
            # Receive the segment number (header) first
            segment_number = socket.recv_string()
            medaiid = socket.recv_string()
            mediatype = socket.recv_string()
            username = socket.recv_string()
            source = socket.recv_string()
            message = socket.recv_string()
            audio_text = socket.recv_string()
            duration = socket.recv_string()

            # Now, receive the binary audio data
            audio_samples = socket.recv()

            print(f"Received audio segment #%s of {duration} duration." % segment_number)

            # Check if we need to output to a file
            if args.save_file:
                audio_file = f"{args.output_directory}/{mediaid}/{segment_number}.wav"
                ## create directory recrusively for the file if it doesn't exist
                os.makedirs(os.path.dirname(audio_file), exist_ok=True)
                if args.audio_format == "wav":
                    with open(audio_file, 'wb') as f:
                        f.write(audio_samples)
                    print(f"Audio saved to {audio_file} as WAV")
                else:
                    with open(audio_file, 'wb') as f:
                        f.write(audio_samples)
                    print(f"Payload written to {audio_file}\n")

            # Convert the payload to its hex representation and display
            if args.show_hex:
                payload_hex = audio_samples.hex()
                print(f"Payload (Hex): {textwrap.fill(payload_hex, width=80)}\n")

            play_audio(audio_samples)

            print(f"Audio #{segment_number} of {duration} duration recieved.\nAudio Text: {audio_text}\nMessage: {message}\n")

        except Exception as e:
            print(f"Error: %s" % str(e))
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, default=2001, required=False, help="Port for receiving audio numpy arrays")
    parser.add_argument("--input_host", type=str, default="127.0.0.1", required=False, help="Host for receiving audio input")
    parser.add_argument("--output_directory", default="audio", type=str, help="Directory path to save the received wave files in")
    parser.add_argument("--save_file", action="store_true", help="Save the received audio as WAV files")
    parser.add_argument("--show_hex", action="store_true", help="Show the hex representation of the audio payload")
    parser.add_argument("--audio_format", type=str, choices=["wav", "raw"], default="wav", help="Audio format to save as. Choices are 'wav' or 'raw'. Default is 'wav'.")
    args = parser.parse_args()

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    print("connected to ZMQ in: %s:%d" % (args.input_host, args.input_port))
    socket.connect(f"tcp://{args.input_host}:{args.input_port}")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    main()

