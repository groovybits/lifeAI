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
import wave

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
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    print("connected to ZMQ in: %s:%d" % (args.input_host, args.input_port))
    socket.connect(f"tcp://{args.input_host}:{args.input_port}")
    #socket.setsockopt_string(zmq.SUBSCRIBE, "")

    while True:
        try:
            # Receive the segment number (header) first
            header_str = socket.recv_string()
            audio_text = socket.recv_string()
            duration = socket.recv_string()

            # Now, receive the binary audio data
            audio_samples = socket.recv()

            print(f"Received audio segment #%s of {duration} duration." % header_str)

            # Check if we need to output to a file
            if args.output_file:
                if args.audio_format == "wav":
                    with open(args.output_file, 'wb') as f:
                        f.write(audio_samples)
                    print(f"Audio saved to {args.output_file} as WAV")
                else:
                    with open(args.output_file, 'wb') as f:
                        f.write(audio_samples)
                    print(f"Payload written to {args.output_file}\n")

            # Convert the payload to its hex representation and display
            payload_hex = audio_samples.hex()
            print(f"Payload (Hex): {textwrap.fill(payload_hex, width=80)}\n")

            play_audio(audio_samples)

            print(f"Audio #{header_str} of {duration} duration recieved.\nAudio Text: {audio_text}\n")

        except Exception as e:
            print(f"Error: %s" % str(e))
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, default=2001, required=False, help="Port for receiving audio numpy arrays")
    parser.add_argument("--input_host", type=str, default="127.0.0.1", required=False, help="Host for receiving audio input")
    parser.add_argument("--output_file", type=str, help="Path to save the received audio")
    parser.add_argument("--audio_format", type=str, choices=["wav", "raw"], default="wav", help="Audio format to save as. Choices are 'wav' or 'raw'. Default is 'wav'.")
    args = parser.parse_args()

    main()

