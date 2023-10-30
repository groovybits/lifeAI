#!/usr/bin/env python

# Life AI Text to Speech module using a new TTS API
# Adapted from original script by Chris Kennedy 2023 (C) GPL

import zmq
import argparse
import requests
import io
import warnings
import re

# Suppress warnings
warnings.simplefilter(action='ignore', category=Warning)

def get_tts_audio(text, voice=None, noise_scale=None, noise_w=None, length_scale=None, ssml=None, audio_target=None):
    params = {
        'text': text,
        'voice': voice or 'en_US/cmu-arctic_low#slt',
        'noiseScale': noise_scale or '0.667',
        'noiseW': noise_w or '0.8',
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

        print("Text to Speech received request:\n%s" % header_message)

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
            print(f"Exception: ERROR TTS error with API request for text: {text}")
            print(e)
            continue

        audiobuf = io.BytesIO(audio_blob)
        audiobuf.seek(0)

        # Fill in the header
        header_message["duration"] = duration

        # Send the header and the audio
        sender.send_json(header_message, zmq.SNDMORE)
        sender.send(audiobuf.getvalue())

        print(f"Text to Speech: sent audio #{segment_number}\n{header_message}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, default=2000, required=False, help="Port for receiving text input")
    parser.add_argument("--output_port", type=int, default=2001, required=False, help="Port for sending audio output")
    parser.add_argument("--input_host", type=str, default="127.0.0.1", required=False, help="Host for receiving text input")
    parser.add_argument("--output_host", type=str, default="127.0.0.1", required=False, help="Host for sending audio output")
    parser.add_argument("--voice", type=str, default='en_US/cmu-arctic_low#slt', help="Voice parameter for TTS API")
    parser.add_argument("--noise_scale", type=str, default='0.667', help="Noise scale parameter for TTS API")
    parser.add_argument("--noise_w", type=str, default='0.8', help="Noise weight parameter for TTS API")
    parser.add_argument("--length_scale", type=str, default='1', help="Length scale parameter for TTS API")
    parser.add_argument("--ssml", type=str, default='false', help="SSML parameter for TTS API")
    parser.add_argument("--audio_target", type=str, default='client', help="Audio target parameter for TTS API")

    args = parser.parse_args()

    context = zmq.Context()
    receiver = context.socket(zmq.SUB)
    print(f"Connected to ZMQ in: {args.input_host}:{args.input_port}")
    receiver.connect(f"tcp://{args.input_host}:{args.input_port}")
    receiver.setsockopt_string(zmq.SUBSCRIBE, "")

    sender = context.socket(zmq.PUB)
    print(f"Bounded to ZMQ out: {args.output_host}:{args.output_port}")
    sender.bind(f"tcp://{args.output_host}:{args.output_port}")

    main()
