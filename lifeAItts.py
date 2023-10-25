#!/usr/bin/env python

## Life AI Text to Speech module
#
# Chris Kennedy 2023 (C) GPL
#
# Free to use for any use as in truly free software
# as Richard Stallman intended it to be.
#

import zmq
import argparse
import numpy as np
from transformers import AutoProcessor, SeamlessM4TModel
import wave
import textwrap
from transformers import VitsModel, AutoTokenizer
import torch
import scipy
import io
import soundfile as sf

model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")


# Initialize the model and processor
#processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
#model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium")

def save_as_wav(data, output_file, sample_rate=model.config.sampling_rate, channels=1, width=2):
    with wave.open(output_file, 'w') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(data)

def main(input_port, output_port):
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind(f"tcp://*:{input_port}")

    sender = context.socket(zmq.PUSH)
    sender.bind(f"tcp://*:{output_port}")

    segment_number = 1

    while True:
        message = receiver.recv_string()
        _, text = message.split(":", 1)

        #text_inputs = processor(text=text, src_lang=args.source_lang, return_tensors="pt")
        #audio_output = model.generate(**text_inputs, tgt_lang=args.target_lang, generate_speech=True)
        #audio_numpy = audio_output[0].cpu().numpy().squeeze().tobytes()
        inputs = tokenizer(text, return_tensors="pt")
        inputs['input_ids'] = inputs['input_ids'].long()

        output = None
        with torch.no_grad():
            output = model(**inputs).waveform
        waveform_np = output.squeeze().numpy().T
        audiobuf = io.BytesIO()
        sf.write(audiobuf, waveform_np, model.config.sampling_rate, format='WAV')
        audiobuf.seek(0)

        if args.output_file:
            if args.audio_format == "wav":
                with open(args.output_file, 'wb') as f:
                    f.write(audiobuf.getvalue())
                print(f"Audio saved to {args.output_file} as WAV\n")
            else:
                with open(args.output_file, 'wb') as f:
                    f.write(output)
                print(f"Payload written to {args.output_file}\n")
        else:
            payload_hex = output.hex()
            print(f"Payload (Hex): {textwrap.fill(payload_hex, width=80)}\n")

        sender.send_string(str(segment_number), zmq.SNDMORE)
        sender.send(waveform_np)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, required=True, help="Port for receiving text input")
    parser.add_argument("--output_port", type=int, required=True, help="Port for sending audio output")
    parser.add_argument("--target_lang", type=str, default="eng", help="Target language")
    parser.add_argument("--source_lang", type=str, default="eng", help="Source language")
    parser.add_argument("--output_file", type=str, default="", help="Output payloads to a file for analysis")
    parser.add_argument("--audio_format", choices=["wav", "raw"], default="raw", help="Audio format to save as. Choices are 'wav' or 'raw'.")

    args = parser.parse_args()
    main(args.input_port, args.output_port)

