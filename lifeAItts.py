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
from transformers import VitsModel, AutoTokenizer
import torch
import io
import soundfile as sf
from transformers import logging as trlogging
import warnings
import urllib3
import inflect
import re

warnings.simplefilter(action='ignore', category=Warning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
from urllib3.exceptions import NotOpenSSLWarning
warnings.simplefilter(action='ignore', category=NotOpenSSLWarning)
trlogging.set_verbosity_error()

model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

def convert_numbers_to_words(text):
    p = inflect.engine()

    def num_to_words(match):
        number = match.group()
        if '.' in number:
            parts = number.split('.')
            words = f"{p.number_to_words(parts[0])} point {p.number_to_words(parts[1])}"
        else:
            words = p.number_to_words(number)
        return words

    text_with_words = re.sub(r'\b\d+(\.\d+)?\b', num_to_words, text)
    return text_with_words

def clean_text_for_tts(text):
    # Convert numbers to words
    p = inflect.engine()
    text = re.sub(r'\b\d+(\.\d+)?\b', lambda match: p.number_to_words(match.group()), text)

    convert_numbers_to_words(text)

    # Add a pause after punctuation
    text = text.replace('.', '. ')
    text = text.replace(',', ', ')
    text = text.replace('?', '? ')
    text = text.replace('!', '! ')

    return text

def main():
    context = zmq.Context()
    receiver = context.socket(zmq.SUB)
    print("connected to ZMQ in: %s:%d" % (args.input_host, args.input_port))
    receiver.connect(f"tcp://{args.input_host}:{args.input_port}")
    receiver.setsockopt_string(zmq.SUBSCRIBE, "")

    sender = context.socket(zmq.PUSH)
    print("binded to ZMQ out: %s:%d" % (args.output_host, args.output_port))
    sender.bind(f"tcp://{args.output_host}:{args.output_port}")

    while True:
        try:
            segment_number = receiver.recv_string()
            text = receiver.recv_string()

            inputs = tokenizer(clean_text_for_tts(text), return_tensors="pt")
            inputs['input_ids'] = inputs['input_ids'].long()

            output = None
            with torch.no_grad():
                output = model(**inputs).waveform
            waveform_np = output.squeeze().numpy().T
            audiobuf = io.BytesIO()
            sf.write(audiobuf, waveform_np, model.config.sampling_rate, format='WAV')
            audiobuf.seek(0)

            duration = len(waveform_np) / model.config.sampling_rate
            sender.send_string(str(segment_number), zmq.SNDMORE)
            sender.send_string(text, zmq.SNDMORE)
            sender.send_string(str(duration), zmq.SNDMORE)
            sender.send(audiobuf.getvalue())
            
            print("Text to Speech: sent audio #%s" % segment_number)
        except Exception as e:
            print("Error in lifeAItts: %s" % str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, default=2000, required=False, help="Port for receiving text input")
    parser.add_argument("--output_port", type=int, default=2001, required=False, help="Port for sending audio output")
    parser.add_argument("--target_lang", type=str, default="eng", help="Target language")
    parser.add_argument("--source_lang", type=str, default="eng", help="Source language")
    parser.add_argument("--audio_format", choices=["wav", "raw"], default="raw", help="Audio format to save as. Choices are 'wav' or 'raw'.")
    parser.add_argument("--input_host", type=str, default="127.0.0.1", required=False, help="Port for receiving text input")
    parser.add_argument("--output_host", type=str, default="127.0.0.1", required=False, help="Port for sending audio output")

    args = parser.parse_args()
    main()

