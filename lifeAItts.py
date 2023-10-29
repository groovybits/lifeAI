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

def clean_text_for_tts(text):
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

    return text


def main():
    while True:
        segment_number = receiver.recv_string()
        mediaid = receiver.recv_string()
        mediatype = receiver.recv_string()
        username = receiver.recv_string()
        source = receiver.recv_string()
        message = receiver.recv_string()
        text = receiver.recv_string()

        print("Text to Speech recieved text #%s: %s" % (segment_number, text))

        inputs = tokenizer(clean_text_for_tts(text), return_tensors="pt")
        inputs['input_ids'] = inputs['input_ids'].long()

        output = None
        try:
            with torch.no_grad():
                output = model(**inputs).waveform
            waveform_np = output.squeeze().numpy().T
        except Exception as e:
            print(f"Exception: ERROR STT error with output.squeeze().numpy().T on audio: {text}")
            continue
        audiobuf = io.BytesIO()
        sf.write(audiobuf, waveform_np, model.config.sampling_rate, format='WAV')
        audiobuf.seek(0)

        duration = len(waveform_np) / model.config.sampling_rate
        sender.send_string(str(segment_number), zmq.SNDMORE)
        sender.send_string(mediaid, zmq.SNDMORE)
        sender.send_string(mediatype, zmq.SNDMORE)
        sender.send_string(username, zmq.SNDMORE)
        sender.send_string(source, zmq.SNDMORE)
        sender.send_string(message, zmq.SNDMORE)
        sender.send_string(text, zmq.SNDMORE)
        sender.send_string(str(duration), zmq.SNDMORE)
        sender.send(audiobuf.getvalue())
        
        print("Text to Speech: sent audio #%s" % segment_number)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, default=2000, required=False, help="Port for receiving text input")
    parser.add_argument("--output_port", type=int, default=2001, required=False, help="Port for sending audio output")
    parser.add_argument("--target_lang", type=str, default="eng", help="Target language")
    parser.add_argument("--source_lang", type=str, default="eng", help="Source language")
    parser.add_argument("--audio_format", choices=["wav", "raw"], default="raw", help="Audio format to save as. Choices are 'wav' or 'raw'.")
    parser.add_argument("--input_host", type=str, default="127.0.0.1", required=False, help="Port for receiving text input")
    parser.add_argument("--output_host", type=str, default="127.0.0.1", required=False, help="Port for sending audio output")
    parser.add_argument("--gputype", type=str, default="cpu", required=False, help="GPU type to use. Default is cpu")

    args = parser.parse_args()

    context = zmq.Context()
    receiver = context.socket(zmq.SUB)
    print("connected to ZMQ in: %s:%d" % (args.input_host, args.input_port))
    receiver.connect(f"tcp://{args.input_host}:{args.input_port}")
    receiver.setsockopt_string(zmq.SUBSCRIBE, "")

    sender = context.socket(zmq.PUB)
    print("binded to ZMQ out: %s:%d" % (args.output_host, args.output_port))
    sender.bind(f"tcp://{args.output_host}:{args.output_port}")

    model = VitsModel.from_pretrained("facebook/mms-tts-eng")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

    model.to(args.gputype)

    main()

