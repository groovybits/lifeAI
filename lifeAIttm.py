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
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import warnings
import urllib3

warnings.simplefilter(action='ignore', category=Warning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
from urllib3.exceptions import NotOpenSSLWarning
warnings.simplefilter(action='ignore', category=NotOpenSSLWarning)
trlogging.set_verbosity_error()

def main():
    while True:
        segment_number = receiver.recv_string()
        mediaid = receiver.recv_string()
        mediatype = receiver.recv_string()
        username = receiver.recv_string()
        source = receiver.recv_string()
        message = receiver.recv_string()
        text = receiver.recv_string()
        optimized_prompt = receiver.recv_string()

        print("\n---\nText to Music recieved text #%s: %s" % (segment_number, optimized_prompt))

        inputs = processor(
            text=[optimized_prompt],
            padding=True,
            return_tensors="pt",
        )

        audio_values = model.generate(**inputs, max_new_tokens=256)
        audio_values = audio_values.numpy().reshape(-1)

        audiobuf = io.BytesIO()
        sf.write(audiobuf, audio_values, model.config.sampling_rate, format='WAV')
        audiobuf.seek(0)

        duration = len(audio_values) / model.config.sampling_rate
        sender.send_string(str(segment_number), zmq.SNDMORE)
        sender.send_string(mediaid, zmq.SNDMORE)
        sender.send_string(mediatype, zmq.SNDMORE)
        sender.send_string(username, zmq.SNDMORE)
        sender.send_string(source, zmq.SNDMORE)
        sender.send_string(message, zmq.SNDMORE)
        sender.send_string(text, zmq.SNDMORE)
        sender.send_string(optimized_prompt, zmq.SNDMORE)
        sender.send_string(str(duration), zmq.SNDMORE)
        sender.send(audiobuf.getvalue())
        
        print("\nText to Music: sent audio #%s" % segment_number)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, default=4001, required=False, help="Port for receiving text input")
    parser.add_argument("--output_port", type=int, default=4002, required=False, help="Port for sending audio output")
    parser.add_argument("--target_lang", type=str, default="eng", help="Target language")
    parser.add_argument("--source_lang", type=str, default="eng", help="Source language")
    parser.add_argument("--audio_format", choices=["wav", "raw"], default="raw", help="Audio format to save as. Choices are 'wav' or 'raw'.")
    parser.add_argument("--input_host", type=str, default="127.0.0.1", required=False, help="Port for receiving text input")
    parser.add_argument("--output_host", type=str, default="127.0.0.1", required=False, help="Port for sending audio output")
    parser.add_argument("--duration", type=int, default=10, help="Duration of the audio in seconds")
    parser.add_argument("--model", type=str, required=False, default="facebook/musicgen-small", help="Text to music model to use")
    parser.add_argument("--gpu", type=str, default="cpu", required=False, help="GPU type, cpu, cuda or mps")
    parser.add_argument("--seconds", type=int, default=30, required=False, help="Seconds to create, default is 30")

    args = parser.parse_args()

    context = zmq.Context()
    receiver = context.socket(zmq.SUB)
    print("connected to ZMQ in: %s:%d" % (args.input_host, args.input_port))
    receiver.connect(f"tcp://{args.input_host}:{args.input_port}")
    receiver.setsockopt_string(zmq.SUBSCRIBE, "")

    sender = context.socket(zmq.PUB)
    print("binded to ZMQ out: %s:%d" % (args.output_host, args.output_port))
    sender.bind(f"tcp://{args.output_host}:{args.output_port}")

    """
    synthesiser = pipeline("text-to-audio", args.model)
    music = synthesiser("lo-fi music with a soothing melody", forward_params={"do_sample": True})
    scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], music=audio["audio"])
    """

    processor = AutoProcessor.from_pretrained(args.model)
    model = MusicgenForConditionalGeneration.from_pretrained(args.model)
    """
    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=args.duration
    )
    """

    model = model.to(args.gpu)

    main()

