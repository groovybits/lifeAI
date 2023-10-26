#!/usr/bin/env python

## Life AI Text to Speech test for ZMQ module
#
# Chris Kennedy 2023 (C) GPL
#
# Free to use for any use as in truly free software
# as Richard Stallman intended it to be.
#

import zmq
import argparse

def main():
    context = zmq.Context()

    # Socket to send messages on
    tti_socket = context.socket(zmq.PUSH)
    print("binding TTI to ports in: %s:%d" % (args.tti_output_host, args.tti_output_port))
    tti_socket.bind(f"tcp://{args.tti_output_host}:{args.tti_output_port}")

    # Send the message
    tti_socket.send_string(args.segment_number, zmq.SNDMORE)
    tti_socket.send_string(args.message)

    # Socket to send messages on
    tts_socket = context.socket(zmq.PUSH)
    print("binding TTS to ports in: %s:%d" % (args.tts_output_host, args.tts_output_port))
    tts_socket.bind(f"tcp://{args.tts_output_host}:{args.tts_output_port}")

    # Send the message
    tts_socket.send_string(args.segment_number, zmq.SNDMORE)
    tts_socket.send_string(args.message)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tti_output_port", type=int, default=2000, required=False, help="Port to send TTI message to")
    parser.add_argument("--tti_output_host", type=str, default="127.0.0.1", required=False, help="Host for sending TTI text to.")
    parser.add_argument("--tts_output_port", type=int, default=3000, required=False, help="Port to send TTS message to")
    parser.add_argument("--tts_output_host", type=str, default="127.0.0.1", required=False, help="Host for sending TTS text to.")
    parser.add_argument("--message", type=str, required=True, help="Message to be sent")
    parser.add_argument("--segment_number", type=str, required=True, help="Segment number")
    args = parser.parse_args()

    main()

