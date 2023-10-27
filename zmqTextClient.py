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
    print("binding to send message: %s:%d" % (args.output_host, args.output_port))
    tti_socket.bind(f"tcp://{args.output_host}:{args.output_port}")

    # Send the message
    tti_socket.send_string(args.segment_number, zmq.SNDMORE)
    tti_socket.send_string(args.message)

    print("Message sent")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_port", type=int, default=1500, required=False, help="Port to send message to")
    parser.add_argument("--output_host", type=str, default="127.0.0.1", required=False, help="Host for sending message to.")
    parser.add_argument("--message", type=str, required=True, help="Message to be sent")
    parser.add_argument("--segment_number", type=str, required=True, help="Segment number")
    args = parser.parse_args()

    main()

