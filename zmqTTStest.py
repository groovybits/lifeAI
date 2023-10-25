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

def main(target_port, message):
    context = zmq.Context()

    # Socket to send messages on
    socket = context.socket(zmq.PUSH)
    socket.connect(f"tcp://localhost:{target_port}")

    # Send the message
    socket.send_string(message)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_port", type=int, required=True, help="Port to send message to")
    parser.add_argument("--message", type=str, required=True, help="Message to be sent")
    args = parser.parse_args()

    main(args.target_port, args.message)

