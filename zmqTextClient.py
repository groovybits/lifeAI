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
import uuid

def main():
    context = zmq.Context()

    # Socket to send messages on
    tti_socket = context.socket(zmq.PUSH)
    print("binding to send message: %s:%d" % (args.output_host, args.output_port))
    tti_socket.connect(f"tcp://{args.output_host}:{args.output_port}")

    # Send the message
    tti_socket.send_string(args.segment_number, zmq.SNDMORE)
    tti_socket.send_string(args.id, zmq.SNDMORE)
    tti_socket.send_string(args.type, zmq.SNDMORE)
    tti_socket.send_string(args.username, zmq.SNDMORE)
    tti_socket.send_string(args.source, zmq.SNDMORE)
    tti_socket.send_string(args.message)

    print("Message sent")

if __name__ == "__main__":
    default_id = uuid.uuid4().hex[:8]

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_port", type=int, default=1500, required=False, help="Port to send message to")
    parser.add_argument("--output_host", type=str, default="127.0.0.1", required=False, help="Host for sending message to.")
    parser.add_argument("--message", type=str, required=True, help="Message to be sent")
    parser.add_argument("--segment_number", type=str, required=True, help="Segment number")
    parser.add_argument("--id", type=str, required=False, default=default_id, help="ID of the message")
    parser.add_argument("--type", type=str, required=False, default="chat", help="Type of message")
    parser.add_argument("--username", type=str, required=False, default="anonymous", help="Username of sender")
    parser.add_argument("--source", type=str, required=False, default="lifeAI", help="Source of message")
    args = parser.parse_args()

    main()

