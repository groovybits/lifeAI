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
    socket = context.socket(zmq.PUSH)
    print("binding to send message: %s:%d" % (args.output_host, args.output_port))
    socket.connect(f"tcp://{args.output_host}:{args.output_port}")

    history = []
    aipersonality = args.ai_personality
    ainame = args.ai_name

    # Send the message
    client_request = {
        "segment_number": args.segment_number,
        "mediaid": args.id,
        "mediatype": args.type,
        "username": args.username,
        "source": args.source,
        "message": args.message,
        "aipersonality": aipersonality,
        "ainame": ainame,
        "history": history,
    }
    socket.send_json(client_request)

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
    parser.add_argument("--ai_personality", type=str, required=False, default="I am GAIB the AI Bot of Life AI, I am helpful and approach the chat with love, compassion, equinimity, joy and courage.", help="AI personality")
    parser.add_argument("--ai_name", type=str, required=False, default="GAIB", help="AI name")
    args = parser.parse_args()

    main()

