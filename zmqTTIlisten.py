#!/usr/bin/env python

## Life AI Text to Image listener ZMQ client
#
# Chris Kennedy 2023 (C) GPL
#
# Free to use for any use as in truly free software
# as Richard Stallman intended it to be.
#

import io
import zmq
import argparse
import textwrap
import soundfile as sf
from PIL import Image

def image_to_ascii(image, width):
    image = image.resize((width, int((image.height/image.width) * width * 0.55)), Image.LANCZOS)
    image = image.convert('L')  # Convert to grayscale

    pixels = list(image.getdata())
    ascii_chars = ["@", "#", "S", "%", "?", "*", "+", ";", ":", ",", "."]
    ascii_image = [ascii_chars[pixel//25] for pixel in pixels]
    ascii_image = ''.join([''.join(ascii_image[i:i+width]) + '\n' for i in range(0, len(ascii_image), width)])
    return ascii_image


def main(input_port, output_file=None, audio_format=None):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    print("connecting to ports in: %s:%d" % (args.input_host, args.input_port))
    socket.connect(f"tcp://{args.input_host}:{args.input_port}")
    #socket.setsockopt_string(zmq.SUBSCRIBE, "")

    while True:
        try:
            # Receive the segment number (header) first
            header_str = socket.recv_string()

            # Now, receive the binary audio data
            image = socket.recv()

            # Print the header
            print(f"Header: {header_str}")

            # Check if we need to output to a file
            if output_file:
                if image_format == "pil":
                    with open(output_file, 'wb') as f:
                        f.write(image)
                    print(f"Image saved to {args.output_file} as PIL\n")
                else:
                    with open(output_file, 'wb') as f:
                        f.write(image)
                    print(f"Payload written to {output_file}\n")

            # Convert the bytes back to a PIL Image object
            image = Image.open(io.BytesIO(image))
            payload_hex = image_to_ascii(image, 80)
            print(f"Image Payload (Hex):\n{payload_hex}\n")
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, required=False, default=3002, help="Port for receiving image as PIL numpy arrays")
    parser.add_argument("--input_host", type=str, required=False, default="127.0.0.1", help="Host for receiving image as PIL numpy arrays")
    parser.add_argument("--output_file", type=str, help="Path to save the received image")
    parser.add_argument("--image_format", choices=["pil", "raw"], default="pil", help="Image format to save as. Choices are 'pil' or 'raw'.")
    args = parser.parse_args()

    main(args.input_port, args.output_file, args.image_format)

