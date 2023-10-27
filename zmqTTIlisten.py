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
import soundfile as sf
from PIL import Image
import re

def image_to_ascii(image):
    image = image.resize((args.width, int((image.height/image.width) * args.width * 0.55)), Image.LANCZOS)
    image = image.convert('L')  # Convert to grayscale

    pixels = list(image.getdata())
    ascii_chars = ["@", "#", "S", "%", "?", "*", "+", ";", ":", ",", "."]
    ascii_image = [ascii_chars[pixel//25] for pixel in pixels]
    ascii_image = ''.join([''.join(ascii_image[i:i+args.width]) + '\n' for i in range(0, len(ascii_image), args.width)])
    return ascii_image


def main():
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    print("connected to ZMQ in: %s:%d" % (args.input_host, args.input_port))
    socket.connect(f"tcp://{args.input_host}:{args.input_port}")
    #socket.setsockopt_string(zmq.SUBSCRIBE, "")

    while True:
        try:
            # Receive the segment number (header) first
            segment_number = socket.recv_string()
            image_prompt = socket.recv_string()
            image_text = socket.recv_string()

            # Now, receive the binary audio data
            image = socket.recv()

            # Print the header
            print(f"Received image segment #{segment_number}")

            # create an output file using the prompt and segment_number
            prompt_summary = re.sub(r'[^a-zA-Z0-9]', '', image_prompt)[:50]
            image_file = f"{args.output_directory}/{segment_number}_{prompt_summary}.png"
            if args.image_format == "pil":
                with open(image_file, 'wb') as f:
                    f.write(image)
                print(f"Image saved to {image_file} as PIL\n")
            else:
                with open(image_file, 'wb') as f:
                    f.write(image)
                print(f"Payload written to {image_file}\n")

            # Convert the bytes back to a PIL Image object
            image = Image.open(io.BytesIO(image))
            payload_hex = image_to_ascii(image)
            print(f"Image #{segment_number} Payload (Hex):\n{payload_hex}\nImage Prompt: {image_prompt}\nImage Text: {image_text}\n")
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, required=False, default=3003, help="Port for receiving image as PIL numpy arrays")
    parser.add_argument("--input_host", type=str, required=False, default="127.0.0.1", help="Host for receiving image as PIL numpy arrays")
    parser.add_argument("--output_directory", default="images", type=str, help="Directory path to save the received images in")
    parser.add_argument("--image_format", choices=["pil", "raw"], default="pil", help="Image format to save as. Choices are 'pil' or 'raw'.")
    parser.add_argument("--width", type=int, default=80, help="Width of the output image")
    args = parser.parse_args()

    main()

