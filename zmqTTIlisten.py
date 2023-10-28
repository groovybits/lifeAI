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
import os
import cv2
import numpy as np

def image_to_ascii(image):
    image = image.resize((args.width, int((image.height/image.width) * args.width * 0.55)), Image.LANCZOS)
    image = image.convert('L')  # Convert to grayscale

    pixels = list(image.getdata())
    ascii_chars = ["@", "#", "S", "%", "?", "*", "+", ";", ":", ",", "."]
    ascii_image = [ascii_chars[pixel//25] for pixel in pixels]
    ascii_image = ''.join([''.join(ascii_image[i:i+args.width]) + '\n' for i in range(0, len(ascii_image), args.width)])
    return ascii_image

def render(image):
    # Convert PIL Image to NumPy array
    image_np = np.array(image)

    # Convert RGB to BGR (OpenCV uses BGR format)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    cv2.imshow('GAIB The Groovy AI Bot', image_bgr)

    k = cv2.waitKey(10)
    if k == ord('f'):
        cv2.setWindowProperty('GAIB The Groovy AI Bot', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    elif k == ord('m'):
        cv2.setWindowProperty('GAIB The Groovy AI Bot', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    elif k == ord('q') or k == 27:
        cv2.destroyAllWindows()

def main():
    while True:
        # Receive the segment number (header) first
        segment_number = socket.recv_string()
        id = socket.recv_string()
        type = socket.recv_string()
        username = socket.recv_string()
        source = socket.recv_string()
        message = socket.recv_string()
        image_prompt = socket.recv_string()
        image_text = socket.recv_string()

        # Now, receive the binary audio data
        image = socket.recv()

        # Print the header
        print(f"Received image segment #{segment_number}")

        # create an output file using the prompt and segment_number
        image_file = f"{args.output_directory}/{id}/{segment_number}.png"
        os.makedirs(os.path.dirname(image_file), exist_ok=True)
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
        print(f"Image #{segment_number} Payload (Hex):\n{payload_hex}\nImage Prompt: {image_prompt}\nImage Text: {image_text}\nMessage: {message}\n")

        if args.render:
            render(image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, required=False, default=3003, help="Port for receiving image as PIL numpy arrays")
    parser.add_argument("--input_host", type=str, required=False, default="127.0.0.1", help="Host for receiving image as PIL numpy arrays")
    parser.add_argument("--output_directory", default="images", type=str, help="Directory path to save the received images in")
    parser.add_argument("--image_format", choices=["pil", "raw"], default="pil", help="Image format to save as. Choices are 'pil' or 'raw'.")
    parser.add_argument("--width", type=int, default=80, help="Width of the output image")
    parser.add_argument("-r", "--render", action="store_true", default=False, help="Render the output to a GUI OpenCV window for playback viewing.")
    args = parser.parse_args()

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    print("connected to ZMQ in: %s:%d" % (args.input_host, args.input_port))
    socket.connect(f"tcp://{args.input_host}:{args.input_port}")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    main()

