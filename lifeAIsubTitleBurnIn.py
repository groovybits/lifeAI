#!/usr/bin/env python

## Life AI Subtitle Burn-In module
#
# Chris Kennedy 2023 (C) GPL
#
# Free to use for any use as in truly free software
# as Richard Stallman intended it to be.
#

import zmq
import argparse
import io
from PIL import Image, ImageDraw, ImageFont
import cv2
import warnings
import urllib3
import numpy as np
import textwrap

warnings.simplefilter(action='ignore', category=Warning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
from urllib3.exceptions import NotOpenSSLWarning
warnings.simplefilter(action='ignore', category=NotOpenSSLWarning)


## Japanese writing on images
def draw_japanese_text_on_image(image_np, text, position, font_path, font_size):
    # Convert to a PIL Image
    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

    # Prepare drawing context
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype(font_path, font_size)

    # Define the border width for the text
    border_width = 5

    # Get text size using getbbox
    x, y = position
    bbox = font.getbbox(text)
    text_width, text_height = bbox[2], bbox[3]
    y = y - text_height
    x = x + text_width / 2

    # Draw text border (outline)
    for i in range(-border_width, border_width + 1):
        for j in range(-border_width, border_width + 1):
            draw.text((x + i, y + j), text, font=font, fill=(0, 0, 0))  # Black border

    # Draw text on image
    draw.text((x, y), text, font=font, fill=(255, 255, 255))  # White fill

    # Convert back to NumPy array
    image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    return image_np


def add_text_to_image(image, text):
    if image is not None:
        print("Adding text to image")

        # Maintain aspect ratio and add black bars
        width, height = image.size
        desired_ratio = width / height
        current_ratio = width / height

        # Convert the PIL Image to a NumPy array for OpenCV operations
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR for OpenCV

        if current_ratio > desired_ratio:
            new_width = int(height * desired_ratio)
            padding = max(0, (new_width - width) // 2)
            image = cv2.copyMakeBorder(image, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            new_height = int(width / desired_ratio)
            padding = max(0, (new_height - height) // 2)
            image = cv2.copyMakeBorder(image, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # Resize to the desired resolution
        # Calculate the scaling factors
        x_scale = args.width / image.shape[1]
        y_scale = args.height / image.shape[0]
        scale_factor = min(x_scale, y_scale)

        # Compute new width and height while maintaining the aspect ratio
        new_width = int(image.shape[1] * scale_factor)
        new_height = int(image.shape[0] * scale_factor)

        # Resize the image
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # If the resized image doesn't match the desired resolution, pad with black
        if new_width != args.width or new_height != args.height:
            top_padding = (args.height - new_height) // 2
            bottom_padding = args.height - new_height - top_padding
            left_padding = (args.width - new_width) // 2
            right_padding = args.width - new_width - left_padding
            image = cv2.copyMakeBorder(image, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        width, height = image.shape[1], image.shape[0]
        current_ratio = width / height

        def contains_japanese(text):
            for char in text:
                if any([start <= ord(char) <= end for start, end in [
                    (0x3040, 0x309F),  # Hiragana
                    (0x30A0, 0x30FF),  # Katakana
                    (0x4E00, 0x9FFF),  # Kanji
                    (0x3400, 0x4DBF)   # Kanji (extension A)
                ]]):
                    return True
            return False

        wrap_width = 30
        is_wide = False
        if current_ratio > 512/512:
            wrap_width = 45
            is_wide = True
        wrapped_text = textwrap.wrap(text, width=wrap_width, fix_sentence_endings=False, break_long_words=False, break_on_hyphens=False)  # Adjusted width
        y_pos = height - 40  # Adjusted height from bottom

        font_size = 1
        font_thickness = 2  # Adjusted for bolder font
        border_thickness = 8  # Adjusted for bolder border

        if is_wide:
            font_size = 2
            font_thickness = 4
            border_thickness = 15
        elif width < 600:  # Assuming smaller images have widths less than 600, adjust if necessary
            font_size = 1
            font_thickness = 3
            border_thickness = 10

        for line in reversed(wrapped_text):
            text_width, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_DUPLEX, font_size, font_thickness)[0]
            x_pos = (width - text_width) // 2  # Center the text
            if contains_japanese(line):
                image = draw_japanese_text_on_image(image, line, (x_pos, y_pos), args.japanesefont,60)
            else:
                cv2.putText(image, line, (x_pos, y_pos), cv2.FONT_HERSHEY_DUPLEX, font_size, (0, 0, 0), border_thickness)
                cv2.putText(image, line, (x_pos, y_pos), cv2.FONT_HERSHEY_DUPLEX, font_size, (255, 255, 255), font_thickness)
            y_pos -= 60

        ## Convert back from numpy array
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    return image  # returning the modified image

def main():
    while True:
        header_message = receiver.recv_json()
        """ 
          header_message = {
            "segment_number": segment_number,
            "mediaid": mediaid,
            "mediatype": mediatype,
            "username": username,
            "source": source,
            "message": message,
            "text": text,
            "optimized_text": optimized_text,
        }"""
        # fill out variables from header_message
        segment_number = header_message["segment_number"]
        text = header_message["text"]
        optimized_prompt = ""
        if "optimized_text" in header_message:
            optimized_prompt = header_message["optimized_text"]
        elif not args.use_prompt:
            print(f"Subtitle Burn-In: No optimized text, using original text.")
        image = receiver.recv()

        print(f"Subtitle Burn-In: recieved image {header_message}")

        # check if we have optimized text
        
        ## Convert the bytes back to a PIL Image object
        image = Image.open(io.BytesIO(image))

        if args.use_prompt and optimized_prompt.strip():            
            image = add_text_to_image(image, optimized_prompt)
        else:
            image = add_text_to_image(image, text)
        
        # Convert PIL Image
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=args.format)  # Save it as PNG or JPEG depending on your preference
        image = img_byte_arr.getvalue()

        sender.send_json(header_message, zmq.SNDMORE)
        sender.send(image)

        print("Subtitle Burn-In: sent image #%s" % segment_number)
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, default=3002, required=False, help="Port for receiving text input")
    parser.add_argument("--output_port", type=int, default=3003, required=False, help="Port for sending image output")
    parser.add_argument("--input_host", type=str, default="127.0.0.1", required=False, help="Port for receiving text input")
    parser.add_argument("--output_host", type=str, default="127.0.0.1", required=False, help="Port for sending image output")
    parser.add_argument("--use_prompt", action="store_true", default=False, help="Burn in the prompt that created the image")
    parser.add_argument("--format", type=str, default="PNG", help="Image format to save as. Choices are 'PNG' or 'JPEG'. Default is 'PNG'.")
    parser.add_argument("--width", type=int, default=1024, help="Width of the output image")
    parser.add_argument("--height", type=int, default=1024, help="Height of the output image")
    parser.add_argument("--maxlines", type=int, default=3, help="Maximum number of lines per subtitle group")

    args = parser.parse_args()

    context = zmq.Context()
    receiver = context.socket(zmq.SUB)
    print("connected to ZMQ in: %s:%d" % (args.input_host, args.input_port))
    receiver.connect(f"tcp://{args.input_host}:{args.input_port}")
    receiver.setsockopt_string(zmq.SUBSCRIBE, "")

    sender = context.socket(zmq.PUB)
    print("binded to ZMQ out: %s:%d" % (args.output_host, args.output_port))
    sender.bind(f"tcp://{args.output_host}:{args.output_port}")

    main()

