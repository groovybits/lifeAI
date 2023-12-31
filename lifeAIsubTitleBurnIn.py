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
import logging
import time

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
        logger.info(f"Adding text to image: {text}")

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
        optimized_prompt = text
        if "optimized_text" in header_message:
            optimized_prompt = header_message["optimized_text"]
        elif not args.use_prompt:
            logger.error(f"Subtitle Burn-In: No optimized text, using original text: {text}")
        image = receiver.recv()

        logger.debug(f"Subtitle Burn-In: recieved image {header_message}")
        
        ## Convert the bytes back to a PIL Image object
        image = Image.open(io.BytesIO(image))

        ## check the length of the text, split into lines at breaks that keep them 80 characters or less
        ## like captions on TV, put them in an array, then count out 3 at a time and send them out
        ## repeating the header_message
        images_sent = 0
        wraptext = text
        if args.use_prompt and optimized_prompt.strip():
            wraptext = optimized_prompt
        lines = textwrap.wrap(wraptext, width=args.linewidth)
        lines = [lines[i:i + args.maxlines] for i in range(0, len(lines), args.maxlines)]

        for line in lines:
            line_string = "\n".join(line)
            image_copy = image.copy()
            header_message["index"] = images_sent
            images_sent += 1
            logger.debug(f"Subtitle Burn-In #{images_sent}: line: {line_string}")

            if args.use_prompt and optimized_prompt.strip():            
                image_copy = add_text_to_image(image_copy, optimized_prompt)
            else:
                image_copy = add_text_to_image(image_copy, line_string)
            
            # Convert PIL Image
            img_byte_arr = io.BytesIO()
            image_copy.save(img_byte_arr, format=args.format)  # Save it as PNG or JPEG depending on your preference
            image_copy = img_byte_arr.getvalue()

            ## add the text to the header_message
            header_message["text"] = line_string
            ## add the length of the text to the timestamp
            #header_message["timestamp"] = header_message["timestamp"] + (len(line_string.split(" ")) / 2)

            sender.send_json(header_message, zmq.SNDMORE)
            sender.send(image_copy)

            image_copy = None

            # sleep like 30 fps speed
            # measure length and time the sleep for the time to speak the words
            # 1 second per 10 words
            if args.framesync:
                sleep_time = len(line_string.split(" ")) / 2
                time.sleep(sleep_time)

        # send the original image with no text to clear the screen
        # sleep like 30 fps speed
        if args.clear:
            if args.framesync:
                time.sleep(3)
                
            # Convert PIL Image
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=args.format)  # Save it as PNG or JPEG depending on your preference
            image = img_byte_arr.getvalue()

            header_message["text"] = text
            header_message["index"] = images_sent

            sender.send_json(header_message, zmq.SNDMORE)
            sender.send(image)

        logger.info(f"Subtitle Burn-In: sent {images_sent} Images #{segment_number} {header_message['timestamp']}.")
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, default=3002, required=False, help="Port for receiving text input")
    parser.add_argument("--output_port", type=int, default=6002, required=False, help="Port for sending image output")
    parser.add_argument("--input_host", type=str, default="127.0.0.1", required=False, help="Port for receiving text input")
    parser.add_argument("--output_host", type=str, default="127.0.0.1", required=False, help="Port for sending image output")
    parser.add_argument("--use_prompt", action="store_true", default=False, help="Burn in the prompt that created the image")
    parser.add_argument("--format", type=str, default="PNG", help="Image format to save as. Choices are 'PNG' or 'JPEG'. Default is 'PNG'.")
    parser.add_argument("--width", type=int, default=1920, help="Width of the output image")
    parser.add_argument("--height", type=int, default=1080, help="Height of the output image")
    parser.add_argument("--maxlines", type=int, default=9999, help="Maximum number of lines per subtitle group")
    parser.add_argument("--linewidth", type=int, default=100, help="Maximum number of characters per line")
    parser.add_argument("-ll", "--loglevel", type=str, default="info", help="Logging level: debug, info...")
    parser.add_argument("--framesync", action="store_true", default=False, help="Sync frames output to duration of spoken text")
    parser.add_argument("--clear", action="store_true", default=False, help="Clear the screen after each subtitle")
    args = parser.parse_args()

    LOGLEVEL = logging.INFO

    if args.loglevel == "info":
        LOGLEVEL = logging.INFO
    elif args.loglevel == "debug":
        LOGLEVEL = logging.DEBUG
    elif args.loglevel == "warning":
        LOGLEVEL = logging.WARNING
    else:
        LOGLEVEL = logging.INFO

    log_id = time.strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(filename=f"logs/subtitleBurnIn-{log_id}.log", level=LOGLEVEL)
    logger = logging.getLogger('subTileBurnIn')

    ch = logging.StreamHandler()
    ch.setLevel(LOGLEVEL)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    context = zmq.Context()
    receiver = context.socket(zmq.SUB)
    logger.info("connected to ZMQ in: %s:%d" % (args.input_host, args.input_port))
    receiver.connect(f"tcp://{args.input_host}:{args.input_port}")
    receiver.setsockopt_string(zmq.SUBSCRIBE, "")

    sender = context.socket(zmq.PUSH)
    logger.info("binded to ZMQ out: %s:%d" % (args.output_host, args.output_port))
    sender.connect(f"tcp://{args.output_host}:{args.output_port}")

    main()

