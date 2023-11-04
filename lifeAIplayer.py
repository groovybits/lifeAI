#!/usr/bin/env python

## Life AI Player ZMQ client
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
import logging
import time
import soundfile as sf
import pygame as pygame_speek
import pygame as pygame_music
import queue
import threading
from queue import PriorityQueue
from PIL import Image, ImageDraw, ImageFont
import cv2
import textwrap
import json
from collections import deque
from pydub import AudioSegment
import traceback

# Queue to store the last images
past_images_queue = deque(maxlen=6)  # Assuming 6 images for each side

def create_16_9_image(center_image, side_images, target_width, target_height):
    # Scale the main image to fit the height of the 16:9 image
    main_image_scaled = center_image.resize((target_height, target_height), Image.LANCZOS)

    # Create a new image with the target 16:9 dimensions
    final_image = Image.new('RGB', (target_width, target_height))

    # Calculate the width of the area on each side of the main image
    side_area_width = (target_width - target_height) // 2

    # Calculate the size for the side images to fill the space as much as possible
    # Given we want to fit 3 images per side, we divide the height by 3
    side_image_size = target_height // 3

    # Split the side images for left and right
    left_side_images = side_images[:3]
    right_side_images = side_images[3:6]

    # Resize side images to fill the space
    left_resized_side_images = [img.resize((side_image_size, side_image_size), Image.LANCZOS) for img in left_side_images]
    right_resized_side_images = [img.resize((side_image_size, side_image_size), Image.LANCZOS) for img in right_side_images]

    # Paste the side images to fill the left and right areas
    for i in range(3):
        # Left side images
        final_image.paste(left_resized_side_images[i], (0, i * side_image_size))

        # Right side images
        final_image.paste(right_resized_side_images[i], (target_width - side_image_size, i * side_image_size))

    # Paste the scaled main image in the center
    final_image.paste(main_image_scaled, (side_area_width, 0))

    return final_image

def create_filmstrip_images(center_image, side_images):
    # Assuming side_images is a list of 6 images, 3 for left and 3 for right
    left_images = side_images[:3]
    right_images = side_images[3:]
    
    # Combine the left images, the center image, and the right images horizontally
    images_to_combine = left_images + [center_image] + right_images
    combined_width = sum(img.size[0] for img in images_to_combine)
    combined_height = max(img.size[1] for img in images_to_combine)

    # Create a new image with the combined dimensions
    wide_image = Image.new('RGB', (combined_width, combined_height))

    # Paste the images into the wide_image
    x_offset = 0
    for img in images_to_combine:
        wide_image.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    
    return wide_image

# Main function to process the new image
def process_new_image(new_image, text, args):
    target_width = args.width  # This should be set to the desired width for 16:9 aspect ratio
    target_height = args.height  # This should be set to the height corresponding to the 16:9 aspect ratio
    
    # Check if we have enough images to fill the sides
    if len(past_images_queue) >= 6:
        # Use the 6 most recent images for each side
        side_images = list(past_images_queue)
        final_image = create_16_9_image(new_image, side_images, target_width, target_height)
        final_image = add_text_to_image(final_image, text)
    else:
        # Not enough images, just add text to the new_image
        final_image = add_text_to_image(new_image, text)

    # Add the new image to the queue for future use
    past_images_queue.appendleft(new_image)

    return final_image

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
        logger.info(f"Adding text to image: {text[:100]}")

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

def save_json(header, mediaid):
    assets_dir = "assets"
    os.makedirs(assets_dir, exist_ok=True)
    with open(f"{assets_dir}/{mediaid}.json", 'w') as json_file:
        json.dump(header, json_file, indent=4)

def save_asset(asset, mediaid, segment_number, asset_type):
    directory = f"{asset_type}/{mediaid}"
    os.makedirs(directory, exist_ok=True)
    file_path = f"{directory}/{segment_number}"

    if asset_type == "audio":
        file_path += ".wav"
        with open(file_path, 'wb') as file:
            file.write(asset)
    elif asset_type == "images":
        file_path += ".png"
        img_byte_arr = io.BytesIO()
        asset.save(img_byte_arr, format="PNG")  # Save it as PNG or JPEG depending on your preference
        asset = img_byte_arr.getvalue()

        with open(file_path, 'wb') as f:
            f.write(asset)
    elif asset_type == "music":
        file_path += ".wav"
        with open(file_path, 'wb') as file:
            file.write(asset)

class AudioMixer:
    def __init__(self, pygame_mixer):
        self.pygame_mixer = pygame_mixer
        self.music_segment = None
        self.music_position = 0  # Position in the music track
        self.speech_sample_rate = 22500  # Assuming the speech is also at 32kHz

    def load_music(self, music_audio_samples):
        """Load initial music audio and prepare it for looping."""
        self.music_segment = AudioSegment.from_file(io.BytesIO(music_audio_samples), format="wav")

    def mix_speech_with_music(self, speech_audio_samples):
        speech_segment = AudioSegment.from_file(io.BytesIO(speech_audio_samples), format="wav", frame_rate=self.speech_sample_rate)

        if self.music_segment:
            # Ensure music and speech segments have the same frame rate
            if speech_segment.frame_rate != self.music_segment.frame_rate:
                self.music_segment = self.music_segment.set_frame_rate(speech_segment.frame_rate)

            # Calculate the amount of music needed for the current speech segment
            required_music_length = len(speech_segment)

            # Loop the music if it is shorter than the speech segment
            while len(self.music_segment) - self.music_position < required_music_length:
                self.music_segment += self.music_segment

            # Extract the portion of the music that we need
            music_to_mix = self.music_segment[self.music_position:self.music_position + required_music_length]
            self.music_position += required_music_length

            # Mix speech and music
            mixed_segment = speech_segment.overlay(music_to_mix)
        else:
            # If there is no music, just use the speech segment
            mixed_segment = speech_segment

        # Output the mixed audio
        mixed_audio_data = mixed_segment.raw_data
        return mixed_audio_data

    def change_music(self, new_music_audio_samples):
        # Create a new music segment from the raw audio samples
        new_music_segment = AudioSegment.from_file(io.BytesIO(new_music_audio_samples), format="wav")

        # Check if the sample rate of the new music matches the speech sample rate
        if new_music_segment.frame_rate != self.speech_sample_rate:
            # If not, set the frame rate of the music to match the speech
            new_music_segment = new_music_segment.set_frame_rate(self.speech_sample_rate)

        # Replace the old music with the new one and reset the music position
        self.music_segment = new_music_segment
        self.music_position = 0

def play_audio(audio_samples, pygame):  
    audiobuf = io.BytesIO(audio_samples)
    if audiobuf:
        ## Speak WAV TTS Output using pygame
        pygame.mixer.music.load(audiobuf)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(1)

def mix_and_loop_music_with_speech(speech_audio_samples, music_audio_samples, pygame_mixer):
    # Load the speech and music into pydub.AudioSegment ojects
    speech=None
    if speech_audio_samples:
        speech = AudioSegment.from_file(io.BytesIO(speech_audio_samples), format="wav")
    music = AudioSegment.from_file(io.BytesIO(music_audio_samples), format="wav")
    
    # Calculate how many times the music needs to be looped
    loop_count = 1
    if speech:
        loop_count = int(speech.duration_seconds // music.duration_seconds) + 1
    looped_music = music * loop_count  # Loop the music
    
    # Overlay the speech onto the looped music
    mixed = None
    if speech:
        # lower volume of music to .5
        looped_music = looped_music - 10
        mixed = looped_music.overlay(speech, position=0)
    else:
        mixed = looped_music
    
    # Export the mixed audio to a byte buffer
    mixed_buffer = io.BytesIO()
    mixed.export(mixed_buffer, format="wav")
    
    # Load the mixed audio into pygame and play
    mixed_buffer.seek(0)  # Rewind buffer to the start
    pygame_mixer.mixer.music.load(mixed_buffer)
    pygame_mixer.mixer.music.play()
    while pygame_mixer.mixer.music.get_busy():
            pygame_mixer.time.Clock().tick(1)

def playback(image, audio, pygame_player, music=None):
    # play both audio and display image with audio blocking till finished
    if image:
        render(image)
    if music:
        mix_and_loop_music_with_speech(audio, music, pygame_player)
    elif audio:
        play_audio(audio, pygame_player)

def get_audio_duration(audio_samples):
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_samples), format="wav")
    duration_ms = len(audio_segment)  # Duration in milliseconds
    duration_s = duration_ms / 1000.0  # Convert to seconds
    return duration_s

def main():
    ## Main routine
    pygame_speek.mixer.init(frequency=22500, size=-16, channels=1, buffer=1024)
    pygame_speek.init()
    audio_mixer = AudioMixer(pygame_speek)

    last_image_header = None
    last_image_segment = None
    last_music_segment = None

    last_music_change = 0

    while True:
        header_message = socket.recv_json()

        segment_number = header_message["segment_number"]
        timestamp = header_message["timestamp"]
        mediaid = header_message["mediaid"]
        
        message = header_message["message"]
        text = header_message["text"]
        
        optimized_prompt = text
        if 'optimized_text' in header_message:
            optimized_prompt = header_message["optimized_text"]

        type = header_message["stream"]
        if type == "music":
            # Now, receive the binary audio data
            music = socket.recv()

            # Print the header
            print(f"Received music segment {type} #{segment_number} {timestamp}: {mediaid} {len(text)} characters")
            
            save_json(message, mediaid)  # or image_message, if it's the one to be saved

            # Save audio asset
            save_asset(music, mediaid, segment_number, "music")
        
            # queue in music_buffer header and music
            music_buffer.put((header_message, music))

        if type == "speek":
            # Now, receive the binary audio data
            audio = socket.recv()

            # Print the header
            print(f"Received audio segment {type} #{segment_number} {timestamp}: {mediaid} {len(text)} characters")

            # queue the header and audio together
            audio_buffer.put((header_message, audio))

        ## Image
        if type == "image":
            # Now, receive the binary audio data
            image = socket.recv()

            # Print the header
            print(f"Received image segment {type} #{segment_number} {timestamp}: {mediaid} {len(text)} characters")

            try:
                # Convert the bytes back to a PIL Image object
                image = Image.open(io.BytesIO(image))

                print(f"Image Prompt: {optimized_prompt}\Original Text: {text}\nOriginal Question:{message}")

                # queue the header and image together
                image_buffer.put((header_message, image))
            except Exception as e:
                logger.error(f"Error converting image to ascii: {e}")

        ## get an audio sample and header, get the text field from it, then get an image and header and burn in the text from the audio header to the image and render it while playing the audio
        if not audio_buffer.empty() and (not image_buffer.empty() or last_image_header is not None):
            audio_message, audio_asset = audio_buffer.get()
            music_message, music_asset = None, None
            if not args.nomusic and  not music_buffer.empty():
                music_message, music_asset = music_buffer.get()
                last_music_segment = music_asset
            image_message = None
            image_asset = None
            if last_image_header is not None:
                image_message = last_image_header
                image_asset = last_image_segment
                last_image_header = None
                last_image_segment = None
            else:
                image_message, image_asset = image_buffer.get()

            text = audio_message["text"]
            optimized_prompt = text
            if 'optimized_text' in audio_message:
                optimized_prompt = audio_message["optimized_text"]

            if audio_message['timestamp'] < image_message['timestamp']:
                logger.debug(f"Audio segment #{audio_message['segment_number']} is older than image segment #{image_message['segment_number']}.")

            if audio_message['timestamp'] > image_message['timestamp']:
                logger.debug(f"Audio segment #{audio_message['segment_number']} is newer than image segment #{image_message['segment_number']}.")

            # Convert the PIL Image to a NumPy array for OpenCV operations
            image_np = np.array(image)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            if args.burn_prompt:
                #image_np = add_text_to_image(image_asset, optimized_prompt)
                image_np = process_new_image(image_asset, optimized_prompt)
            else:
                #image_np = add_text_to_image(image_asset, text)
                image_np = process_new_image(image_asset, text, args)

            ## write out json into a directory assets/{mediaid}.json with it pretty pretty printed, 
            ## write out assets to file locations audio/ and images/ as mediaid/segment_number.wav 
            ## and mediaid/segment_number.png too.
            ## audio_message and image_message are the headers, image_np and audio_asset are the assets
            # Save JSON header
            save_json(audio_message, mediaid)  # or image_message, if it's the one to be saved

            # Save audio asset
            save_asset(audio_asset, mediaid, segment_number, "audio")

            # Save image asset
            save_asset(image_np, mediaid, segment_number, "images")

            # Play audio and display image
            playback(image_np, audio_asset, pygame_speek, last_music_segment)
        else:
            # rebroadcast the last image and audio and music if we don't have any current
            if last_music_segment is not None:
                playback(None, None, pygame_speek, last_music_segment)
        
        if not music_buffer.empty():
            if args.nomusic:
                music_message, music = music_buffer.get()
                while not music_buffer.empty():
                    music_message, music = music_buffer.get()
                continue

            if last_music_change == 0 or time.time() - last_music_change > args.music_interval:
                music_message, music = music_buffer.get()
                print(f"Music Prompt: {music_message['message']}\nOriginal Text: {music_message['text']}\nOriginal Question:{music_message['message']}")
                if last_music_change > 0:
                    print(f"Changing music because it's been {time.time() - last_music_change} seconds since the last music change.")
                    audio_mixer.change_music(music)
                else:
                    # Load the initial music track
                    audio_mixer.load_music(music)

                last_music_segment = music
                last_music_change = time.time()
            else:
                print(f"Skipping music because it's too soon since the last music change {time.time() - last_music_change}.")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, required=False, default=6003, help="Port for receiving image as PIL numpy arrays")
    parser.add_argument("--input_host", type=str, required=False, default="127.0.0.1", help="Host for receiving image as PIL numpy arrays")
    parser.add_argument("-ll", "--loglevel", type=str, default="info", help="Logging level: debug, info...")
    parser.add_argument("-f", "--freq", type=int, default=22050, help="Sampling frequency for audio playback")
    parser.add_argument("-c", "--channels", type=int, default=1, help="Number of channels for audio playback")
    parser.add_argument("--burn_prompt", action="store_true", default=False, help="Burn in the prompt that created the image")
    parser.add_argument("--width", type=int, default=1920, help="Width of the output image")
    parser.add_argument("--height", type=int, default=1080, help="Height of the output image")
    parser.add_argument("--volume", type=float, default=0.5, help="Volume for audio playback")
    parser.add_argument("--music_interval", type=float, default=60, help="Interval between music changes")
    parser.add_argument("--nomusic", action="store_true", default=False, help="Disable music")
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
    logging.basicConfig(filename=f"logs/lifeAIplayer-{log_id}.log", level=LOGLEVEL)
    logger = logging.getLogger('lifeAIplayer')

    ch = logging.StreamHandler()
    ch.setLevel(LOGLEVEL)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    logger.info("connected to ZMQ in: %s:%d" % (args.input_host, args.input_port))
    socket.connect(f"tcp://{args.input_host}:{args.input_port}")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    audio_buffer = queue.Queue()
    music_buffer = queue.Queue()    
    image_buffer = queue.Queue()

    main()

