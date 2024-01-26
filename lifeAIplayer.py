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
import pygame
import queue
import threading
from queue import PriorityQueue
from PIL import Image, ImageDraw, ImageFont
import cv2
import textwrap
import json
from collections import deque
from pydub import AudioSegment
import magic
from dotenv import load_dotenv

load_dotenv()

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
def process_new_image(new_image, text, args, unique_image=False, banner=""):
    target_width = args.width  # This should be set to the desired width for 16:9 aspect ratio
    target_height = args.height  # This should be set to the height corresponding to the 16:9 aspect ratio

    # Check if we have enough images to fill the sides
    if len(past_images_queue) >= 6:
        # Use the 6 most recent images for each side
        side_images = list(past_images_queue)
        final_image = create_16_9_image(new_image, side_images, target_width, target_height)
        final_image = add_text_to_image(final_image, text, banner)
    else:
        # Not enough images, just add text to the new_image
        final_image = add_text_to_image(new_image, text, banner)

    # Add the new image to the queue for future use
    if unique_image:
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

def add_text_to_image(image, text, banner=""):
    if image is not None:
        logger.info(f"Adding text to image: {text[:80]}")

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
        if current_ratio > 1.0:
            wrap_width = 50
        wrapped_text = textwrap.wrap(text, width=wrap_width, fix_sentence_endings=False, break_long_words=False, break_on_hyphens=False)
        y_pos = height  # Adjusted height from bottom

        font_size = 2
        font_thickness = 6
        border_thickness = 15

        # Configuration for the banner text
        banner_font_size = 1  # Smaller font size for the banner
        banner_font_thickness = 2  # Thickness of the banner font
        banner_outline_color = (0, 0, 0)  # Black outline for better visibility
        banner_border_thickness = 1  # Thickness of the text outline

        # Draw banner if it's not empty on the top of the image from left to right
        if banner != "":
            ((text_width, text_height), baseline) = cv2.getTextSize(
                banner[:width-10], cv2.FONT_HERSHEY_DUPLEX, banner_font_size, banner_font_thickness)
            x_pos_t = 5  # Adjusted width from left
            y_pos_t = text_height + 10  # Adjusted height from top, with some padding

            # Draw a solid black rectangle for the banner background
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (width, y_pos_t + 10), (0, 0, 0), -1)

            # Draw text shadow for the banner
            shadow_offset = 2  # Smaller offset for the shadow
            cv2.putText(image, banner[:width-10], (x_pos_t + shadow_offset, y_pos_t - shadow_offset),
                        cv2.FONT_HERSHEY_DUPLEX, banner_font_size, (0, 0, 0), banner_font_thickness)
            alpha = 0.5  # Transparency factor.
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

            # Draw text outline for the banner
            cv2.putText(image, banner[:width-10], (x_pos_t, y_pos_t), cv2.FONT_HERSHEY_DUPLEX, banner_font_size, banner_outline_color, banner_border_thickness)

            # Draw the main banner text
            cv2.putText(image, banner[:width-10], (x_pos_t, y_pos_t), cv2.FONT_HERSHEY_DUPLEX, banner_font_size, (255, 255, 0), banner_font_thickness)

        for line in reversed(wrapped_text):
            # Get the text size, baseline, and adjust the y_pos
            ((text_width, text_height), baseline) = cv2.getTextSize(line[:width], cv2.FONT_HERSHEY_DUPLEX, font_size, font_thickness)
            x_pos = (width - text_width) // 2  # Center the text
            y_pos -= (baseline + text_height + 10)  # Adjust y_pos for each line, reduce padding

            # Calculate the rectangle coordinates with less height
            rect_x_left = x_pos - 10
            rect_y_top = y_pos - text_height - 10  # Reduced padding for height
            rect_x_right = x_pos + text_width + 10
            rect_y_bottom = y_pos + 18  # Reduced padding for height

            # Draw a semi-transparent rectangle
            overlay = image.copy()
            cv2.rectangle(overlay, (rect_x_left, rect_y_top), (rect_x_right, rect_y_bottom), (0, 0, 0), -1)
            alpha = 0.5  # Transparency factor.
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

            # Draw text shadow for a drop shadow effect
            shadow_offset = 4  # Offset for the shadow, adjust as needed
            cv2.putText(image, line[:width], (x_pos + shadow_offset, y_pos + shadow_offset), cv2.FONT_HERSHEY_DUPLEX, font_size, (0, 0, 0), font_thickness)

            # Draw text outline
            cv2.putText(image, line[:width], (x_pos, y_pos), cv2.FONT_HERSHEY_DUPLEX, font_size, (0, 0, 0), border_thickness)

            # Draw the main text
            cv2.putText(image, line[:width], (x_pos, y_pos), cv2.FONT_HERSHEY_DUPLEX, font_size, (255, 255, 255), font_thickness)

        # Convert back from numpy array
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

def update_image(duration_ms):
    k = cv2.waitKey(duration_ms) & 0xFF
    if k == ord('f'):
        print(f"Render: Toggling fullscreen.")
        cv2.moveWindow(args.title, 0, 0)
        # maximize_window()
        cv2.setWindowProperty(args.title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    elif k == ord('m'):
        print(f"Render: Toggling maximized.")
        cv2.setWindowProperty(args.title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    elif k == ord('q') or k == 27:
        print(f"Render: Quitting.")
        cv2.destroyAllWindows()

def render(image, duration):
    # Convert PIL Image to NumPy array
    image_np = np.array(image)

    # Convert RGB to BGR (OpenCV uses BGR format)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    cv2.imshow(args.title, image_bgr)

    duration_ms = 1 #int(duration * 1000 - 200)
    update_image(duration_ms)

def save_json(header, mediaid, type, segment_number):
    assets_dir = f"assets/{mediaid}/{type}"
    os.makedirs(assets_dir, exist_ok=True)
    with open(f"{assets_dir}/{mediaid}_{type}_{segment_number}.json", 'w') as json_file:
        json.dump(header, json_file)

def save_asset(asset, mediaid, segment_number, type):
    directory = f"assets/{mediaid}/{type}"
    os.makedirs(directory, exist_ok=True)
    file_path = f"{directory}/{mediaid}_{type}_{segment_number}"

    if type == "speek":
        file_path += ".wav"
        with open(file_path, 'wb') as file:
            file.write(asset)
    elif type == "image":
        file_path += ".png"
        img_byte_arr = io.BytesIO()
        asset.save(img_byte_arr, format="PNG")  # Save it as PNG or JPEG depending on your preference
        asset = img_byte_arr.getvalue()

        with open(file_path, 'wb') as f:
            f.write(asset)
    elif type == "music":
        file_path += ".wav"
        with open(file_path, 'wb') as file:
            file.write(asset)

def get_audio_duration(audio_samples):
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_samples), format="wav")
    duration_ms = len(audio_segment)  # Duration in milliseconds
    duration_s = duration_ms / 1000.0  # Convert to seconds
    return duration_s

class BackgroundMusic(threading.Thread):
    def __init__(self):
        super().__init__()
        self.audio_buffer = None
        self.running = True
        self.lock = threading.Lock()  # Lock to synchronize access to audio_buffer
        self.channel = audio_channel_music  # Assign a specific channel
        self.complete = True
        self.switching = False

    def run(self):
        while self.running:
            if self.audio_buffer:
                self.play_audio()
            else:
                time.sleep(0.1)

    def play_audio(self):
        audiobuf = None
        with self.lock:
            if self.audio_buffer != None:
                audiobuf = io.BytesIO(self.audio_buffer)

        if audiobuf:
            # Load the audio data into a Sound object
            sound = pygame.mixer.Sound(audiobuf)
            self.switching = False
            self.channel.play(sound, loops=0, maxtime=0, fade_ms=10)  # Play the Sound object on this channel
            while self.channel.get_busy():  # Wait for playback to finish
                time.sleep(0.1)
        else:
            print(f"Music Thread: *** No audio buffer to play for play_audio().")
            time.sleep(1)

    def change_track(self, audio_buffer):
        with self.lock:
            # Update the audio buffer with the new track
            self.audio_buffer = audio_buffer
            self.switching = True
            self.channel.fadeout(10)  # Fade out the audio

    def pause(self):
        with self.lock:
            self.channel.pause()

    def unpause(self):
        with self.lock:
            self.channel.unpause()

    def stop(self):
        self.running = False

def play_audio(audio_data, target_sample_rate=22050):
    # Detect the mime type of the audio data
    mime_type = magic.from_buffer(audio_data, mime=True)

    # Load the audio data into an AudioSegment
    if mime_type in ['audio/x-wav', 'audio/wav']:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format='wav')
    elif mime_type in ['audio/aac', 'audio/x-aac', 'audio/x-hx-aac-adts']:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format='aac')
        # Increase the volume by 10 dB
        audio_segment += 10
    else:
        raise ValueError(f"Unsupported audio format: {mime_type}")

    # Resample the audio to the target sample rate if necessary
    if audio_segment.frame_rate != target_sample_rate:
        audio_segment = audio_segment.set_frame_rate(target_sample_rate)

    # Export the resampled audio to a bytes buffer
    wav_io = io.BytesIO()
    audio_segment.export(wav_io, format='wav')
    wav_io.seek(0)
    audio_data = wav_io.read()

    # Load the WAV data into Pygame
    sound = pygame.mixer.Sound(io.BytesIO(audio_data))

    # Play the audio on the selected channel
    print(f"*** Playing audio on channel {audio_channel_speech.get_busy()}")
    audio_channel_speech.play(sound)

def playback(image, audio, duration):
    # play both audio and display image with audio blocking till finished
    if image and not args.norender:
        render(image, duration)

    if audio:
        play_audio(audio, 22050)
        print(f"Audio playback initiated.")

def get_audio_duration(audio_samples):
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_samples), format="wav")
    duration_ms = len(audio_segment)  # Duration in milliseconds
    duration_s = duration_ms / 1000.0  # Convert to seconds
    return duration_s

def main():
    ## Main routine
    bg_music = BackgroundMusic()
    bg_music.start()

    last_music_change = 0

    last_image_asset = None

    image_segment_number = 0
    audio_segment_number = 0
    last_sent_segments = time.time()

    audio_playback_complete_speech = True

    audio_latency_delta = 0
    image_latency_delta = 0

    stats_last_sent_ts = 0
    stats_last_sent_duration = 0.0

    end_of_stream = True
    last_sent_break = 0

    while True:
        # check if we will block, if so then don't and check events instead of pygame
        header_message = None
        segment_number = 0
        timestamp = 0
        mediaid = 0
        message = ""
        text = ""
        optimized_prompt = ""
        type = ""
        music = None
        audio = None
        image = None

        if socket.poll(timeout=0):
            # Receive the header message
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
                logger.info(f"Received {type} segment #{segment_number} {timestamp}: {mediaid} {len(text)} characters: {text[:20]}")

                try:
                    if args.nosave == False:
                        # Save audio asset
                        save_json(header_message, mediaid, type, segment_number)
                        save_asset(music, mediaid, segment_number, type)
                except Exception as e:
                    logger.error(f"Error saving music asset: {e}")

                # queue in music_buffer header and music
                music_buffer.put((header_message, music))

                print(f"M", end="", flush=True)

            if type == "speek":
                # Now, receive the binary audio data
                audio = socket.recv()

                # Print the header
                logger.info(f"Received {type} segment #{segment_number} {timestamp}: {mediaid} {len(text)} characters: {text[:20]}")

                # queue the header and audio together
                audio_buffer.put((header_message, audio))

                try:
                    if args.nosave == False:
                        save_json(header_message, mediaid, type, segment_number)
                        save_asset(audio, mediaid, segment_number, type)
                except Exception as e:
                    logger.error(f"Error saving audio asset: {e}")

                print(f"S", end="", flush=True)

            ## Image
            if type == "image":
                # Now, receive the binary audio data
                image = socket.recv()

                # Print the header
                logger.info(f"Received image segment {type} #{segment_number} {timestamp}: {mediaid} {len(text)} characters: {text[:20]}")

                try:
                    # Convert the bytes back to a PIL Image object
                    image = Image.open(io.BytesIO(image))
                    if args.show_ascii_art:
                        payload_hex = image_to_ascii(image)
                        print(f"\n{payload_hex}\n", flush=True)

                    logger.info(f"Image Prompt: {optimized_prompt[:20]}\Original Text: {text[:10]}...\nOriginal Question:{message[:10]}...")

                    # queue the header and image together
                    image_buffer.put((header_message, image))
                except Exception as e:
                    logger.error(f"Error converting image: {e}")

                try:
                    if args.nosave == False:
                        save_json(header_message, mediaid, type, segment_number)
                        save_asset(image, mediaid, segment_number, type)
                except Exception as e:
                    logger.error(f"Error saving image asset: {e}")

                print(f"I", end="", flush=True)
        else:
            ## No ZMQ message available, check for events
            print(f".", end="", flush=True)

        worked = False
        ## Update the image if we are rendering during speaking
        if not audio_playback_complete_speech:
            update_image(1)
            worked = True

        # No message available, check for events
        if pygame.event.peek(AUDIO_END_EVENT_SPEECH):
            for event in pygame.event.get(AUDIO_END_EVENT_SPEECH):
                if event.type == AUDIO_END_EVENT_SPEECH:
                    print(f"X", end="", flush=True)
                    audio_playback_complete_speech = True
                else:
                    logger.error(f"Unknown event on get event: {event}")
            worked = True

        # Send status of last playback time for the audio and image assets
        status = {}
        status["timestamp"] = time.time()
        status["audio_segment_number"] = audio_segment_number
        status["image_segment_number"] = image_segment_number
        status["audio_playback_complete_speech"] = audio_playback_complete_speech
        status["last_sent_segments"] = last_sent_segments
        status["last_music_change"] = last_music_change
        status["audio_buffer_size"] = audio_buffer.qsize()
        status["image_buffer_size"] = image_buffer.qsize()
        status["music_buffer_size"] = music_buffer.qsize()
        status["audio_playback_complete_speech"] = audio_playback_complete_speech
        status["audio_channel_speech_busy"] = audio_channel_speech.get_busy()
        status["audio_channel_music_busy"] = audio_channel_music.get_busy()
        status["audio_channel_music_volume"] = audio_channel_music.get_volume()
        status["audio_channel_speech_volume"] = audio_channel_speech.get_volume()
        status["audio_channel_music_switching"] = bg_music.switching
        status["audio_channel_music_complete"] = bg_music.complete
        status["audio_channel_music_running"] = bg_music.running
        status["audio_latency_delta"] = audio_latency_delta
        status["image_latency_delta"] = image_latency_delta
        status["eos"] = end_of_stream

        # calculate the duration field total for all the queued audio packets
        # don't remove them from the queue, just get the duration field and add it to the total
        total_duration = 0.0
        for audio_message, audio_asset in audio_buffer.queue:
            total_duration += audio_message["duration"]
        status["audio_buffer_duration"] = total_duration

        # send status to zmq output
        sent_delta = time.time() - last_sent_segments
        status["sent_delta"] = sent_delta
        if sent_delta < args.startup_delay:
            # check if we have 0 seconds of buffer, if so then fake it as 60 seconds and send it
            if status["audio_buffer_duration"] == 0.0:
                status["audio_buffer_duration"] = 60.0

        if time.time() - stats_last_sent_ts > args.stats_interval or stats_last_sent_duration != status["audio_buffer_duration"]:
            logger.info(f"Sending status: {status}")
            sender.send_json(status)
            stats_last_sent_ts = time.time()
            stats_last_sent_duration = status["audio_buffer_duration"]

        ## get an audio sample and header, get the text field from it, then get an image and header and burn in the text from the audio header to the image and render it while playing the audio
        if args.nobuffer and args.norender and not audio_buffer.empty() and audio_playback_complete_speech:
            audio_message, audio_asset = audio_buffer.get()
            text = audio_message["text"]
            duration = audio_message["duration"]
            optimized_prompt = text
            if 'optimized_text' in audio_message:
                optimized_prompt = audio_message["optimized_text"]
            else:
                optimized_prompt = text
            audio_playback_complete_speech = False
            playback(None, audio_asset, duration)
            last_sent_segments = time.time()
            audio_segment_number = audio_message["segment_number"]
            logger.info(f"Sent audio segment #{audio_message['segment_number']} at timestamp {audio_message['timestamp']}")
            worked = True
        elif not audio_buffer.empty() and not image_buffer.empty() and audio_playback_complete_speech:
            audio_message, audio_asset = audio_buffer.get()
            image_message, image_asset = image_buffer.get()

            image_segment_number = image_message["segment_number"]
            audio_segment_number = audio_message["segment_number"]
            last_sent_segments = time.time()

            duration = audio_message["duration"]

            if "eos" in audio_message and audio_message["eos"] == True:
                end_of_stream = True
                bg_music.unpause()
            else:
                end_of_stream = False
                bg_music.pause()

            text = audio_message["text"]
            optimized_prompt = text
            if 'optimized_text' in audio_message:
                optimized_prompt = audio_message["optimized_text"]

            if audio_message['timestamp'] < image_message['timestamp']:
                logger.debug(f"Audio segment #{audio_message['segment_number']} is older than image segment #{image_message['segment_number']}.")

            if audio_message['timestamp'] > image_message['timestamp']:
                logger.debug(f"Audio segment #{audio_message['segment_number']} is newer than image segment #{image_message['segment_number']}.")

            unique_image = True
            if "throttle" in image_message:
                if image_message["throttle"] == "true":
                    unique_image = False
            last_image_asset = image_asset.copy()
            banner_msg = f"{audio_message['username']} asked {audio_message['message'][:300]}"
            if 'eos' in audio_message and audio_message['eos'] == True:
                banner_msg = f"{audio_message['message']}"
            if args.burn_prompt:
                image_np = process_new_image(
                    image_asset, optimized_prompt, args, unique_image, banner_msg)
            else:
                image_np = process_new_image(image_asset, text, args, unique_image, banner_msg)

            # Play audio and display image
            try:
                audio_playback_complete_speech = False
                playback(image_np, audio_asset, duration)
            except Exception as e:
                logger.error(f"Error playing back audio and displaying image: {e}")

            worked = True
            audio_timestamp = audio_message["timestamp"]
            image_timestamp = image_message["timestamp"]
            audio_latency_delta = 0
            image_latency_delta = 0
            if audio_timestamp != 0:
                audio_latency_delta = int(round(time.time()*1000)) - int(audio_timestamp)
            if image_timestamp != 0:
                image_latency_delta = int(round(time.time()*1000)) - int(image_timestamp)
            logger.info(f"Sent audio segment #{audio_message['segment_number']} at timestamp {audio_message['timestamp']} with latency delta {audio_latency_delta} ms.")
            logger.info(f"Sent image segment #{image_message['segment_number']} at timestamp {image_message['timestamp']} with latency delta {image_latency_delta} ms.")
        else:
            # check last sent segments and if it's been more than 5 seconds, send a blank image and audio
            if time.time() - last_sent_segments > 15 and last_image_asset is not None:
                # confirm image_segment_number and audio_segment_number are both matching, else we need see if audio has buffered
                # samples to send and use the last image if there are no more image buffers
                if image_segment_number != audio_segment_number and audio_playback_complete_speech:
                    logger.debug(f"Image segment number {image_segment_number} does not match audio segment number {audio_segment_number}.")
                    if image_buffer.empty():
                        if not audio_buffer.empty():
                            logger.info(f"Warning: A/V Alignment offset, audio buffer is not empty, using last image and audio.")
                            audio_message, audio_asset = audio_buffer.get()
                            text = audio_message["text"]
                            optimized_prompt = text
                            duration = audio_message["duration"]
                            new_image = False
                            if 'optimized_text' in audio_message:
                                optimized_prompt = audio_message["optimized_text"]
                            else:
                                optimized_prompt = text
                            banner_msg = f"{audio_message['username']} asked {audio_message['message'][:300]}"
                            if 'eos' in audio_message and audio_message['eos'] == True:
                                banner_msg = f"{audio_message['message']}"
                            image_np = process_new_image(
                                last_image_asset, optimized_prompt, args, new_image, banner_msg)
                            audio_playback_complete_speech = False
                            playback(image_np, audio_asset, duration)
                            last_sent_segments = time.time()
                            audio_segment_number = audio_message["segment_number"]
                            logger.info(f"Sent audio segment #{audio_message['segment_number']} at timestamp {audio_message['timestamp']}")
                            worked = True
                            timestamp = audio_message["timestamp"]
                            latency_delta = 0
                            if timestamp != 0:
                                latency_delta = int(round(time.time()*1000)) - int(timestamp)
                            logger.info(
                                f"Sent audio segment #{audio_message['segment_number']} at timestamp {audio_message['timestamp']} with latency delta {latency_delta} ms.")
                else:
                    # check if we are in eos condition, if so send the last image seen with a special text burnin
                    if end_of_stream and time.time() - last_sent_break > 10: # send a break every 10 seconds
                        # check if any images are in the queue still
                        if not image_buffer.empty():
                            # get one image from the image buffer
                            image_message, image_asset = image_buffer.get()
                            image_segment_number = image_message["segment_number"]
                            last_image_asset = image_asset.copy()

                        logger.info(f"End of stream, sending last image with special text.")
                        # burn in special text
                        image_np = process_new_image(last_image_asset, "What would you like to talk about?", args, False, "Welcome, ask me a question or how to use the chat commands.")
                        playback(image_np, None, 0.0)
                        last_sent_break = time.time()
                        worked = True
                        logger.info(f"Sent last image segment #{image_segment_number} at timestamp {timestamp}")

        if not music_buffer.empty():
            if args.nomusic:
                music_message, music = music_buffer.get()
                while not music_buffer.empty():
                    music_message, music = music_buffer.get()
                    worked = True
                continue

            if not music_buffer.empty() and (last_music_change == 0 or time.time() - last_music_change > args.music_interval):
                music_message, music = music_buffer.get()
                logger.info(f"Loading Music: {music_message['mediaid']} {music_message['timestamp']} {music_message['segment_number']} {music_message['message'][:20]}")
                if last_music_change > 0:
                    logger.info(f"Last Music change was {time.time() - last_music_change} seconds since the last music change.")
                    bg_music.change_track(music)
                else:
                    # Load the initial music track
                    bg_music.change_track(music)

                last_music_change = time.time()
                worked = True
            else:
                logger.info(f"Skipping music because it's too soon since the last music change {time.time() - last_music_change}.")

        ## avoid busy loop
        if not worked:
            time.sleep(0.1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, required=False, default=6003, help="Port for receiving image as PIL numpy arrays")
    parser.add_argument("--input_host", type=str, required=False, default="127.0.0.1", help="Host for receiving image as PIL numpy arrays")
    parser.add_argument("--output_port", type=int, required=False, default=6004, help="Port for sending status of last image and audio segments")
    parser.add_argument("--output_host", type=str, required=False, default="127.0.0.1", help="Host for sending status of last image and audio segments")
    parser.add_argument("-ll", "--loglevel", type=str, default="info", help="Logging level: debug, info...")
    parser.add_argument("-f", "--freq", type=int, default=22050, help="Sampling frequency for audio playback")
    parser.add_argument("--burn_prompt", action="store_true", default=False, help="Burn in the prompt that created the image")
    parser.add_argument("--width", type=int, default=1920, help="Width of the output image")
    parser.add_argument("--height", type=int, default=1080, help="Height of the output image")
    parser.add_argument("--music_volume", type=float, default=0.50, help="Volume for music audio playback, defualt is 0.55")
    parser.add_argument("--speech_volume", type=float, default=0.75, help="Volume for speech audio playback")
    parser.add_argument("--music_interval", type=float, default=60, help="Interval between music changes")
    parser.add_argument("--nomusic", action="store_true", default=False, help="Disable music")
    parser.add_argument("--nosave", action="store_true", default=False, help="Don't save assets to disk")
    parser.add_argument("--norender", action="store_true", default=False, help="Disable rendering of images")
    parser.add_argument("--nobuffer", action="store_true", default=False, help="Disable buffering of images")
    parser.add_argument("--title", type=str, default="Groovy Life AI", help="Title for the window")
    parser.add_argument("--buffer_size", type=int, default=32768, help="Size of the buffer for images and audio")
    parser.add_argument("--show_ascii_art", action="store_true", default=False, help="Show images as ascii art")
    parser.add_argument("--startup_delay", type=float, default=30.0, help="Delay before sending status messages")
    parser.add_argument("--stats_interval", type=float, default=10.0, help="Interval between sending status messages")
    parser.add_argument("--sdl_audiodriver", type=str, default="GroovyLifeAI", help="SDL Audio Driver, default is GroovyLifeAI")
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

    #os.environ['SDL_AUDIODRIVER'] = args.sdl_audiodriver

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

    sender = context.socket(zmq.PUB)
    logger.info("connected to ZMQ out: %s:%d" % (args.output_host, args.output_port))
    sender.bind(f"tcp://{args.output_host}:{args.output_port}")

    pygame.init()
    pygame.mixer.init(frequency=args.freq, size=-16, channels=2, buffer=args.buffer_size, devicename=args.sdl_audiodriver)
    AUDIO_END_EVENT_MUSIC = pygame.USEREVENT + 1
    AUDIO_END_EVENT_SPEECH = pygame.USEREVENT + 2

    audio_channel_speech = pygame.mixer.Channel(1)
    audio_channel_speech.set_endevent(AUDIO_END_EVENT_SPEECH)
    audio_channel_speech.set_volume(args.speech_volume)

    audio_channel_music = pygame.mixer.Channel(2)
    audio_channel_music.set_volume(args.music_volume)

    audio_buffer = queue.Queue()
    music_buffer = queue.Queue()
    image_buffer = queue.Queue()

    main()
