#!/usr/bin/env python

## Life AI Framesync module
#
# Chris Kennedy 2023 (C) GPL
#
# Free to use for any use as in truly free software
# as Richard Stallman intended it to be.
#

import zmq
import argparse
import warnings
import urllib3
import logging
import time
import queue
import threading

warnings.simplefilter(action='ignore', category=Warning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
from urllib3.exceptions import NotOpenSSLWarning
warnings.simplefilter(action='ignore', category=NotOpenSSLWarning)

def process_and_send_buffers():
    while True:
        # Wait until we have at least one of each type
        if not audio_buffer.empty() and not music_buffer.empty() and not image_buffer.empty():
            # Get the next messages from each buffer
            audio_message, audio_asset = audio_buffer.get()
            music_message, music_asset = music_buffer.get()
            image_message, image_asset = image_buffer.get()

            # Synchronization logic here based on segment_number and timestamp
            # Example: use the earliest timestamp of the three as the reference

            # Send the music asset continuously until a new one comes in
            while True:
                sender.send_json(music_message, zmq.SNDMORE)
                sender.send(music_asset)

                # Check if there's a new music message in the buffer
                if not music_buffer.empty():
                    music_message, music_asset = music_buffer.get()
                    break

            # Send the audio message once
            sender.send_json(audio_message, zmq.SNDMORE)
            sender.send(audio_asset)

            # Send the image message once
            sender.send_json(image_message, zmq.SNDMORE)
            sender.send(image_asset)

            logger.info(f"Framesync: sent audio, music, and image for segment #{audio_message['segment_number']}")

def sync_media_buffers(sender, logger, max_delay):
    # Dictionaries to hold the latest assets by segment number
    latest_assets = {'audio': {}, 'music': {}, 'image': {}}

    while True:
        try:
            # Check for the smallest segment number available in all buffers
            if audio_buffer.qsize() > 0 and music_buffer.qsize() > 0 and image_buffer.qsize() > 0:
                # Get the next messages from each buffer, but do not remove them yet
                audio_message, audio_asset = audio_buffer.queue[0]
                music_message, music_asset = music_buffer.queue[0]
                image_message, image_asset = image_buffer.queue[0]

                # Extract segment numbers
                audio_segment = audio_message['segment_number']
                music_segment = music_message['segment_number']
                image_segment = image_message['segment_number']

                # Synchronize based on the segment number
                if audio_segment == music_segment == image_segment:
                    # Remove items from the buffer
                    audio_buffer.get()
                    music_buffer.get()
                    image_buffer.get()

                    # Check delay for image and drop if necessary
                    if (time.time() - float(image_message['timestamp'])) > max_delay:
                        logger.warning(f"Dropping image frame for segment {image_segment} due to high delay.")
                        continue

                    # Send the synchronized assets
                    for message, asset in [(audio_message, audio_asset), (music_message, music_asset), (image_message, image_asset)]:
                        sender.send_json(message, zmq.SNDMORE)
                        sender.send(asset)
                        logger.info(f"Sent synchronized segment #{message['segment_number']} for {message['stream']}")

                # Implement your logic if the segment numbers do not match
                # e.g., skip the oldest segment, or wait for the next iteration
                else:
                    # Log the mismatch and wait for the next iteration
                    logger.warning("Segment numbers do not match for synchronization. Waiting for the next segments.")
                    time.sleep(0.1)  # Avoid tight loop if waiting is necessary

        except Exception as e:
            logger.error(f"Error while syncing media buffers: {e}")

def sync_media_buffers(audio_buffer, music_buffer, image_buffer, sender, logger, max_delay):
    # Dictionaries to hold the latest assets by segment number
    latest_assets = {'audio': {}, 'music': {}, 'image': {}}

    while True:
        try:
            # Check for the smallest segment number available in all buffers
            if audio_buffer.qsize() > 0 and music_buffer.qsize() > 0 and image_buffer.qsize() > 0:
                # Get the next messages from each buffer, but do not remove them yet
                audio_message, audio_asset = audio_buffer.queue[0]
                music_message, music_asset = music_buffer.queue[0]
                image_message, image_asset = image_buffer.queue[0]

                # Extract segment numbers
                audio_segment = audio_message['segment_number']
                music_segment = music_message['segment_number']
                image_segment = image_message['segment_number']

                # Synchronize based on the segment number
                if audio_segment == music_segment == image_segment:
                    # Remove items from the buffer
                    audio_buffer.get()
                    music_buffer.get()
                    image_buffer.get()

                    # Check delay for image and drop if necessary
                    if (time.time() - float(image_message['timestamp'])) > max_delay:
                        logger.warning(f"Dropping image frame for segment {image_segment} due to high delay.")
                        continue

                    # Send the synchronized assets
                    for message, asset in [(audio_message, audio_asset), (music_message, music_asset), (image_message, image_asset)]:
                        sender.send_json(message, zmq.SNDMORE)
                        sender.send(asset)
                        logger.info(f"Sent synchronized segment #{message['segment_number']} for {message['stream']}")

                else:
                    # If the segments do not match, find the one with the earliest timestamp to requeue
                    timestamps = [audio_message['timestamp'], music_message['timestamp'], image_message['timestamp']]
                    min_timestamp = min(timestamps)
                    min_index = timestamps.index(min_timestamp)
                    
                    # Requeue the message with the earliest timestamp
                    if min_index == 0:  # Audio has the earliest timestamp
                        audio_buffer.get()  # Remove the message from the queue
                        audio_buffer.put((audio_message, audio_asset))  # Requeue it
                    elif min_index == 1:  # Music has the earliest timestamp
                        music_buffer.get()  # Remove the message from the queue
                        music_buffer.put((music_message, music_asset))  # Requeue it
                    elif min_index == 2:  # Image has the earliest timestamp
                        image_buffer.get()  # Remove the message from the queue
                        image_buffer.put((image_message, image_asset))  # Requeue it
                        
                    logger.warning("Segment numbers do not match for synchronization. Requeued the earliest segment.")

        except Exception as e:
            logger.error(f"Error while syncing media buffers: {e}")

def main():
    if not args.passthrough:
        # Start a thread for processing and sending buffers
        threading.Thread(target=process_and_send_buffers, daemon=True).start()

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
        asset = receiver.recv()

        if args.passthrough:
            stream = header_message['stream']
            text = header_message['text']
            duration = 0
            if "duration" in header_message:
                duration = "%d" % int(header_message["duration"])
            timestamp = 0
            if "timestamp" in header_message:
                timestamp = "%d" % int(header_message["timestamp"])
            mediaid = "none"
            if "mediaid" in header_message:
                mediaid = header_message['mediaid']
            sender.send_json(header_message, zmq.SNDMORE)
            sender.send(asset)
            logger.info(f"Framesync: mediaid {mediaid} {timestamp} sent segment #{segment_number} {stream} {duration}: {text}")
            continue

        if "stream" not in header_message:
            logger.error(f"Framesync: No stream type in header message: {header_message}")
            continue

        timestamp = 0
        duration = 0
        is_audio = False
        is_music = False
        is_image = False
        if 'duration' in header_message:
            duration = "%d" % int(header_message["duration"])
        if 'timestamp' in header_message:
            timestamp = "%d" % int(header_message["timestamp"])

        type = header_message["stream"]
        if type == "speech":
            is_audio = True
        elif type == "music":
            is_music = True
        elif type == "image":
            is_image = True
        else:
            logger.error(f"Unknown type: {type}")
            continue

        ## buffer messages till we have one audio, one music, and one image
        ## once we have all three, send them out in order
        ## use the timestamp and segment number to sync them up
        ## duplicate frames if necessary
        ## duplicate the music and send it continusouly till a new one comes in too
        ## do not repeat the audio though, just send it once
        # Add the received messages to the appropriate buffers
        if is_audio:
            audio_buffer.put((header_message, asset))
        elif is_music:
            music_buffer.put((header_message, asset))
        elif is_image:
            image_buffer.put((header_message, asset))
        
        sender.send_json(header_message, zmq.SNDMORE)
        sender.send(asset)

        logger.info("Framesync: sent segment #%s" % segment_number)
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, default=6002, required=False, help="Port for receiving text input")
    parser.add_argument("--output_port", type=int, default=6003, required=False, help="Port for sending image output")
    parser.add_argument("--input_host", type=str, default="127.0.0.1", required=False, help="Port for receiving text input")
    parser.add_argument("--output_host", type=str, default="127.0.0.1", required=False, help="Port for sending image output")
    parser.add_argument("-ll", "--loglevel", type=str, default="info", help="Logging level: debug, info...")
    parser.add_argument("--max_delay", type=int, default=5, help="Maximum allowed delay in seconds for image frames before they are dropped")
    parser.add_argument("--passthrough", action="store_true", help="Pass through all messages without synchronization")
    parser.add_argument("--max_segment_diff", type=int, default=2, help="Maximum allowed segment number difference before older segments are skipped")
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
    logging.basicConfig(filename=f"logs/framesync-{log_id}.log", level=LOGLEVEL)
    logger = logging.getLogger('Framesync')

    ch = logging.StreamHandler()
    ch.setLevel(LOGLEVEL)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    logger.info("connected to ZMQ in: %s:%d" % (args.input_host, args.input_port))
    receiver.bind(f"tcp://{args.input_host}:{args.input_port}")

    sender = context.socket(zmq.PUB)
    logger.info("binded to ZMQ out: %s:%d" % (args.output_host, args.output_port))
    sender.bind(f"tcp://{args.output_host}:{args.output_port}")

    if not args.passthrough:
        # Define the buffer queues for each media type
        audio_buffer = queue.Queue()
        music_buffer = queue.Queue()
        image_buffer = queue.Queue()

        # Start the sync_media_buffers function in a separate thread
        threading.Thread(target=sync_media_buffers, args=(audio_buffer, music_buffer, image_buffer, sender, logger), daemon=True).start()

    main()

