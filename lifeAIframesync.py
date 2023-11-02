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
from queue import PriorityQueue
import hashlib

warnings.simplefilter(action='ignore', category=Warning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
from urllib3.exceptions import NotOpenSSLWarning
warnings.simplefilter(action='ignore', category=NotOpenSSLWarning)

def sync_media_buffers(audio_buffer, music_buffer, image_buffer, sender, logger, max_delay):
    master_clock = None  # Initialize the master clock to None
    mux_pq = PriorityQueue()

    while True:
        try:
            # Process music buffer
            if not music_buffer.empty():
                music_message, music_asset = music_buffer.get()
                sender.send_json(music_message, zmq.SNDMORE)
                sender.send(music_asset)
                logger.info(f"Sent music segment #{music_message['segment_number']} at timestamp {music_message['timestamp']}")

            if not image_buffer.empty():
                image_message, image_asset = image_buffer.get()
                if media_type == 'image':
                    sender.send_json(message, zmq.SNDMORE)
                    sender.send(asset)
                    logger.info(f"Sent {media_type} segment #{message['segment_number']} at timestamp {message['timestamp']}")
                    master_clock = message['timestamp']


            # Process audio and image buffers
            if not mux_pq.empty() or (not audio_buffer.empty()):
                if not audio_buffer.empty():
                    audio_message, audio_asset = audio_buffer.get()
                    mux_pq.put((audio_message['timestamp'], ('audio', audio_message, audio_asset)))
              
                # Send out assets in timestamp order, waiting for images to catch up to audio
                if not mux_pq.empty():
                    _, (media_type, message, asset) = mux_pq.get()
                    current_time = time.time()

                    if media_type == 'audio' and (master_clock is None or message['timestamp'] <= master_clock):
                        sender.send_json(message, zmq.SNDMORE)
                        sender.send(asset)
                        logger.info(f"Sent {media_type} segment #{message['segment_number']} at timestamp {message['timestamp']}")
                        master_clock = message['timestamp']
                        continue

                    # Check for delay
                    if current_time - message['timestamp'] > max_delay:
                        logger.warning(f"Dropping {media_type} segment #{message['segment_number']} due to high delay.")
                    else:
                        # queue the audio again
                        mux_pq.put((message['timestamp'], (media_type, message, asset)))

                    time.sleep(0.01)  # Sleep to prevent CPU overuse

        except Exception as e:
            logger.error(f"Error while syncing media buffers: {e}")

def main():
    while True:
        header_message = receiver.recv_json()
        asset = receiver.recv()

        # fill out variables from header_message
        if "segment_number" not in header_message:
            logger.error(f"Framesync: No segment number in header message: {header_message}")
            continue

        segment_number = header_message["segment_number"]
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

        tokens = 0
        if "tokens" in header_message:
            tokens = header_message['tokens']
        
        # md5sum text
        md5text = hashlib.md5(text.encode('utf-8')).hexdigest()
        clean_text = text[:30].replace('\n', ' ').replace('\t','').strip()
        md5sig = "none"
        if 'md5sum' in header_message:
            md5sig = header_message['md5sum']
        segment_index = 0
        if 'index' in header_message:
            segment_index = header_message['index']
        logger.info(f"Framesync: {stream} #{segment_number}/{segment_index} - {timestamp}: {mediaid} {duration} seconds {len(text)} characters {tokens} tokens {md5sig}/{md5text}: {clean_text}")

        if args.passthrough:
            sender.send_json(header_message, zmq.SNDMORE)
            sender.send(asset)
            text = text.replace('\n', ' ').replace('  ','').strip()
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
        if type == "speek":
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

        logger.info(f"Framesync: {type} buffered segment #{segment_number} timestamp {timestamp}")
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, default=6002, required=False, help="Port for receiving text input")
    parser.add_argument("--output_port", type=int, default=6003, required=False, help="Port for sending image output")
    parser.add_argument("--input_host", type=str, default="127.0.0.1", required=False, help="Port for receiving text input")
    parser.add_argument("--output_host", type=str, default="127.0.0.1", required=False, help="Port for sending image output")
    parser.add_argument("-ll", "--loglevel", type=str, default="info", help="Logging level: debug, info...")
    parser.add_argument("--max_delay", type=int, default=60, help="Maximum allowed delay in seconds for image frames before they are dropped")
    parser.add_argument("--passthrough", action="store_true", help="Pass through all messages without synchronization")
    parser.add_argument("--max_segment_diff", type=int, default=2, help="Maximum allowed segment number difference before older segments are skipped")
    parser.add_argument("--buffer_delay", type=int, default=0, help="Delay in seconds to buffer messages before sending them out")
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
        threading.Thread(target=sync_media_buffers, args=(audio_buffer, music_buffer, image_buffer, sender, logger, args.max_delay), daemon=True).start()

    main()

