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

def sync_media_buffers(audio_buffer, music_buffer, image_buffer, sender, logger, max_delay, buffer_delay):
    wall_clock_music = time.time() # Get the current wall clock time
    wall_clock_audio = time.time() # Get the current wall clock time
    wall_clock_image = time.time() # Get the current wall clock time
    master_clock = time.time() # Set the master clock to the current wall clock time
    last_audio_duration = 0
    last_music_duration = 0
    last_image_duration = 0
    last_image_timestamp = 0
    last_image_index = 0
    last_audio_timestamp = 0
    last_music_timestamp = 0
    found_image = False
    mediaid_music = ""
    mediaid_audio = ""
    mediaid_image = ""
    audio_buffering = 0
    waiting_for_audio = False
    first_run = True
    while True:
        try:
            if music_buffer.qsize() > 0 and time.time() - wall_clock_music > last_music_duration:
                music_message, music_asset = (music_buffer.queue[0] if music_buffer and music_buffer.qsize() > 0 else (None, None))
                if music_message['timestamp'] >= time.time() + buffer_delay:
                    music_buffer.get()
                    last_music_duration = music_message['duration']
                    sender.send_json(music_message, zmq.SNDMORE)
                    sender.send(music_asset)
                    logger.info(f"Sent music segment #{music_message['segment_number']} for {music_message['stream']} {music_message['timestamp']}']")
                    wall_clock_music = time.time()
                    mediaid_music = music_message['mediaid']
                    last_music_timestamp = music_message['timestamp']
                elif time.time() - wall_clock_music > buffer_delay + max_delay:
                    logger.warning(f"Dropping music buffer with latency of {time.time() - wall_clock_music} seconds.")
                    music_buffer.get()
                    wall_clock_music = time.time()
                else:
                    logger.info(f"Music buffer delay {time.time() - wall_clock_music} seconds.")

            if audio_buffer.qsize() > 0 and image_buffer.qsize() > 0:
                if time.time() - wall_clock_audio >= last_audio_duration:
                    audio_message, audio_asset = (audio_buffer.queue[0] if audio_buffer and audio_buffer.qsize() > 0 else (None, None))
                    image_message, image_asset = (image_buffer.queue[0] if image_buffer and image_buffer.qsize() > 0 else (None, None))
                    if found_image: # and audio_message['timestamp'] >= image_message['timestamp']:
                        # if we found an image then we can start sending audio or keep sending it
                        if first_run or (last_image_timestamp > 0 and (waiting_for_audio) or audio_message['timestamp'] >= last_image_timestamp):
                            audio_buffer.get()
                            last_audio_duration = audio_message['duration']
                            sender.send_json(audio_message, zmq.SNDMORE)
                            sender.send(audio_asset)
                            logger.info(f"Sent audio segment #{audio_message['segment_number']} for {audio_message['stream']} {audio_message['timestamp']}']")
                            wall_clock_audio = time.time()
                            mediaid_audio = audio_message['mediaid']
                            last_audio_timestamp = audio_message['timestamp']
                            waiting_for_audio = False
                            first_run = False
                        else:
                            logger.info(f"Audio buffer delay {time.time() - wall_clock_audio} seconds.")
                    else:
                        # if we have not found an image yet then we need to wait for one
                        if time.time() - audio_message['timestamp'] > max_delay:
                            logger.warning(f"Dropping audio buffer with latency of {time.time() - audio_message['timestamp']} seconds.")
                            audio_buffer.get()
                            wall_clock_audio = time.time()
                        else:
                            logger.info(f"Audio buffer delay {time.time() - wall_clock_audio} seconds.")

                    if not found_image or not waiting_for_audio or image_message['timestamp'] <= last_audio_timestamp:
                        # need to send image first before starting audio
                        image_buffer.get()
                        sender.send_json(image_message, zmq.SNDMORE)
                        sender.send(image_asset)
                        logger.info(f"Sending First image segment #{image_message['segment_number']} index {image_message['index']} for {image_message['stream']} {image_message['timestamp']}']")
                        last_image_timestamp = image_message['timestamp']
                        wall_clock_image = time.time()
                        found_image = True
                        mediaid_image = image_message['mediaid']
                        if 'index' in image_message:
                            last_image_index = image_message['index']
                        last_image_duration = min(5, len(image_message['text'].split(' ')) / 3) # calculate from words the delay
                        waiting_for_audio = True

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
        
        logger.info(f"Framesync: {mediaid}:{timestamp}#{segment_number}:{stream}:{duration} seconds {len(text)} characters {tokens} tokens: {text[:50]}")

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
        threading.Thread(target=sync_media_buffers, args=(audio_buffer, music_buffer, image_buffer, sender, logger, args.max_delay, args.buffer_delay), daemon=True).start()

    main()

