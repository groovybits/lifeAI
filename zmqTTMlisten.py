#!/usr/bin/env python

## Life AI Text to Music listener ZMQ client
#
# Chris Kennedy 2023 (C) GPL
#
# Free to use for any use as in truly free software
# as Richard Stallman intended it to be.
#

import time
import io
import zmq
import argparse
import textwrap
import soundfile as sf
import pygame
import re
import os
import sys
import threading
from pydub import AudioSegment
import logging

def get_audio_duration(audio_samples):
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_samples), format="wav")
    duration_ms = len(audio_segment)  # Duration in milliseconds
    duration_s = duration_ms / 1000.0  # Convert to seconds
    return duration_s

class BackgroundMusic(threading.Thread):
    def __init__(self):
        super().__init__()
        pygame.mixer.init(frequency=32000, size=-16, channels=args.channels, buffer=args.buffer_size)
        pygame.init()
        self.audio_buffer = None
        self.running = True
        self.lock = threading.Lock()  # Lock to synchronize access to audio_buffer

    def run(self):
        while self.running:
            with self.lock:
                if self.audio_buffer:
                    self.play_audio(self.audio_buffer)
            pygame.time.Clock().tick(1)  # Limit the while loop to 1 iteration per second

    def play_audio(self, audio_samples):
        audiobuf = io.BytesIO(audio_samples)
        if audiobuf:
            pygame.mixer.music.load(audiobuf)
            pygame.mixer.music.set_volume(args.volume)  # Set the volume
            pygame.mixer.music.play(fade_ms=100)
            logger.info(f"Playing music segment")
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(1)

    def change_track(self, audio_buffer):
        with self.lock:
            logger.info(f"Changing music track")
            pygame.mixer.music.stop()  # Stop the currently playing audio
            self.audio_buffer = audio_buffer
            pygame.mixer.music.fadeout(100)  # Fade out the audio

    def stop(self):
        self.running = False
        pygame.mixer.music.stop()

def main():
    # Instantiate and start the background music thread
    bg_music = BackgroundMusic()
    bg_music.start()

    audio_samples = None
    while True:
        try:
            header_message = socket.recv_json()
           
            # fill out the variables from the header
            segment_number = header_message["segment_number"]
            mediaid = header_message["mediaid"]

            # Now, receive the binary audio data
            audio_samples = socket.recv()

            if header_message['stream'] != "music":
                logger.debug(f"Received non-music stream {header_message['stream']}")
                continue

            duration = header_message["duration"]

            logger.debug(f"Received music segment mediaid: {header_message}")
            logger.info(f"Received music segment #{segment_number} mediaid: {mediaid}")

            # Signal thread to play new audio, sleep for duration so we don't interupt it
            if audio_samples:
                bg_music.change_track(audio_samples)
                duration = get_audio_duration(audio_samples)
                time.sleep(duration)
        except Exception as e:
            logger.error(f"Error: %s" % str(e))
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, default=6003, required=False, help="Port for receiving audio numpy arrays")
    parser.add_argument("--input_host", type=str, default="127.0.0.1", required=False, help="Host for receiving audio input")
    parser.add_argument("--output_directory", default="music", type=str, help="Directory path to save the received wave files in")
    parser.add_argument("--audio_format", type=str, choices=["wav", "raw"], default="wav", help="Audio format to save as. Choices are 'wav' or 'raw'. Default is 'wav'.")
    parser.add_argument("--volume", type=float, default=0.30, help="Playback volume (0.0 to 1.0, default is 0.30)")
    parser.add_argument("-ll", "--loglevel", type=str, default="info", help="Logging level: debug, info...")
    parser.add_argument("--buffer_size", type=int, default=32786, help="Audio buffer size (default is 32786)")
    parser.add_argument("--channels", type=int, default=2, help="Number of audio channels (default is 2)")

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
    logging.basicConfig(filename=f"logs/zmqTTMlisten-{log_id}.log", level=LOGLEVEL)
    logger = logging.getLogger('TTMlistener')

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

    main()

