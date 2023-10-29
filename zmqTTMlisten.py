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

def get_audio_duration(audio_samples):
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_samples), format="wav")
    duration_ms = len(audio_segment)  # Duration in milliseconds
    duration_s = duration_ms / 1000.0  # Convert to seconds
    return duration_s

class BackgroundMusic(threading.Thread):
    def __init__(self):
        super().__init__()
        pygame.mixer.init(frequency=16000, size=-16, channels=1, buffer=1024)
        pygame.init()
        self.audio_buffer = None
        self.running = True
        self.lock = threading.Lock()  # Lock to synchronize access to audio_buffer

    def run(self):
        while self.running:
            with self.lock:
                if self.audio_buffer:
                    self.play_audio(self.audio_buffer)
                    self.audio_buffer = None  # Reset audio_buffer to prevent replaying the same buffer
            pygame.time.Clock().tick(1)  # Limit the while loop to 1 iteration per second

    def play_audio(self, audio_samples):
        audiobuf = io.BytesIO(audio_samples)
        if audiobuf:
            pygame.mixer.music.load(audiobuf)
            pygame.mixer.music.set_volume(args.volume)  # Set the volume
            pygame.mixer.music.play(-1)  # -1 instructs Pygame to loop the audio indefinitely

    def change_track(self, audio_buffer):
        with self.lock:
            pygame.mixer.music.stop()  # Stop the currently playing audio
            self.audio_buffer = audio_buffer

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
            # Receive the segment number (header) first
            segment_number = socket.recv_string()
            id = socket.recv_string()
            type = socket.recv_string()
            username = socket.recv_string()
            source = socket.recv_string()
            message = socket.recv_string()
            audio_text = socket.recv_string()
            duration = socket.recv_string()

            # Now, receive the binary audio data
            audio_samples = socket.recv()

            print(f"Received audio segment #%s of {duration} duration." % segment_number)

            # Check if we need to output to a file
            if args.save_file:
                audio_file = f"{args.output_directory}/{id.strip().replace(' ','')}/{segment_number}.wav"
                ## create directory recrusively for the file if it doesn't exist
                os.makedirs(os.path.dirname(audio_file), exist_ok=True)
                if args.audio_format == "wav":
                    with open(audio_file, 'wb') as f:
                        f.write(audio_samples)
                    print(f"Audio saved to {audio_file} as WAV")
                else:
                    with open(audio_file, 'wb') as f:
                        f.write(audio_samples)
                    print(f"Payload written to {audio_file}\n")

            # Convert the payload to its hex representation and display
            if args.show_hex:
                payload_hex = audio_samples.hex()
                print(f"Payload (Hex): {textwrap.fill(payload_hex, width=80)}\n")

            # Signal thread to play new audio, sleep for duration so we don't interupt it
            if audio_samples:
                bg_music.change_track(audio_samples)
                duration = get_audio_duration(audio_samples)
                time.sleep(duration)

            print(f"Audio #{segment_number} of {duration} duration recieved.\nAudio Text: {audio_text}\nMessage: {message}\n")

        except Exception as e:
            print(f"Error: %s" % str(e))
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, default=4002, required=False, help="Port for receiving audio numpy arrays")
    parser.add_argument("--input_host", type=str, default="127.0.0.1", required=False, help="Host for receiving audio input")
    parser.add_argument("--output_directory", default="music", type=str, help="Directory path to save the received wave files in")
    parser.add_argument("--save_file", action="store_true", help="Save the received audio as WAV files")
    parser.add_argument("--show_hex", action="store_true", help="Show the hex representation of the audio payload")
    parser.add_argument("--audio_format", type=str, choices=["wav", "raw"], default="wav", help="Audio format to save as. Choices are 'wav' or 'raw'. Default is 'wav'.")
    parser.add_argument("--volume", type=float, default=0.6, help="Playback volume (0.0 to 1.0, default is 0.6)")
    args = parser.parse_args()

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    print("connected to ZMQ in: %s:%d" % (args.input_host, args.input_port))
    socket.connect(f"tcp://{args.input_host}:{args.input_port}")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    main()

