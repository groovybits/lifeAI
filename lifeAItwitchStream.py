#!/usr/bin/env python

## Life AI Send audio/video output to Twitch Stream RTMP

# Chris Kennedy 2023 (C) GPL
#
# Free to use for any use as in truly free software
# as Richard Stallman intended it to be.
#

from dotenv import load_dotenv
import os
import argparse
import zmq
import time
import threading
import numpy as np
import cv2
from queue import Queue
from pydub import AudioSegment
from twitchstream.outputvideo import TwitchBufferedOutputStream
from PIL import Image
import io

load_dotenv()

def draw_default_frame():
    # Create a black image
    default_img = np.zeros((args.width, args.height, 3), dtype=np.uint8)

    # Text settings
    text = "The Groovy AI Bot"
    font_scale = 2
    font_thickness = 4
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)  # White color

    # Calculate text size to center the text
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    x_centered = (default_img.shape[1] - text_width) // 2
    y_centered = (default_img.shape[0] + text_height) // 2

    # Draw the text onto the image
    cv2.putText(default_img, text, (x_centered, y_centered), font, font_scale, color, font_thickness, lineType=cv2.LINE_AA)

    default_img = Image.fromarray(cv2.cvtColor(default_img, cv2.COLOR_BGR2RGB))
    return default_img

    return None

class TwitchStreamer:
    def __init__(self, twitch_stream_key, width, height):
        self.data_queue = Queue()
        self.stop_event = threading.Event()
        self.twitch_stream_key = twitch_stream_key
        self.width = width
        self.height = height
        self.videostream = None
        self.video_writer = None
        self.audio_segment = AudioSegment.empty()
        self.last_video_frame = None
        self.last_audio_left = None
        self.last_audio_right = None
        self.audio_frame_counter = 0

    def add_data(self, data):
        self.data_queue.put(data)

    def stop_streaming(self):
        self.stop_event.set()
        if self.video_writer is not None:
            self.video_writer.release()

    def stream(self):
        with TwitchBufferedOutputStream(
                twitch_stream_key=self.twitch_stream_key,
                width=self.width,
                height=self.height,
                fps=30,
                enable_audio=True,
                verbose=True) as self.videostream:
            
            last_frame_time = time.time()
            first_frame = False
            default_image = draw_default_frame()
            limage = default_image

            while not self.stop_event.is_set():
                image = None
                audio = None

                if not self.data_queue.empty():
                    data = self.data_queue.get()

                    image = data['image']
                    audio = data['audio']      

                    if image != None:
                        first_frame = True
                        last_frame_time = time.time()

                        limage = image
                        #limage = np.array(limage)
                        #limage = cv2.cvtColor(limage, cv2.COLOR_RGB2BGR)
                else:
                    if not first_frame or time.time() - last_frame_time > 0.03:
                        image = limage
                        last_frame_time = time.time()
                    else:
                        time.sleep(.1)

                ## Send Video
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
                #print(f"\n===\nSending image type to video #{type(image)}")
                #image = np.array(image)
                #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                #assert image.shape == (args.width, args.height, 3), f"Unexpected shape: {image.shape}"
                #print(f"Sending converted image type to video #{type(image)}")
                image = np.array(image)
                self.videostream.send_video_frame(image)

                ## Send Audio
                if audio is not None:
                    self.videostream.send_audio(audio, audio)

                    with TwitchBufferedOutputStream(
                            twitch_stream_key=self.twitch_stream_key,
                            width=self.width,
                            height=self.height,
                            fps=30,
                            enable_audio=True,
                            verbose=True) as self.videostream:
                
                        last_frame_time = time.time()
                        first_frame = False
                        default_image = draw_default_frame()
                        limage = default_image

                        while not self.stop_event.is_set():
                            image = None
                            audio = None

                            if not self.data_queue.empty():
                                data = self.data_queue.get()

                                image = data['image']
                                audio = data['audio']      

                                image = np.array(image)

                                if image != None:
                                    first_frame = True
                                    last_frame_time = time.time()

                                    limage = image
                                    #limage = np.array(limage)
                                    #limage = cv2.cvtColor(limage, cv2.COLOR_RGB2BGR)
                            else:
                                if not first_frame or time.time() - last_frame_time > 0.03:
                                    image = limage
                                    last_frame_time = time.time()
                                else:
                                    time.sleep(.1)

                            ## Send Video
                            #image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
                            #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            #assert image.shape == (args.width, args.height, 3), f"Unexpected shape: {image.shape}"
                            image = np.array(image)
                             #print(f"Sending converted image type to video #{type(image)}")
                            self.videostream.send_video_frame(image)

                            ## Send Audio
                            if audio is not None:
                                self.videostream.send_audio(audio, audio)

## Allows async running in thread for events
def main():
    ## Twitch streaming
    streamer = None
    stream_id = os.environ['TWITCH_STREAM_KEY']
    if stream_id == "":
        stream_id = args.twitchstreamkey
    
    ## Setup Twitch streaming thread
    if stream_id != "":
        streamer = TwitchStreamer(args.twitchstreamkey, args.width, args.height)
        streaming_thread = threading.Thread(target=streamer.stream)
        streaming_thread.start()

        last_image = draw_default_frame()

        while not exit_program:
            # Receive the segment number (header) first
            segment_number = image_socket.recv_string()
            id = image_socket.recv_string()
            type = image_socket.recv_string()
            username = image_socket.recv_string()
            source = image_socket.recv_string()
            message = image_socket.recv_string()
            image_prompt = image_socket.recv_string()
            image_text = image_socket.recv_string()
            # Now, receive the binary audio data
            image = image_socket.recv()

            # Convert to PIL Image
            image = Image.open(io.BytesIO(image))

            # Convert PIL to numpy array  
            image = np.asarray(image)

            # Convert RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            print(f"\n===\nReceived image type #{type(image)}\n")
            if image:
                # Convert the bytes back to a PIL Image object
                #image = Image.open(io.BytesIO(image))
                #image = np.array(image)
                #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR for OpenCV
                #image = np.array(image)
                #image = np.frombuffer(image, np.uint8)
                #image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decode the image
                last_image = image
            else:
                print(f"Twitch Stream: image is empty")

            ## find audio in audio queue
            aid = audio_socket.recv_string()
            atype = audio_socket.recv_string()
            ausername = audio_socket.recv_string()
            asource = audio_socket.recv_string()
            amessage = audio_socket.recv_string()
            audio_text = audio_socket.recv_string()
            duration = audio_socket.recv_string()
            audio = audio_socket.recv()

            streamer.add_data({'image': last_image, 'audio': audio})  # Add data to be streamed

            # Print the header
            print(f"Received image segment #{segment_number}")


        # wait till finished running thread
        streaming_thread.join()
    else:
        print("No Twitch stream key provided, skipping Twitch streaming")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_input_port", type=int, required=False, default=3003, help="Port for receiving image as PNG ")
    parser.add_argument("--image_input_host", type=str, required=False, default="127.0.0.1", help="Host for receiving image as PNG")
    parser.add_argument("--audio_input_port", type=int, required=False, default=2001, help="Port for receiving audio as WAV ")
    parser.add_argument("--audio_input_host", type=str, required=False, default="127.0.0.1", help="Host for receiving audio as WAV")
    parser.add_argument("--twitchstreamkey", type=str, default="", required=False, help="Twitch stream key")
    parser.add_argument("--width", type=int, default=1920, help="Width of the output image")
    parser.add_argument("--height", type=int, default=1080, help="Height of the output image")
    args = parser.parse_args()

    context = zmq.Context()
    image_socket = context.socket(zmq.PULL)
    print("connected to ZMQ Images in: %s:%d" % (args.image_input_host, args.image_input_port))
    image_socket.connect(f"tcp://{args.image_input_host}:{args.image_input_port}")
    #image_socket.setsockopt_string(zmq.SUBSCRIBE, "")

    audio_socket = context.socket(zmq.PULL)
    print("connected to ZMQ Audio in: %s:%d" % (args.audio_input_host, args.audio_input_port))
    audio_socket.connect(f"tcp://{args.audio_input_host}:{args.audio_input_port}")
    #image_socket.setsockopt_string(zmq.SUBSCRIBE, "")

    exit_program = False
    main()
