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
import numpy as np
import cv2
from twitchstream.outputvideo import TwitchBufferedOutputStream
from PIL import Image
import librosa
import io
import soundfile as sf
import wave
import numpy as np
from pydub import AudioSegment

load_dotenv()

def chunk_audio(audio_data, chunk_size):
    num_chunks = len(audio_data) // chunk_size  # Use integer division
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        yield audio_data[start_idx:end_idx]

def upsample_audio(audio_data, original_sr, target_sr):
    audio_buffer = io.BytesIO(audio_data)
    with wave.open(audio_buffer, 'rb') as wave_file:
        n_frames = wave_file.getnframes()
        audio_frames = wave_file.readframes(n_frames)
        audio_array = np.frombuffer(audio_frames, dtype=np.int16)
        resampled_audio = librosa.resample(audio_array.astype(np.float32), orig_sr=original_sr, target_sr=target_sr)
    return resampled_audio

def draw_default_frame():
    # Create a black image with white text
    default_img = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    # fill in frame with black
    default_img[:] = (0, 0, 0)

    # Text settings
    text = "The Groovy AI Bot"
    font_scale = 6
    font_thickness = 12
    font = cv2.FONT_HERSHEY_DUPLEX
    color = (255, 255, 255)  # White color

    # Calculate text size to center the text
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    x_centered = (args.width - text_width) // 2
    y_centered = (args.height + text_height) // 2

    # Draw the text onto the image
    cv2.putText(default_img, text, (x_centered, y_centered), font, font_scale, color, font_thickness, lineType=cv2.LINE_AA)

    print(f"\nBlackframe: Image dimensions: {default_img.shape}")
    print(f"\nBlackframe: Text position: ({x_centered}, {y_centered})")
    
    return default_img

def main():
    ## Twitch streaming
    if stream_id != "":
        with TwitchBufferedOutputStream(
                            twitch_stream_key=args.twitchstreamkey,
                            width=args.width,
                            height=args.height,
                            fps=args.fps,
                            enable_audio=True,
                            verbose=True) as videostream:
            
            last_image_time = time.time()
            last_audio_time = time.time()
            first_frame = False
            frequency = 100
            last_phase = 0

            last_image = draw_default_frame()
            videostream.send_video_frame(last_image)
            t = np.arange(0, 1.0, 1.0/args.samplerate)  # 1 second of audio
            frequency = 440  # Frequency in Hz (A4 note)
            audio_tone = np.sin(2 * np.pi * frequency * t)

            videostream.send_audio(audio_tone, audio_tone)

            # Main frame server
            while not exit_program:
                ## serve the video frame
                if videostream.get_video_frame_buffer_state() < args.fps:
                    # Receive the segment number (header) first
                    segment_number = image_socket.recv_string()
                    id = image_socket.recv_string()
                    type = image_socket.recv_string()
                    username = image_socket.recv_string()
                    source = image_socket.recv_string()
                    message = image_socket.recv_string()
                    image_prompt = image_socket.recv_string()
                    image_text = image_socket.recv_string()
                    image = image_socket.recv()

                    if image:
                       # 2. Convert and possibly resize the image data
                        # Convert the byte data to a NumPy array
                        image_array = np.frombuffer(image, dtype=np.uint8)
                        # Decode the image data
                        image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
                        # Convert the image from BGR to RGB (OpenCV loads images in BGR by default)
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        # Check if the image needs to be resized
                        desired_dimensions = (args.width, args.height)  # Adjust as necessary
                        if image_rgb.shape[:2] != desired_dimensions:
                            image = cv2.resize(image_rgb, (desired_dimensions[1], desired_dimensions[0]))
                        else:
                            image = image_rgb

                        # Inside the if image: block
                        image_rgb_resized = cv2.resize(image_rgb, (args.width, args.height))
                        assert image_rgb_resized.shape == (args.height, args.width, 3), f"Unexpected frame shape: {image_rgb_resized.shape}"
                        videostream.send_video_frame(image_rgb_resized)
                        image = image_rgb_resized

                        last_image = image
                        videostream.send_video_frame(image)
                    else:
                        image = last_image
                        videostream.send_video_frame(image)
                else:
                    print(f"Twitch Stream: video frame buffer is full:", videostream.get_video_frame_buffer_state())
                    time.sleep(0.001)

                last_audio_time = time.time()
                if videostream.get_audio_buffer_state() < args.fps:
                    ## find audio in audio queue
                    aid = audio_socket.recv_string()
                    atype = audio_socket.recv_string()
                    ausername = audio_socket.recv_string()
                    asource = audio_socket.recv_string()
                    amessage = audio_socket.recv_string()
                    audio_text = audio_socket.recv_string()
                    duration = audio_socket.recv_string()
                    audio = audio_socket.recv()

                    if audio:
                        # Create a BytesIO object from the audio data
                        #audiobuf = io.BytesIO(audio)
    
                        # Use soundfile to read the audio data from the BytesIO object
                        #audio_data, original_sr = sf.read(audiobuf)

                        # Assuming audio is your audio data in byte format, and original_sr is the original sample rate
                        #audio_data, original_sr = librosa.load(io.BytesIO(audio), sr=args.samplerate, mono=True)
                        original_sr = 16000
                        upsampled_audio_data = upsample_audio(audio, 16000, 44100)
                        chunk_size = 44100 // args.fps  # Calculate chunk size based on the new sample rate and frame rate
                        print(f"\nTwitch Stream: upsample audio data: chunks {chunk_size} original_sr {original_sr}")
                        #upsampled_audio_data = audio
                        #chunk_size = 16000 // args.fps
                        chunk_count = 0
                        for chunk in chunk_audio(upsampled_audio_data, chunk_size):
                            chunk_count += 1
                            videostream.send_audio(chunk, chunk)
                        print(f"\nTwitch Stream: audio chunks sent: {chunk_count}")
                    else:
                        print(f"Twitch Stream: audio is empty:", audio.size())
                        # send a tone
                        phase = last_phase + frequency * 2 * np.pi / args.samplerate
                        last_phase = phase
                        audio_tone = np.sin(phase * np.arange(args.samplerate // args.fps)).astype(np.float32)
                        videostream.send_audio(audio_tone, audio_tone)
                else:
                    print(f"Twitch Stream: audio buffer is full:", videostream.get_audio_buffer_state())
                    time.sleep(0.001)
    else:
        print("No Twitch stream key provided, skipping Twitch streaming")       
     
# Usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_input_port", type=int, required=False, default=3003, help="Port for receiving image as PNG ")
    parser.add_argument("--image_input_host", type=str, required=False, default="127.0.0.1", help="Host for receiving image as PNG")
    parser.add_argument("--audio_input_port", type=int, required=False, default=2001, help="Port for receiving audio as WAV ")
    parser.add_argument("--audio_input_host", type=str, required=False, default="127.0.0.1", help="Host for receiving audio as WAV")
    parser.add_argument("--twitchstreamkey", type=str, default="", required=False, help="Twitch stream key")
    parser.add_argument("--width", type=int, default=1920, help="Width of the output image")
    parser.add_argument("--height", type=int, default=1080, help="Height of the output image")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS of the output video")
    parser.add_argument("--samplerate", type=int, default=16000, help="Sample rate of the output audio")
    args = parser.parse_args()

    context = zmq.Context()
    image_socket = context.socket(zmq.SUB)
    print("connected to ZMQ Images in: %s:%d" % (args.image_input_host, args.image_input_port))
    image_socket.connect(f"tcp://{args.image_input_host}:{args.image_input_port}")
    image_socket.setsockopt_string(zmq.SUBSCRIBE, "")

    audio_socket = context.socket(zmq.SUB)
    print("connected to ZMQ Audio in: %s:%d" % (args.audio_input_host, args.audio_input_port))
    audio_socket.connect(f"tcp://{args.audio_input_host}:{args.audio_input_port}")
    audio_socket.setsockopt_string(zmq.SUBSCRIBE, "")

    exit_program = False
 
    stream_id = os.environ['TWITCH_STREAM_KEY']
    if stream_id == "":
        stream_id = args.twitchstreamkey
    else:
        args.twitchstreamkey = stream_id

    main()
