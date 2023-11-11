#!/usr/bin/env python

## Life AI Text to Speech module
#
# Chris Kennedy 2023 (C) GPL
#
# Free to use for any use as in truly free software
# as Richard Stallman intended it to be.
#

import zmq
import argparse
import requests
import io
import warnings
import re
import logging
import time
import os
from dotenv import load_dotenv
import inflect
import traceback
import soundfile as sf
import torch
from transformers import VitsModel, AutoTokenizer
from transformers import logging as trlogging
from pydub import AudioSegment
import gender_guesser.detector as gender

trlogging.set_verbosity_error()

load_dotenv()

# Suppress warnings
warnings.simplefilter(action='ignore', category=Warning)

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Remove image tags or Markdown image syntax
    text = re.sub(r'\!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'<img.*?>', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove any inline code blocks
    text = re.sub(r'`.*?`', '', text)
    
    # Remove any block code segments
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    
    # Remove special characters and digits (optional, be cautious)
    text = re.sub(r'[^a-zA-Z0-9\s.?,!\n:\'\"\-\t]', '', text)

    if args.service == "mms-tts":
        p = inflect.engine()

        def num_to_words(match):
            number = match.group()
            try:
                words = p.number_to_words(number)
            except inflect.NumOutOfRangeError:
                words = "[number too large]"
            return words

        text = re.sub(r'\b\d+(\.\d+)?\b', num_to_words, text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())

    return text

def get_aac_duration(aac_data):
    audio_segment = AudioSegment.from_file(io.BytesIO(aac_data), format='aac')
    return len(audio_segment) / 1000.0  # Convert from milliseconds to seconds

def get_tts_audio(service, text, voice=None, noise_scale=None, noise_w=None, length_scale=None, ssml=None, audio_target=None):
    
    if service == "mimic3":
        params = {
            'text': text,
            'voice': voice or 'en_US/cmu-arctic_low#slt',
            'noiseScale': noise_scale or '0.333',
            'noiseW': noise_w or '0.333',
            'lengthScale': length_scale or '1.0',
            'ssml': ssml or 'false',
            'audioTarget': audio_target or 'client'
        }

        response = requests.get('http://earth:59125/api/tts', params=params)
        response.raise_for_status()
        return response.content
    elif service == "openai":
        """
        curl https://api.openai.com/v1/audio/speech \
            -H "Authorization: Bearer $OPENAI_API_KEY" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"tts-1\",
                \"input\": \"AI is amazing and Anime is good. It is a miracle that GPT-4 is so good.\",
                \"voice\": \"$v\",
                \"response_format\": \"aac\",
                \"speed\": \"1.0\"
            }" \
                --output speech_$v.aac
        """
        params = {
            'model': 'tts-1',
            'input': text,
            'voice': voice or 'nova',
            'speed': length_scale or '1.0',
            'response_format':'aac'
        }

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
        }

        response = requests.post('https://api.openai.com/v1/audio/speech', headers=headers, json=params)
        response.raise_for_status()
        return response.content
    elif service == "mms-tts":
        inputs = tokenizer(text, return_tensors="pt")
        inputs['input_ids'] = inputs['input_ids'].long()

        output = None
        try:
            with torch.no_grad():
                output = model(**inputs).waveform
            waveform_np = output.squeeze().numpy().T
        except Exception as e:
            logger.error(f"{traceback.print_exc()}")
            logger.error(f"Exception: ERROR STT error with output.squeeze().numpy().T on audio: {text}")
            return None
        
        audiobuf = io.BytesIO()
        sf.write(audiobuf, waveform_np, model.config.sampling_rate, format='WAV')
        audiobuf.seek(0)

        #duration = len(waveform_np) / model.config.sampling_rate
        
        return audiobuf.getvalue()

def main():
    last_gender = args.gender
    last_voice_model = args.voice
    male_voice_index = 0
    female_voice_index = 0
    while True:
        header_message = receiver.recv_json()
        segment_number = header_message["segment_number"]
        text = header_message["text"]
        episode = header_message["episode"]

        # voice, gender
        speaker_map = {}
        male_voices = []
        female_voices = []

        tts_api = args.service

        ## set the defaults
        voice_model = last_voice_model
        voice_speed = "1.0"
        if tts_api == "mimic3":
            voice_speed = "1.2"
        else:
            voice_speed = args.length_scale

        if 'voice_model' in header_message:
            voice_data = header_message["voice_model"]
            # "voice_model": "mimic3:en_US/cmu-arctic_low#eey:1.2",
            # TTS API, Voice Model to use, Voice Model Speed to use
            tts_api = voice_data.split(":")[0]
            voice_model = voice_data.split(":")[1]
            voice_speed = voice_data.split(":")[2]
            logger.info(f"Text to Speech: Voice Model selected: {voice_model} at speed {voice_speed} using API {tts_api}.")
        else:
            logger.info(f"Text to Speech: Voice Model default, no 'voice_model' in request: {voice_model} at speed {voice_speed} using API {tts_api}.")

        if tts_api == "openai":
            if args.length_scale == "1.2":
                args.length_scale = "0.8"
            male_voices = ['alloy', 'echo', 'fabel', 'oynx']
            female_voices = ['nova', 'shimmer']
            speaker_map['gabriella'] = {'voice': 'nova', 'gender': 'female'}
        elif tts_api == "mimic3":
            male_voices = [
                'en_US/cmu-arctic_low#rms',
                'en_US/cmu-arctic_low#ksp',
                'en_US/cmu-arctic_low#aew',
                'en_US/cmu-arctic_low#bdl',
                'en_US/cmu-arctic_low#jmk',
                'en_US/cmu-arctic_low#fem',
                'en_US/cmu-arctic_low#ahw',
                'en_US/cmu-arctic_low#aup',
                'en_US/cmu-arctic_low#gke'
            ]
            female_voices = [
                'en_US/cmu-arctic_low#ljm',
                'en_US/cmu-arctic_low#slp',
                'en_US/cmu-arctic_low#axp',
                'en_US/cmu-arctic_low#eey',
                'en_US/cmu-arctic_low#lnh',
                'en_US/cmu-arctic_low#elb',
                'en_US/cmu-arctic_low#slt'
            ]
            speaker_map['Gaibriella'] = {'voice': args.voice, 'gender': 'female'}
            speaker_map['Giabriella'] = {'voice': args.voice, 'gender': 'female'}
            speaker_map['Narrator'] = {'voice': args.voice, 'gender': 'female'}

        # Guess gender
        gender = last_gender

        # Find and assign voices to speakers
        new_voice_model = None

        # Regex pattern to find speaker names with different markers
        speaker_pattern = r'^(?:\[/INST\])?<<([A-Za-z]+)>>|^(?:\[\w+\])?([A-Za-z]+):'

        for line in text.split('\n'):
            speaker_match = re.search(speaker_pattern, line)
            if speaker_match:
                # Extracting speaker name from either of the capturing groups
                speaker = speaker_match.group(1) or speaker_match.group(2)
                speaker = speaker.strip()

                if speaker not in speaker_map:
                    guessed_gender = d.get_gender(speaker.split()[0])  # assuming the first word is the name
                    
                    if guessed_gender in ['male', 'mostly_male']:
                        gender = "male"
                    elif guessed_gender in ['female', 'mostly_female']:
                        gender = "female"
                    else:
                        gender = "nonbinary"

                        # Identify gender from text
                        if re.search(r'\[m\]', text):
                            gender = "male"
                        elif re.search(r'\[f\]', text):
                            gender = "female"
                        elif re.search(r'\[n\]', text):
                            gender = "nonbinary"
                        else:
                            gender = last_gender

                    last_gender = gender

                    if gender == "male":
                        voice_choice = male_voices[male_voice_index % len(male_voices)]
                        male_voice_index += 1
                    else:  # Female and nonbinary use female voices
                        voice_choice = female_voices[female_voice_index % len(female_voices)]
                        female_voice_index += 1

                    speaker_map[speaker] = {'voice': voice_choice, 'gender': gender}
                    new_voice_model = voice_choice
                else:
                    new_voice_model = speaker_map[speaker]['voice']
                    gender = speaker_map[speaker]['gender']
                    last_gender = gender

                logger.info(f"Text to Speech: Speaker found: {speaker} with voice {speaker_map[speaker]['voice']}.")
                break  # If you want to process one speaker at a time, otherwise remove this line
        else:
            logger.debug(f"Text to Speech: Speaker not found in text {text}.")

        if new_voice_model:
            logger.info(f"Text to Speech: Speaker found, switching from {last_voice_model} to voice {new_voice_model}.")
            last_voice_model = new_voice_model
        else:
            logger.info(f"Text to Speech: Speaker not found, using default voice {voice_model}.")
            last_voice_model = voice_model
        
        # clean text of end of line spaces after punctuation
        text = clean_text(text)
        text = re.sub(r'([.,!?;:])\s+', r'\1', text)

        logger.debug("Text to Speech received request:\n%s" % header_message)
        logger.info(f"Text to Speech received request #{segment_number}:\n{text}")

        # add ssml tags
        if args.ssml == 'true' and tts_api == "mimic3":
            text = f"<speak><prosody pitch=\"{args.pitch}\" range=\"{args.range}\" rate=\"{args.rate}\">" + text + f"</prosody></speak>"
            logger.info(f"Text to Speech: SSML enabled, using pitch={args.pitch}, range={args.range}, rate={args.rate}.")
            logger.debug(f"Text to Speech: SSML text:\n{text}")

        duration = 0
        try:
            audio_blob = get_tts_audio(
                tts_api,
                text,
                voice=last_voice_model,
                noise_scale=args.noise_scale,
                noise_w=args.noise_w,
                length_scale=voice_speed,
                ssml=args.ssml,
                audio_target=args.audio_target
            )
            if tts_api == "mimic3" or tts_api == "mms-tts":
                duration = len(audio_blob) / (22050 * 2)  # Assuming 22.5kHz 16-bit audio for duration calculation
            elif tts_api == "openai":
                duration = get_aac_duration(audio_blob)
        except Exception as e:
            logger.error(f"Exception: ERROR TTS error with API request for text: {text}")
            logger.error(e)
            continue

        if duration == 0:
            logger.error(f"Exception: ERROR TTS {tts_api} {voice_model} x{voice_speed} returned 0 duration audio blobt: {text}")
            continue

        audiobuf = io.BytesIO(audio_blob)
        audiobuf.seek(0)

        # Fill in the header
        header_message["duration"] = duration
        header_message["stream"] = "speek"

        # Send the header and the audio
        sender.send_json(header_message, zmq.SNDMORE)
        sender.send(audiobuf.getvalue())

        logger.debug(f"Text to Speech: sent audio #{segment_number}\n{header_message}")
        logger.info(f"Text to Speech: sent audio #{segment_number} of {duration} duration.\n{text}")

        header_message = None
        text = ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, default=2000, required=False, help="Port for receiving text input")
    parser.add_argument("--output_port", type=int, default=6002, required=False, help="Port for sending audio output")
    parser.add_argument("--input_host", type=str, default="127.0.0.1", required=False, help="Host for receiving text input")
    parser.add_argument("--output_host", type=str, default="127.0.0.1", required=False, help="Host for sending audio output")
    parser.add_argument("--voice", type=str, default='en_US/cmu-arctic_low#eey', help="Voice parameter for TTS API")
    parser.add_argument("--noise_scale", type=str, default='0.6', help="Noise scale parameter for TTS API")
    parser.add_argument("--noise_w", type=str, default='0.6', help="Noise weight parameter for TTS API")
    parser.add_argument("--length_scale", type=str, default='1.0', help="Length scale parameter for TTS API")
    parser.add_argument("--ssml", type=str, default='false', help="SSML parameter for TTS API")
    parser.add_argument("--audio_target", type=str, default='client', help="Audio target parameter for TTS API")
    parser.add_argument("-ll", "--loglevel", type=str, default="info", help="Logging level: debug, info...")
    parser.add_argument("--sub", action="store_true", default=False, help="Publish to a topic")
    parser.add_argument("--pub", action="store_true", default=False, help="Publish to a topic")
    parser.add_argument("--bind_output", action="store_true", default=False, help="Bind to a topic")
    parser.add_argument("--bind_input", action="store_true", default=False, help="Bind to a topic")
    parser.add_argument("--rate", type=str, default="default", help="Speech rate, slow, medium, fast")
    parser.add_argument("--range", type=str, default="high", help="Speech range, low, medium, high")
    parser.add_argument("--pitch", type=str, default="high", help="Speech pitch, low, medium, high")
    parser.add_argument("--delay", type=int, default=0, help="Delay in seconds after timestamp before sending audio")
    parser.add_argument("--service", type=str, default="mimic3", help="TTS service to use. mms-tts, mimic3, openai")
    parser.add_argument("--metal", action="store_true", default=False, help="offload to metal mps GPU")
    parser.add_argument("--cuda", action="store_true", default=False, help="offload to cuda GPU")
    parser.add_argument("--gender", type=str, default="female", help="Gender default for characters without [m], [f], or [n] markers")

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
    logging.basicConfig(filename=f"logs/ttsMimic3-{log_id}.log", level=LOGLEVEL)
    logger = logging.getLogger('tts')

    ch = logging.StreamHandler()
    ch.setLevel(LOGLEVEL)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    context = zmq.Context()
    # Set up the subscriber
    receiver = context.socket(zmq.SUB)
    print(f"Setup ZMQ in {args.input_host}:{args.input_port}")
    receiver.connect(f"tcp://{args.input_host}:{args.input_port}")
    receiver.setsockopt_string(zmq.SUBSCRIBE, "")

    # Set up the publisher
    sender = context.socket(zmq.PUSH)
    print(f"binded to ZMQ out {args.output_host}:{args.output_port}")
    sender.connect(f"tcp://{args.output_host}:{args.output_port}")

    model = None
    tokenizer = None
    if args.service == "mms-tts":
        model = VitsModel.from_pretrained("facebook/mms-tts-eng")
        tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

        if args.metal:
            model.to("mps")
        elif args.cuda:
            model.to("cuda")
        else:
            model.to("cpu")


    d = gender.Detector(case_sensitive=False)

    main()
