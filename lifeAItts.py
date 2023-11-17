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
from openai import OpenAI

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
    #text = re.sub(r'[^a-zA-Z0-9\s.?,!\n:\'\"\-\t]', '', text)

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
        response = openai_client.audio.speech.create(
            model='tts-1',
            voice= voice or 'nova',
            input=text,
            speed=length_scale or '1.0',
            response_format='aac'
        )

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
        
        return audiobuf.getvalue()

def main():
    last_gender = args.gender
    last_voice_model = args.voice
    male_voice_index = 0
    female_voice_index = 0
    speaker_map = {}
    last_speaker = None
    voice_service = args.service
    service_switch = False
    while True:
        header_message = receiver.recv_json()
        segment_number = header_message["segment_number"]
        text = header_message["text"]
        episode = header_message["episode"]

        # voice, gender
        male_voices = []
        female_voices = []

        tts_api = args.service
        service_switch = False

        ## set the defaults
        voice_speed = "1.0"
        voice_model = None
        if tts_api == "mimic3":
            voice_speed = "1.5"
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
            male_voices = ['alloy', 'echo', 'fabel', 'oynx']
            female_voices = ['nova', 'shimmer']
            speaker_map['gabriella'] = {'voice': 'nova', 'gender': 'female'}
            default_voice = 'nova'
            if voice_service != "openai":
                voice_service = "openai"
                voice_model = default_voice
                service_switch = True
                last_voice_model = default_voice
        elif tts_api == "mimic3":
            male_voices = [
                'en_US/hifi-tts_low#6097',
                'en_US/hifi-tts_low#9017',
                'en_US/vctk_low#p259',
                'en_US/vctk_low#p247',
                'en_US/vctk_low#p263',
                'en_US/vctk_low#p274',
                'en_US/vctk_low#p286',
                'en_US/vctk_low#p270',
                'en_US/vctk_low#p281',
                'en_US/vctk_low#p271',
                'en_US/vctk_low#p273',
                'en_US/vctk_low#p284',
                'en_US/vctk_low#p287',
                'en_US/vctk_low#p360',
                'en_US/vctk_low#p274',
                'en_US/vctk_low#p376',
                'en_US/vctk_low#p304',
                'en_US/vctk_low#p347',
                'en_US/vctk_low#p311',
                'en_US/vctk_low#p334',
                'en_US/vctk_low#p316',
                'en_US/vctk_low#p363',
                'en_US/vctk_low#p275',
                'en_US/vctk_low#p258',
                'en_US/vctk_low#p232',
                'en_US/vctk_low#p292',
                'en_US/vctk_low#p272',
                'en_US/vctk_low#p278',
                'en_US/vctk_low#p298',
                'en_US/vctk_low#p279',
                'en_US/vctk_low#p285',
                'en_US/vctk_low#p326', # super deep voice
                'en_US/vctk_low#p254',
                'en_US/vctk_low#p252',
                'en_US/vctk_low#p345',
                'en_US/vctk_low#p243',
                'en_US/vctk_low#p227',
                'en_US/vctk_low#p225',
                'en_US/vctk_low#p251',
                'en_US/vctk_low#p246',
                'en_US/vctk_low#p226',
                'en_US/vctk_low#p260',
                'en_US/vctk_low#p245',
                'en_US/vctk_low#p241',
                'en_US/vctk_low#p237',
                'en_US/vctk_low#p256',
                'en_US/vctk_low#p302',
                'en_US/vctk_low#p264',
                'en_US/vctk_low#p225',
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
                'en_US/hifi-tts_low#92',
                'en_US/vctk_low#s5',
                'en_US/vctk_low#p264',
                'en_US/vctk_low#p239',
                'en_US/vctk_low#p236',
                'en_US/vctk_low#p250',
                'en_US/vctk_low#p261',
                'en_US/vctk_low#p283',
                'en_US/vctk_low#p276',
                'en_US/vctk_low#p277',
                'en_US/vctk_low#p231',
                'en_US/vctk_low#p238',
                'en_US/vctk_low#p257',
                'en_US/vctk_low#p329',
                'en_US/vctk_low#p261',
                'en_US/vctk_low#p310',
                'en_US/vctk_low#p340',
                'en_US/vctk_low#p330',
                'en_US/vctk_low#p308',
                'en_US/vctk_low#p314',
                'en_US/vctk_low#p317',
                'en_US/vctk_low#p339',
                'en_US/vctk_low#p294',
                'en_US/vctk_low#p305',
                'en_US/vctk_low#p266',
                'en_US/vctk_low#p318',
                'en_US/vctk_low#p323',
                'en_US/vctk_low#p351',
                'en_US/vctk_low#p333',
                'en_US/vctk_low#p313',
                'en_US/vctk_low#p244',
                'en_US/vctk_low#p307',
                'en_US/vctk_low#p336',
                'en_US/vctk_low#p312',
                'en_US/vctk_low#p267',
                'en_US/vctk_low#p297',
                'en_US/vctk_low#p295',
                'en_US/vctk_low#p288',
                'en_US/vctk_low#p301',
                'en_US/vctk_low#p280',
                'en_US/vctk_low#p241',
                'en_US/vctk_low#p268',
                'en_US/vctk_low#p299',
                'en_US/vctk_low#p300',
                'en_US/vctk_low#p230',
                'en_US/vctk_low#p269',
                'en_US/vctk_low#p293',
                'en_US/vctk_low#p262',
                'en_US/vctk_low#p343',
                'en_US/vctk_low#p229',
                'en_US/vctk_low#p240',
                'en_US/vctk_low#p248',
                'en_US/vctk_low#p253',
                'en_US/vctk_low#p233',
                'en_US/vctk_low#p228',
                'en_US/vctk_low#p282',
                'en_US/vctk_low#p234',
                'en_US/vctk_low#p303', # nice crackly voice
                'en_US/vctk_low#p265',
                'en_US/vctk_low#p306',
                'en_US/vctk_low#p249',
                'en_US/vctk_low#p362',
                'en_US/ljspeech_low',
                'en_US/cmu-arctic_low#ljm',
                'en_US/cmu-arctic_low#slp',
                'en_US/cmu-arctic_low#axp',
                'en_US/cmu-arctic_low#eey',
                'en_US/cmu-arctic_low#lnh',
                'en_US/cmu-arctic_low#elb',
                'en_US/cmu-arctic_low#slt'
            ]
            default_voice = 'en_US/hifi-tts_low#92',
            if voice_service != "mimic3":
                voice_service = "mimic3"
                if voice_model == None:
                    voice_model = default_voice
                service_switch = True
                last_voice_model = voice_model

        # Guess gender
        gender = last_gender

        # Find and assign voices to speakers
        new_voice_model = None

        if voice_model == None:
            if service_switch:
                voice_model = default_voice
            else:
                voice_model = last_voice_model

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

                    # Identify gender from text if not determined by name
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
                        if male_voice_index > len(male_voices):
                            male_voice_index = 0
                        voice_choice = male_voices[male_voice_index % len(male_voices)]
                        male_voice_index += 1
                    else:  # Female and nonbinary use female voices
                        if female_voice_index > len(female_voices):
                            female_voice_index = 0
                        voice_choice = female_voices[female_voice_index % len(female_voices)]
                        female_voice_index += 1

                    speaker_map[speaker] = {'voice': voice_choice, 'gender': gender}
                    new_voice_model = voice_choice
                    last_speaker = speaker  # Update the last speaker
                else:
                    new_voice_model = speaker_map[speaker]['voice']
                    gender = speaker_map[speaker]['gender']
                    if new_voice_model not in female_voices and new_voice_model not in male_voices:
                        if gender == "male":
                            if male_voice_index > len(male_voices):
                                male_voice_index = 0
                            new_voice_model = male_voices[male_voice_index]
                        else:
                            if female_voice_index > len(female_voices):
                                female_voice_index = 0
                            new_voice_model = female_voices[female_voice_index]
                    last_gender = gender
                    last_speaker = speaker  # Update the last speaker

                logger.info(f"Text to Speech: Speaker found: {speaker} with voice {speaker_map[speaker]['voice']}.")
                break  # If you want to process one speaker at a time, otherwise remove this line
            else:
                # No new speaker found, use last speaker and gender if available
                if last_speaker:
                    speaker = last_speaker
                    gender = last_gender
                    new_voice_model = speaker_map[speaker]['voice'] if speaker in speaker_map else last_voice_model
                    logger.info(f"Text to Speech: Continuing with last speaker: {speaker} and voice {new_voice_model}.")
                else:
                    logger.debug("Text to Speech: No speaker found, and no last speaker to default to.")

        # Outside of the for loop
        if new_voice_model:
            logger.info(f"Text to Speech: Speaker found, switching from {last_voice_model} to voice {new_voice_model}.")
            last_voice_model = new_voice_model
            voice_model = new_voice_model
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
    parser.add_argument("--voice", type=str, default='en_US/ljspeech_low', help="Voice parameter for TTS API")
    parser.add_argument("--noise_scale", type=str, default='0.333', help="Noise scale parameter for TTS API")
    parser.add_argument("--noise_w", type=str, default='0.333', help="Noise weight parameter for TTS API")
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

    openai_client = OpenAI()

    d = gender.Detector(case_sensitive=False)

    main()
