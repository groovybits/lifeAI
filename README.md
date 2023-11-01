# Life AI - Bring your words to life using AI

Mudular python processes using zeromq to communicate between them. This allows mutiple chat models together and mixing of them into a mixer and out to twitch or where ever with ffmpeg/rtmp or anything ffmpeg can do. the nice part is using ffmpeg and packing audio/video into rtmp directly without OBS, and avoid all the overhead of need to decode it locally for broadcasting/streaming 😉.

Can build out endless prompt injection sources, news/twitch/voice-whisper listener/commandline/javascript web interface (that could have the video stream back and shared like youtube).

That’s the goal, you’ll see I am listing the parts as I build them, sort of have the core with llm/tts/stableDiffusion done + image subtitle burn in and prompt groomer for image gen, and generic for music usage (adding music tomorrow). twitch should be easy, I am parting out the parts of the consciousChat <https://github.com/groovybits/consciousChat> that seems more of a poc and experiment, nice, but this will remove the overhead and monolith design. It started to become too much to deal with putting it all into one app and threading everything. now each of these modules/programs are easy to understand for anyone and bypass python threading limitaitons.

## This uses the following models from huggingface

- <https://huggingface.co/facebook/mms-tts-eng> Facebook mms-tts-eng a model that is multilingual for TTS
- <https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF> A 7B parameter GPT-like model fine-tuned on a mix of publicly available, synthetic datasets.
- <https://huggingface.co/runwayml/stable-diffusion-v1-5> Stable Diffusion 1.5
- <https://huggingface.co/facebook/musicgen-small> Facebook MusicGen Music generation model
- <https://github.com/MycroftAI/mimic3> Mimic3 Text to Speech (optionally in place of Facebook mms-tts-eng).
- <https://github.com/ggerganov/llama.cpp/tree/master/examples/server> llama.cpp (install and run server for API access locally)

## ZMQ modules from Input to Output

- [Cmdline Input](zmqTextClient.py)        Send text into lifeAI for simulation seeding.
- [News Input](lifeAInewsCast.py)          Send news feeds into lifeAI for simulation seeding.
- [Twitch Chat Input](lifeAItwitchChat.py) Twitch Chat sent to lifeAI for responses.
- [Javscript Input](TODO.md)               TODO: Future plan to plug into react frontend video viewer.
- [Video Input](TODO.md)                   TODO: Read a video stream and alter it via AI through the system.
- [X Input](TODO.md)                       TODO: Any input via easy connection in a generic way.

- [LLM Broker llama.cpp-Python](lifeAIllm.py)       Llama2 llama.cpp Python SDK
- [LLM Broker llama.cpp-API](lifeAIllmAPI.py)       Llama2 llama.cpp server local API service
- [Prompt Optimizer](lifeAIpromptOptimizer.py)      Optimize prompt or turn text into a prompt.

- [TTS Producer](lifeAItts.py)    Facebook MMS-TTS Text to Speech Conversion.
- [TTM Producer](lifeAIttm.py)    Facebook Music Generation.
- [TTI Producer](lifeAItti.py)    Stable Diffusion Text to Image Generation.

- [Frame Sync](lifeAIframesync.py)           TODO: sync frames, add frames, remove frames...

- [Subtitle Burner](lifeAIsubTitleBurnIn.py) Burn-In subtitles in Anime style white/black bold.

- [TTS Listener](zmqTTSlisten.py) Listen for TTS Audio WAV file output and Playback/Save.
- [TTM Listener](zmqTTMlisten.py) Listen for TTM Audio WAV file output and Playback/Save.
- [TTI Listener](zmqTTIlisten.py) Listen for TTI Image PIL file output and Playback/Save.

- [Audio Mixer](lifeAImixer.py)   TODO: Mix Music and Speaking audio together
- [Muxer](lifeAImux.py)           TODO: Mux Images, Audio, Text and sync output time of each together.
- [ITV](lifeAIimtpy)              TODO: Image to Video, turn images into video sequences matching audio speaking duration.

- [Twitch Stream Output](lifeAITwitchStream.py) Twitch RTMP directly stream and avoid desktop capture.

## Installation

```text
# Create a virtual environment (type `deactivate` to exit it)
cd lifeAIpython
python3 -m venv lifeAI
source lifeAI/bin/activate

# Upgrade pip in venv
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Python Twitch Stream
git clone https://github.com/317070/python-twitch-stream.git
cd python-twitch-stream
pip install .

# Install Mimic3 TTS in place of FB TTS-MMS (requires Python 3.11 from Brew on MacOS X)
# Use lifeAIttsMimic3.py instead of lifeAItts.py
git clone https://github.com/MycroftAI/mimic3.git
cd mimic3/
PYTHON=python3.11 make install
source .venv/bin/activate
mimic3-server # (API Server)
curl -X POST --data 'Hello world.' --output - localhost:59125/api/tts > out.wav

# Get and install llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake .
make
sudo make install
cd examples/server/
# Launch an LLM instance per client, the server doesn't scale well beyond 1 client in llama.cpp currently :(
server -m /Volumes/BrahmaSSD/LLM/models/GGUF/zephyr-7b-alpha.Q2_K.gguf -t 60 -c 0 --mlock -ngl 60 --port 8081
server -m /Volumes/BrahmaSSD/LLM/models/GGUF/zephyr-7b-alpha.Q2_K.gguf -t 60 -c 0 --mlock -ngl 60 --port 8082
server -m /Volumes/BrahmaSSD/LLM/models/GGUF/zephyr-7b-alpha.Q2_K.gguf -t 60 -c 0 --mlock -ngl 60
```

## Running lifeAI

# Program Manager with [config.json](config.json) - [startLifeAI Program Manager](startLifeAI.py)

```text
# Setup for you already!
./startLifeAI.py

# Command line interface
> {list, status, start <program>, stop <program>, restart <program>, exit}

./startLifeAI.py --dry-run

[DRY RUN] Would start program: lifeAItwitchChat
Running in dry run mode. No actual processes will be started or stopped.
Enter command: [DRY RUN] Would start program: lifeAInewsCast
[DRY RUN] Would start program: lifeAIllmAPI
[DRY RUN] Would start program: lifeAItti
[DRY RUN] Would start program: lifeAIttsMimic3
[DRY RUN] Would start program: lifeAIttm
[DRY RUN] Would start program: lifeAIpromptOptimizeMusic
[DRY RUN] Would start program: lifeAIpromptOptimizeImages
[DRY RUN] Would start program: lifeAIsubtitleBurn
[DRY RUN] Would start program: zmqTTIlisten
[DRY RUN] Would start program: zmqTTSlisten
[DRY RUN] Would start program: zmqTTMlisten


```

# Manually running on the command line in screen/tmux session or multi-term

```text
# Run llama.cpp server for localhost API server and llama.cpp LLM handling
server -m /Volumes/BrahmaSSD/LLM/models/GGUF/zephyr-7b-beta.Q8_0.gguf -t 60 -c 0 --mlock

# Test llama.cpp API
curl --request POST --url http://127.0.0.1:8080/completion  \
             --header "Content-Type: application/json" \
             --data '{"prompt": "Building a website can be done in 10 simple steps:","n_predict": 128}'

# ZMQ input Client to send messages through the pipeline for testing
./zmqTextClient.py --message "An apple on a laptop." --segment_number 1 --username "User"

# ZMQ Twitch input Client
./lifeAItwitchClient.py

# ZMQ News feed Client Mediastack (Coming soon)

# ZMQ Whisper speech to text Client (Coming soon)

# LLM (requires llama.cpp server running)
## ./lifeAIllm.py (python client, buggy)
./lifeAIllmAPI.py

# TTS Speech audio
## ./lifeAItts.py # Easier to use, doesn't voice words properly
./lifeAIttsMimic3.py # pretty good local model

# TTI Images for video stream frames
./lifeAItti.py

# Prompt Optimizer for image and other media generation
./lifeAIpromptOptimization.py

# Subtitle Burn In for image subtitles hardsubs
./lifeAIsubTitleBurnIn.py

# Music generation
./lifeAIpromptOptimize.py --input_port 2000 --output_port 4001 --qprompt MusicDescription --aprompt MusicPrompt --topic 'music generation'
./lifeAIttm.py
./zmqTTMlisten.py --save_file

# Muxer and Frame Sync frontend (TODO)
./lifeAIframeSync.py (times all the streams for RTMP audio and video sync with everything timed together)

# ZMQ listener clients for listening, probing and viewing ascii image output
## Stored in audio/ and images/ as wav and png files with burn-in with filename
## metadata inclusion and episode_id, index, prompt string
./zmqTTSlisten.py
./zmqTTMlisten.py
./zmqTTIlisten.py

# Twitch RTMP direct stream without desktop OBS/capture overhead
./lifeAItwitchServe.py

# YouTube direct stream (TODO)

##
```

## Logging

```text
ls -altr logs/
```

## Restarting each individual unit via cmdline interface or kill

You can restart, stop and update the separate parts then restart them without much interuption.

## Twitch Chat Integration + Twitch Streaming to RTMP directly!

- Twitch Client can create personalities to use via !name <name> <personality> in chat.
- Fully functional interation with twitch users.
- Twitch streaming directly to RTMP (TODO: soon)

## News feed input via MediaStack

- Need to get a free or a low cost 25 a month subscription to MediaStack news feed service.
- Pulling news articles in batches of 100 and sends them one by one to the LLM for news reporter services.
- Fully configureable separate personality for news broadcaster feed.

## Mimic3 text to speech has many voices with different values to use

## Stable Diffusion Text to Image generation

- Automatically downloads the right models via transformers library

## Text to Music

- Facebook MusicGen auto-downloads all it needs with models

## Llama llama-cpp-python used yet the llama.cpp API server is preferred

- Need to run a separate server process with a different port and model for each LLM in use. It doesn't handle multiple clients.

## Ask if you want to help improve it, there are plenty of things todo...

## Chris Kennedy (C) 2023 GPL free as in free software, use at your own risk. Do not believe anything the LLM generates without your own validation. We are not responsible for how you use this software.
