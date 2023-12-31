# Life AI - Bring your words to life using AI

Try it <https://twitch.tv/ai_buddha> allows full interactive usage of Groovy Life AI.

Presentation from Nov 2023 at the [Multimodal Weekly](https://twelvelabs-20253029.hs-sites.com/multimodal-weekly-1117): <https://groovylife.ai/mw1123presentation/>

The Groovy Life AI community <https://groovylife.ai> for support and find others to discuss AI with.

Modular python processes using zeromq to communicate between them. This allows mutiple chat models together and mixing of them into a mixer and out to twitch or where ever with ffmpeg/rtmp or anything ffmpeg can do. the nice part is using ffmpeg and packing audio/video into rtmp directly without OBS, and avoid all the overhead of need to decode it locally for broadcasting/streaming 😉.

Can build out endless prompt injection sources, news/twitch/voice-whisper listener/commandline/javascript web interface (that could have the video stream back and shared like youtube).

That’s the goal, you’ll see I am listing the parts as I build them, sort of have the core with llm/tts/stableDiffusion done + image subtitle burn in and prompt groomer for image gen, and generic for music usage (adding music tomorrow). twitch should be easy, I am parting out the parts of the consciousChat <https://github.com/groovybits/consciousChat> that seems more of a poc and experiment, nice, but this will remove the overhead and monolith design. It started to become too much to deal with putting it all into one app and threading everything. now each of these modules/programs are easy to understand for anyone and bypass python threading limitaitons.

Diagram:

![Groovy Life AI](https://storage.googleapis.com/gaib/2/documentation/GroovyLifeAI.png)

## This uses the following dependencies

- <https://github.com/ggerganov/llama.cpp/tree/master/examples/server> LLM  llama.cpp (install and run server for API access locally)
- <https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF> Model 7B parameter GPT-like model fine-tuned on a mix of publicly available, synthetic datasets.
- <https://github.com/MycroftAI/mimic3> Mimic3 Text to Speech (optionally in place of Facebook mms-tts-eng).
- <https://huggingface.co/runwayml/stable-diffusion-v1-5> Text to Image Stable Diffusion 1.5
- <https://huggingface.co/facebook/musicgen-small> Text to Music Facebook MusicGen Music generation model
- <https://github.com/mix1009/sdwebuiapi> Stable Diffusion WebUI using a local API in place of huggingface (faster)
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui> Stable Diffusion WebUI server, use [bin/start_sdwebui.sh](bin/start_sdwebui.sh) to start
- <https://github.com/ggerganov/llama.cpp> Llama2 local LLM API server
- <https://civitai.com/models/3627/protogen-v22-anime-official-release> Default image generation model Protogen

## OpenAI GPT-* API Support!!!

- Coming soon, very soon!

## OpenAI Dalle-3 API Support!!!

- If you have an OpenAI API key you can set it in .env and run the TTI module as --service "openai".

## OpenAI TTS API Support!!!

- Works! You can choose voices for each input client to use or have them choose one dynamically. This will allow Twitch chat to use different voices per personality!

## ZMQ modules from Input to Output

- [Program Manager](lifeAIstart.py)        Startup and control all the modules, uses [config.json](config.json)
- [Cmdline Input](zmqTextClient.py)        Send text into lifeAI for simulation seeding.
- [News Input](lifeAInewsCast.py)          Send news feeds into lifeAI for simulation seeding.
- [Twitch Chat Input](lifeAItwitchChat.py) Twitch Chat sent to lifeAI for responses.
- [Mailing List Input](lifeAIlists.py)     Mailing lists via imap and separate labels for each list box in lists+list-name@ format.
- [Javscript Web Frontend](TODO.md)        TODO: Future plan to plug into react frontend video viewer.
- [Video Stream Input](TODO.md)            TODO: Read a video stream and alter it via AI through the system.
- [X Input](TODO.md)                       TODO: Any input via easy connection in a generic way. (Currently turning it into an email list works well)

- [LLM Broker llama.cpp-API](lifeAIllmAPI.py)       Llama2 llama.cpp server local API service
- [Prompt Optimizer](lifeAIpromptOptimizerAPI.py)      Optimize prompt or turn text into a prompt.

- [TTS Producer](lifeAItts)             Mimic3, MMS-TTS or OpenAI TTS Text to Speech Conversion.
- [TTM Producer](lifeAIttm.py)          Facebook Music Generation.
- [TTI Producer](lifeAItti.py)          Stable Diffusion Text to Image Generation. (extended prompt + NSFW off option)

- [Document Injector](lifeAIdoc.py)     VectorDB Document Retrieval using ChromaDB and Gpt4All embeddings. (requires the privateGPT git to run within as the env, WIP)

- [Frame Sync](lifeAIframesync.py)      sync frames, add frames, remove frames...

- [Subtitle Burner](lifeAIsubTitleBurnIn.py) Burn-In subtitles in Anime style white/black bold.

- [Life AI Player](lifeAIplayer.py)          Life AI Main player for A/V (Music still requires TTM Listener for now)

- [TTS Listener](zmqTTSlisten.py) Listen for TTS Audio WAV file output and Playback/Save.
- [TTM Listener](zmqTTMlisten.py) Listen for TTM Audio WAV file output and Playback/Save.
- [TTI Listener](zmqTTIlisten.py) Listen for TTI Image PIL file output and Playback/Save.

- [Audio Mixer](lifeAImixer.py)   TODO: Mix Music and Speaking audio together
- [Muxer](lifeAImux.py)           TODO: Mux Images, Audio, Text and sync output time of each together.
- [ITV](lifeAIimtpy)              TODO: Image to Video, turn images into video sequences matching audio speaking duration.

- [Twitch Stream Output](lifeAITwitchStream.py)         Twitch RTMP directly stream and avoid desktop capture.
- [Twitch Stream FFmpeg Script](bin/twitch_stream.sh)   Twitch streaming script for avoidance of OBS.

- [Clean Up Assets](bin/cleanup.sh)                     Deletes assets older than "seconds" old, give value on cmdline `cleanup.sh 60` for only keeping the last hours of audio/ music/ images/ logs/ directory files/folders

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
# Use lifeAItts.py
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
# server is installed in /usr/local/bin/server and ran via the config.json

## MacOS deps (or any system)
brew install libmagic
brew install zeromq
brew install ffmpeg

## Spacy package requirement
python -m spacy download en_core_web_sm

```

## Running lifeAI

- Program Manager with [config.json](config.json) - [startLifeAI Program Manager](startLifeAI.py)

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
[DRY RUN] Would start program: lifeAItts
[DRY RUN] Would start program: lifeAIttm
[DRY RUN] Would start program: lifeAIpromptOptimizeMusic
[DRY RUN] Would start program: lifeAIpromptOptimizeImages
[DRY RUN] Would start program: lifeAIsubtitleBurn
[DRY RUN] Would start program: zmqTTIlisten
[DRY RUN] Would start program: zmqTTSlisten
[DRY RUN] Would start program: zmqTTMlisten
```

## Manually running on the command line in screen/tmux session or multi-term

```text
# Run llama.cpp server for localhost API server and llama.cpp LLM handling
server -m /Volumes/BrahmaSSD/LLM/models/GGUF/zephyr-7b-beta.Q8_0.gguf -t 60 -c 0 --mlock -ngl -1

# Prompt Optimization for Music and Images (must run one server per query agent, it is not multi-threaded)
server -m /Volumes/BrahmaSSD/LLM/models/GGUF/zephyr-7b-beta.Q8_0.gguf -t 60 -c 0 --mlock --port 8081 -ngl -1
server -m /Volumes/BrahmaSSD/LLM/models/GGUF/zephyr-7b-beta.Q8_0.gguf -t 60 -c 0 --mlock --port 8082 -ngl -1

# Run Stable Diffusion WebUI (need to get it, see link above)
cd stable-diffusion-webui/ && bin/start_sdwebui.sh

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
./lifeAIllmAPI.py

# TTS Speech audio
## ./lifeAItts.py # Easier to use, doesn't voice words properly
./lifeAItts.py # can use Mimic3, mms-tts or OpenAI tts

# TTI Images for video stream frames
./lifeAItti.py

# Prompt Optimizer for image and other media generation
./lifeAIpromptOptimization.py

# Subtitle Burn In for image subtitles hardsubs
./lifeAIsubTitleBurnIn.py

# Music generation
./lifeAIpromptOptimizeAPI.py --input_port 2000 --output_port 4001 --qprompt MusicDescription --aprompt MusicPrompt --topic 'music generation'
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

# Twitch RTMP direct stream without desktop OBS/capture overhead (doesn't work yet)
./lifeAItwitchServe.py

# YouTube direct stream (use <https://restream.io> for now)

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
- Twitch streaming directly to RTMP (TODO)

## News feed input via MediaStack

- Need to get a free or a low cost 25 a month subscription to MediaStack news feed service.
- Pulling news articles in batches of 100 and sends them one by one to the LLM for news reporter services.
- Fully configureable separate personality for news broadcaster feed.

## Mimic3 text to speech has many voices with different values to use

## Stable Diffusion Text to Image generation

- Automatically downloads the right models via transformers library
- <https://github.com/huggingface/diffusers/issues/2136> allows greater than 77 token prompting! Metal GPU optimized model

## Text to Music

- Facebook MusicGen auto-downloads all it needs with models

## Llama llama-cpp-python used yet the llama.cpp API server is preferred

- Need to run a separate server process with a different port and model for each LLM in use. It doesn't handle multiple clients.

## Ask if you want to help improve it, there are plenty of things todo...

## Chris Kennedy (C) 2023 GPL free as in free software, use at your own risk. Do not believe anything the LLM generates without your own validation. We are not responsible for how you use this software.
