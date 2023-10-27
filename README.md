# Life AI - Bring your words to life using AI

## This uses the following models from huggingface

- <https://huggingface.co/facebook/mms-tts-eng> Facebook mms-tts-eng a model that is multilingual for TTS
- <https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF> A 7B parameter GPT-like model fine-tuned on a mix of publicly available, synthetic datasets.
- <https://huggingface.co/runwayml/stable-diffusion-v1-5> Stable Diffusion 1.5

## modules

- [ZMQ Text Client](zmqTextClient.py) Send text into lifeAI for simulation seeding.
- [ZMQ Twitch Client](zmqTwitchClient.py) Twitch chat sent to lifeAI for responses.
- [ZMQ TTS Listener](zmqTTSlisten.py) Listen for TTS Audio WAV file output.
- [ZMQ TTI Listener](zmqTTIlisten.py) Listen for TTI Image PIL file output.
- [Text to AI Speech](lifeAItts.py)   Facebook MMS-TTS Text to Speech Conversion.
- [Text to AI Image](lifeAItti.py)    Stable Diffusion Text to Image Generation.
- [Prompt Optimizer](lifeAIpromptOptimizer.py) Optimize prompt or turn text into a prompt.
- [Subtitle Burner](lifeAIsubTitleBurnIn.py) Burn-In subtitles in Anime style white/black bold.

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

```

## Running lifeAI

```text
# ZMQ input Client to send messages through the pipeline for testing
python zmqTextClient.py --message "An apple on a laptop." --segment_number 1 --username "User"

# ZMQ Twitch input Client (Coming soon)

# ZMQ News feed Client Mediastack (Coming soon)

# ZMQ Whisper speech to text Client (Coming soon)

# LLM Text track
./lifeAIllm.py

# TTS Speech audio
./lifeAItts.py

# TTI Images for video stream frames
./lifeAItti.py

# Prompt Optimizer for image and other media generation
./lifeAIpromptOptimization.py

# Subtitle Burn In for image subtitles hardsubs
./lifeAIsubTitleBurnIn.py

# ZMQ listener clients for listening, probing and viewing ascii image output
## Stored in audio/ and images/ as wav and png files with burn-in with filename
## metadata inclusion and episode_id, index, prompt string
python zmqTTSlisten.py
python zmqTTIlisten.py

##
```

## Chris Kennedy (C) GPL free as in free software
