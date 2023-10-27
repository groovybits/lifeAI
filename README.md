# Life AI - Bring your words to life using AI

## This uses Facebook mms-tts-eng a model that is multilingual for TTS

- <https://huggingface.co/facebook/mms-tts-eng>

## modules

- [ZMQ Text Client](zmqTextClient.py) Send text into lifeAI TTS and TTI processing.
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
```

## Running lifeAI

```text
# Running TTS module with
# ZQM TCP 900 text in TO ZMQ TCP 1000 numpy audio samples out
./lifeAItts.py
./lifeAItti.py
./lifeAIpromptOptimization.py
./lifeAIsubTitleBurnIn.py --use_prompt
#
# ZMQ listener client
python zmqTTSlisten.py --output_file audio.wav
python zmqTTIlisten.py --output_file image.jpg
#
# ZMQ input test (TTS/TTI input to lifeAItts.py and lifeAItti modules ZMQ Ports)
python zmqTextClient.py --message "An apple on a laptop." --segment_number 1
##
```

## Chris Kennedy (C) GPL free as in free software
