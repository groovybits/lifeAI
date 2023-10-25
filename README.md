# Life AI - Bring your words to life using AI

## This uses Facebook mms-tts-eng a model that is multilingual for TTS

- <https://huggingface.co/facebook/mms-tts-eng>

## modules

- [ZMQ Text Client](zmqTextClient.py) Send text into ZMQ for modules.
- [ZMQ TTS Listener](zmqTTSlisten.py) Listen for TTS Audio WAV file output.
- [ZMQ TTI Listener](zmqTTIlisten.py) Listen for TTI Image PIL file output.
- [Text to AI Speech](lifeAItts.py)
- [Text to AI Image](lifeAItti.py)

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
./lifeAItts.py --input_port 900 --output_port 1000
./lifeAItti.py --input_port 901 --output_port 1001
#
# ZMQ listener client
python zmqTTSlisten.py --input_port 1000
python zmqTTIlisten.py --input_port 1001
#
# ZMQ input test (TTS input to lifeAItts.py module)
python zmqTextClient.py --target_port 900 --message "1:how are you today?"
python zmqTextClient.py --target_port 901 --message "1:how are you today?"
```

## Chris Kennedy (C) GPL free as in free software
