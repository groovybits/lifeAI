# Life AI - Bring your words to life using AI

## This uses Facebook mms-tts-eng a model that is multilingual for TTS

- <https://huggingface.co/facebook/mms-tts-eng>

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
#
# ZMQ listener client
python zmqTTSclient.py --input_port 1000
#
# ZMQ input test (TTS input to lifeAItts.py module)
python zmqTTStest.py --target_port 900 --message "1:how are you today?"
```

## Chris Kennedy (C) GPL free as in free software
