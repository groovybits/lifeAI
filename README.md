# Life AI - Bring your words to life using AI

## This uses SeamleessM4T a model that is multilingual for TTS STT

- https://github.com/facebookresearch/seamless_communication
- https://huggingface.co/docs/transformers/main/en/model_doc/seamless_m4t#seamlessm4t
- https://huggingface.co/facebook/hf-seamless-m4t-medium

## Installation

```
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

```
# Running TTS module with
# ZQM TCP 900 text in TO ZMQ TCP 1000 numpy audio samples out
./lifeAItts.py --input_port 900 --output_port 1000
#
# ZMQ listener client
python zmqTTSclient.py --input_port 1000
#
# ZMQ input test (TTS input to lifeAItts.py module
python zmqTTStest.py --target_port 900 --message "1:how are you today? I like to eat things that smell funny"
```
# 
# Chris Kennedy (C) GPL free as in free software
