#!/bin/bash
#
# https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
#
cd stable-diffusion-webui/ && ./webui.sh --api --api-log --nowebui --port 7860 --skip-torch-cuda-test --no-half --use-cpu all
