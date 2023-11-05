#!/bin/bash
#
# Stream on MacOS to twitch
# Get your stream key and add it, not here but use env values
#
## add value to .env
source .env
#TWITCH_STREAM_KEY=
VIDEO_DEV=2
AUDIO_DEV=2

ffmpeg -y -hide_banner \
    -probesize 500M -pix_fmt uyvy422 \
    -f avfoundation -i $VIDEO_DEV:$AUDIO_DEV \
    -loglevel warning \
    -pix_fmt yuv420p \
    -vcodec h264_videotoolbox \
    -preset slow \
    -b:v 3000k \
    -tune animation \
    -acodec aac -ar 48000 \
    -b:a 128k -ac 2 \
    -maxrate 3000k -minrate 3000k -bufsize 3000k \
    -g 120 \
    -keyint_min 30 \
    -flags:v +global_header \
    -flags:a +global_header \
        -f flv \
            rtmp://sfo.contribute.live-video.net/app/$TWITCH_STREAM_KEY
