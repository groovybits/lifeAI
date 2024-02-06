#!/bin/bash

while [ : ]; do
    # Use find to list files, then pipe to ls to sort by modification time
    playlist=$(find /Volumes/BrahmaSSD/music/AiGen/ -type f -exec ls -tr {} +)

    # Play sorted playlist with mpv
    for file in $playlist; do
        echo "Playing '$file'"
        mpv "$file" --audio-device="coreaudio/com.rogueamoeba.Loopback:FB46673C-58FA-4556-986F-7986B4D8C4E5" # Add your mpv options here
    done
done
