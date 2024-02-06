#!/bin/bash

while true; do
    for file in /Volumes/BrahmaSSD/music/AiGen/*.*; do
        echo "Playing '$file'"
        mpv "$file" --audio-device="coreaudio/com.rogueamoeba.Loopback:FB46673C-58FA-4556-986F-7986B4D8C4E5"
        sleep 1
    done
done

