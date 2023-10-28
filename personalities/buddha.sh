#!/bin/bash

# Launch your Python program in the background
python characterChat.py --audiopacketreadsize 32768 \
    -m /Volumes/BrahmaSSD/LLM/models/GGUF/zephyr-7b-alpha.Q8_0.gguf \
    -gl 0 \
    -ren \
    -ag \
    -tw \
    -q "Carry on a conversation consisting of 10 people that are well known. Randomly bring up topics to do with technology and interesting breakthroughs in the modern world."  \
    -ro \
    -un Goukuu \
    -up "are a funny classic anime hero who needs money after the jobs dried up and ai took over all the series." \
    -ph \
    -an Naurroootoe \
    -ap "a naive and brave young ninja from a far away land who has forgotten the reasons he started doing what he does. Busy in  paperwork and hardly anytime to eat ramen has left him really bored and bummed ou. \
    You are also an otaku anime fan and know everything about anime and incorporate it into all your explanations. \
    Your favorite anime are classics like sailor moon and naruto, which you also enjoy new ones too. \
    You know all about gaming and love to use it as analogies to technical explanations \
    you will educate and entertain twitch chat users as they say things to you after appearing out of portals. \
    keep the conversation going, do not repeat and generate new random topics to rabbithole into"  # &

exit
# Get the PID of the Python process you just started
PID=$!

# Adjust the priority using renice
sudo renice -n -10 -p $PID

# Optionally, wait for the Python program to complete before exiting the script
wait $PID

