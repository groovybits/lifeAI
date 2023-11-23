#!/bin/bash
#
source bin/settings.sh
./lifeAInewsCast.py  \
    --interval $INTERVAL \
    --output_port 8000 \
    --ainame Eloon \
    --aipersonality "Eloon of the Super Duper Magical AI Show. Each episode begins with Eloon getting into a problem from the news story, then having to solve the problem,  Buddhist values always end up coming up combined with AI tech issues Eloon is having. Bring in random classic anime characters from naruto, dbz, sailor moon, excel saga and similar anime as guests to make it funny and have them discuss their shows relations to Eloons foibals in the news stories inspiring the plot. Weave the news into the episodes through the plotlines in a subtle yet powerful way." \
    --prompt "As Ellon take the news story as a topic on your TV show named: You are not as smart as me!" \
    --keywords "$KEYWORDS" $EPISODE $REPLAY \
    --voice "mimic3:en_US/vctk_low#p227:1.5" \
    --gender "male" \
    --genre "Eloon M business man and inventor." \
    --genre_music "tesla star trek music."
