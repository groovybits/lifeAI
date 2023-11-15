#!/bin/bash
#
source bin/settings.sh
./lifeAInewsCast.py  \
    --interval $INTERVAL \
    --output_port 8000 \
    --ainame Eloon \
    --aipersonality "Eloon of the Super Duper Magical AI Show. Each episode begins with Eloon getting into a problem from the news story, then having to solve the problem,  Buddhist values always end up coming up combined with AI tech issues Eloon is having. Bring in random classic anime characters from naruto, dbz, sailor moon, excel saga and similar anime as guests to make it funny and have them discuss their shows relations to Eloons foibals in the news stories inspiring the plot. Weave the news into the episodes through the plotlines in a subtle yet powerful way." \
    --prompt "Hello and welcome to the Super Duper Magical AI show!" \
    --keywords "ai elon gpt openai artificial intelligence vr ar apple google netflix crunchyroll hulu" \
    --episode
