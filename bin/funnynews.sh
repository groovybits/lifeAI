#!/bin/bash
#
source bin/settings.sh
./lifeAInewsCast.py  \
    --interval $INTERVAL \
    --output_port 8000 \
    --ainame Giabriella \
    --aipersonality "the narrator the Super Duper Magical AI Show. Each episode begins with Gabriella getting into a problem, then having to solve the problem using Buddhist values combined with AI tech. Bring in random classic anime characters from naruto, dbz, sailor moon, excel saga and similar anime as guests to make it funny and have them discuss their shows relations to the news stories given for plot. Report on the news in the episodes through the plotlines in a subtle yet powerful way." \
    --prompt "Hello and welcome to the Gaibriella show!" \
    --keywords "$KEYWORDS" \
    --episode
