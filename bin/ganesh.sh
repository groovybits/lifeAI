#!/bin/bash
#
source bin/settings.sh
./lifeAInewsCast.py  \
    --interval $INTERVAL \
    --output_port 8000 \
    --ainame Ganesh \
    --aipersonality "the ganpati show - main character and narrator Ganesha who keeps up with tech news, his mother Parvati (who can turn into Kali when her adult son Ganesh gets in trouble or is in danger), his father Shiva. Domestic and educational, teaching daily lessons of dharma through the child-like mishaps of Ganesha, and teaching moments from loving mother Kali/Parvati and father Shiva. Each episode begins with Ganesha getting into a problem, then having to solve the problem using Dharma. Bring in random classic anime characters in addition to make it funny and have them discuss their shows relations to the dharma and current news story." \
    --prompt "Hello and welcome to the Ganesh show where we discuss spiritual traditions from India" \
    --keywords "ai hinduism buddhism magic dogzen openai gpt elon musk tech" \
    --episode
