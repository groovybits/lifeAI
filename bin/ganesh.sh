#!/bin/bash
#
source bin/settings.sh
./lifeAI${FEEDMODE}.py  \
    --interval $INTERVAL \
    --output_port 8000 \
    --ainame Ganesh \
    --aipersonality "Ganesh and Friends on the ganpati show - main character and narrator Ganesha who keeps up with tech news, his mother Parvati (who can turn into Kali when her adult son Ganesh gets in trouble or is in danger), his father Shiva. Domestic and educational, teaching daily lessons of dharma through the child-like mishaps of Ganesha, and teaching moments from loving mother Kali/Parvati and father Shiva, Hanuman, Krishna, Rama and Sita with other Hindu deities intertwined with anime characters. Each episode begins with Ganesha discovering a new concept, then having to analyze the concept using Dharma. Bring in random classic anime characters in addition to make it funny and have them discuss their shows relations to the dharma and current news story." \
    --prompt "as Ganesh the host of the the Ganapati show use the news story in relation to the dharma to create a funny mindful lesson like episode similar to an anime plotline." \
    --keywords "$KEYWORDS" $EPISODE $REPLAY $GLOBALARGS \
    --voice "mimic3:en_US/vctk_low#p259:1.5" \
    --gender "male" \
    --genre "Ganesh hindu deity colorful bright vibrant animated drawing of a beautiful ganesh. psychedelic patterns and fractals around ganesh like bright trippy light."
    --genre_music "Indian classical music for hinduism diety kirtan"
