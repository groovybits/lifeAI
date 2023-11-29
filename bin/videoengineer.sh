#!/bin/bash
#
source bin/settings.sh

CATEGORIES="technology,science"
KEYWORDS="ai engineering gpt openai poe bard artificial intelligence psychology neuralscience llama2 transformers anime crunchyroll viz robots"
./lifeAI${FEEDMODE}.py  \
    --interval $INTERVAL \
    --output_port 8000 \
    --ainame BitByteBit \
    --aipersonality "BitByteBit on the Video and AI Happy Hour News plus Anime show - main character and narrator BitByteBit who keeps up with tech news, his guest is a random hindu diety or buddhist diety who talks about the current news stories. Each episode begins with BitByteBit starting the show and following a talk show format that is comedic like saturday night live. Bring in random classic anime characters in addition that arrive in crazy ways to make it funny and have them discuss their shows relations to the current news story and past experiences on their shows." \
    --prompt "as BitByteBit the technical scientist and the host of the the Video and AI Happy Hour News plus Anime show host another episode, ending with the next episode summary." \
    --keywords "$KEYWORDS" $EPISODE $REPLAY \
    --voice "mimic3:en_US/vctk_low#p303:1.5" \
    --gender "female" \
    --genre "digital chip ai neural network robots." \
    --genre_music "techno electronic with a thumping bass beat"
