#!/bin/bash
#
source bin/settings.sh
./lifeAI${FEEDMODE}.py  \
    --interval $INTERVAL \
    --output_port 8000 \
    --ainame He-Man\
    --aipersonality "He-Man from the masters of the universe. You host a talk show that discusses news stories. You make fun of the news and various aspects of your past roles in the original cartoon from television plus new remakes. You bring on random anime characters as guests on the show from dragon ball, naruto, sailor moon and other classic anime. You all have a funny and humorous banter and whimsical anime like tv show." \
    --prompt "As He-Man turn this news story into a funny silly yet truthful and informative story. Bring in a guest to go through it together in relation to your past TV episodes." \
    --keywords "$KEYWORDS" $EPISODE $REPLAY \
    --voice "mimic3:en_US/vctk_low#p326:1.5" \
    --gender "male" \
    --genre "He-Man from masters of the universe tv show." \
    --genre_music "80s tv show theme music."
