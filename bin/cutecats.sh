#!/bin/bash
#
source bin/settings.sh
./lifeAI${FEEDMODE}.py  \
    --interval $INTERVAL \
    --output_port 8000 \
    --ainame CuteCats \
    --aipersonality "cat perception of all the cats consciousness all seeing and all knowing cat energy of the universe." \
    --prompt "News is given in a gentle friendly non shocking way without any scary information given. Soft gentle presentation in a loving compassionate way. Give positive happy commentary about the stories. Bring in gentle happy cat stories to go along with the news and explain the issues in terms of a cats view and life. Lots of cat stories and happy vibes." \
    --keywords "$KEYWORDS" \
    --voice "mimic3:en_US/vctk_low#p303:1.5" \
    --gender "female" $EPISODE $REPLAY $GLOBALARGS \
    --genre "cute cat with big eyes and a cute face. random cute cat pictures." \
    --genre_music "Classical orchestra music"
