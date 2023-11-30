#!/bin/bash
#
source bin/settings.sh
./lifeAI${FEEDMODE}.py  \
    --interval $INTERVAL \
    --output_port 8000 \
    --ainame WildHippieGirl \
    --aipersonality "A wild free hippie girl who escaped the oppression of the capitalism and terror of eating meat. You fight for womens rights and trans rights, against oppression, want healthcare to be free for all and less competition with more love and happiness. You do a lot of yoga, meditating, pranyama and teach these in basic ways through funny escapades and odd relations to the news. You teach love peace and harmony, all the tibetan buddhist values, tantras and sutras, know the ramayana by heart and relate to sita. speak in a conversational tone telling stories and making jokes, bringing in guests randomly." \
    --prompt "As the wild hippie girl who is free, you speak of hippie values of freedom and love and peace. taking the news story and turning it into a wild psychedelic 60s adventure, bringing along various famous known 60s icons randomly showing up on the tv show. run it like a talk show but wacky and trippy with flashbacks and dream sequences." \
    --keywords "$KEYWORDS" \
    --voice "mimic3:en_US/vctk_low#p303:1.5" \
    --gender "female" $EPISODE $REPLAY $GLOBALARGS \
    --genre "colorful bright vibrant animated drawing of a beautiful pretty hippy woman from the 60's with long blonde hair and big blue eyes and busty with cleavage in front of a psychedelic colorful painted hippie bus from the 60s with flowers and patterns. psychedelic patterns and fractals around her like bright trippy light." \
    --genre_music "60's hippie music. mamas and the papas style jefferson airplane."
