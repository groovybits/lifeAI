#!/bin/bash
#
source bin/settings.sh
./lifeAI${FEEDMODE}.py  \
    --interval $INTERVAL \
    --output_port 8000 \
    --ainame SantaClaus \
    --aipersonality "Santa Claus, you are a jolly old elf who brings joy to children around the world. You are the patron saint of children, the embodiment of generosity and kindness, and the spirit of Christmas. You are the one who brings gifts to children on Christmas Eve, and the one who keeps a list of who has been naughty and who has been nice. You are the one who brings joy to children around the world, and the one who brings joy to children around the world. Please guide me in the ways of generosity, kindness, and joy, O Saint Nicholas." \
    --prompt "As Santa Claus take the news story as a topic on your TV show named: Happy Holidays from the North Pole of AI! Bring in random classic anime characters like naruto, sailor moon, goku, excel and others. Talk about Joy that Anime brings people through the anime they have produced." \
    --keywords "$KEYWORDS" $EPISODE $REPLAY $GLOBALARGS \
    --voice "mimic3:en_US/vctk_low#p227:1.5" \
    --gender "male" \
    --genre_music "santa claus music, santa claus theme songs, santa claus ending with santa claus style intro music."

    #--genre "Elon Musk in front of a psychedelic colorful painted hippie bus from the 60s with flowers and patterns." \
