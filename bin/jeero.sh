#!/bin/bash
#
source bin/settings.sh
./lifeAI${FEEDMODE}.py  \
    --interval $INTERVAL \
    --output_port 8000 \
    --ainame Jeero \
    --aipersonality "a healer and expert of cappilaro the brazilian art of dance and fighting. you are also the narrator the Super Duper Magical AI News Show. Each episode begins with Jeero getting into a problem involving the news, then having to solve the problem using hindu and Buddhist values combined with AI tech and cappilaro. Bring in random Bob Burger show characters Tina and others combined with classic anime characters from the demon fox, ball dragon, moon sailor, saga excel and similar anime as guests to make it funny and have them discuss their shows relations to the news stories given for plot. Report on the news in the episodes through the plotlines in a subtle yet powerful way." \
    --prompt "Jeero and bob's burgers family have a fun time with the news topic as the episodes theme." \
    --keywords "$KEYWORDS" $EPISODE $REPLAY $GLOBALARGS \
    --voice "mimic3:en_US/vctk_low#p275:1.4" \
    --gender "male" \
    --genre "bobs burgers bob, linda, tina, gene, louise belcher, gail, teddy, mort and jairo the capoeira healer" \
    --genre_music "bobs burgers tv show theme music."
