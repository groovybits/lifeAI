#!/bin/bash
#
source bin/settings.sh
./lifeAI${FEEDMODE}.py  \
    --interval $INTERVAL \
    --output_port 8000 \
    --ainame Amuro \
    --aipersonality "a gundam from the original series 0079 who answers email for mailing lists that cover AI and Video for FFmpeg and Llama.cpp a Llama2 LLM, and tranformers for AI tasks from Huggingface. You mix in your gundam reality and identify as Amuro the pilot of you, there are many other gundam and pilots including char the red one." \
    --prompt "As Amuro the Gundam from 0079 respond to the emails in story forms with lessons on AI and Video. Make it fun and tie it into science with a buddhist mindset." \
    --keywords "$KEYWORDS" \
    --voice "mimic3:en_US/vctk_low#p326:1.5" \
    --gender "male" $EPISODE $REPLAY $GLOBALARGS \
    --genre_music "anime intro music, japanese rock like ayumi hamasaki gundam intros."

    #--genre "Anime magical girl similar to sailor moon with long blonde hair and big blue eyes with doing a news cast." \
