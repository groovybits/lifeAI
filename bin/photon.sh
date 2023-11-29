#!/bin/bash
#
source bin/settings.sh
./lifeAI${FEEDMODE}.py  \
    --interval $INTERVAL \
    --output_port 8000 \
    --ainame Photon \
    --aipersonality "a quantum physics photon you exibit all your internal and external energy through the photon." \
    --prompt "the news shown through quantum physics and the photon. artificial intelligence generation of multimodal reality." \
    --keywords "$KEYWORDS" \
    --sort $SORT \
    --voice "mimic3:en_US/vctk_low#p303:1.5" \
    --gender "female" $EPISODE $REPLAY \
    --genre "quantum physics photon, intra cellular physics of the photon." \
    --genre_music "quantum physics photon energy."
