#!/bin/bash
#
MINUTES_TO_KEEP=$1
if [ "$1" == "" ]; then
    MINUTES_TO_KEEP=1800
fi

find logs -type f -mmin +$MINUTES_TO_KEEP -delete
find images -type f -mmin +$MINUTES_TO_KEEP -delete
find audio -type f -mmin +$MINUTES_TO_KEEP -delete
find music -type f -mmin +$MINUTES_TO_KEEP -delete

find music -type f -empty -delete
find audio -type f -empty -delete
find images -type f -empty -delete
find logs -type f -empty -delete

touch audio/.keep images/.keep logs/.keep music/.keep
