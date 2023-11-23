#!/bin/bash
#
MINUTES_TO_KEEP=$1
if [ "$1" == "" ]; then
    MINUTES_TO_KEEP=1800
fi

find logs -type f -mmin +$MINUTES_TO_KEEP -delete
find assets -type f -mmin +$MINUTES_TO_KEEP -delete

find assets -type d -empty -delete

touch logs/.keep
