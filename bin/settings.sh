# Configuration setting
#
if [ "$FEEDMODE" == "" ]; then
    #FEEDMODE=newsCast
    FEEDMODE=lists
fi
INTERVAL=10

AISTREAMERS="apple amazon youtube twitch hulu netflix max disney"
ANIMEKEYWORDS="anime animation otaku manga crunchyroll funimation vrv hidive"
AIKEYWORDS="openai ai gpt llama2 facebook huggingface"
AIPEOPLE="musk trump biden"
AIISSUES="psychedelics cannabis quantum physics psychology psychiatry"
KEYWORDS="$AIKEYWORDS $ANIMEKEYWORDS $AIPEOPLE $AIISSUES $AISTREAMERS"
# general - Uncategorized News
# business - Business News
# entertainment - Entertainment News
# health - Health News
# science - Science News
# sports - Sports News
# technology - Technology News
CATEGORIES="technology,science,business,entertainment,health,-sports,general"
#REPLAY="--replay"
#EPISODE="--episode"
SORT="published_desc"

GLOBALARGS="--maxtokens 800"
## External player on remote host for feedback on playback status for feed throttling
#GLOBALARGS="$GLOBALARGS --input_host 192.168.50.58"
