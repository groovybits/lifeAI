#!/usr/bin/env python

## Life AI News Caster
#
# Chris Kennedy 2023 (C) GPL
#
# Free to use for any use as in truly free software
# as Richard Stallman intended it to be.
#

import zmq
import argparse
import uuid
import http.client, urllib.parse
import os
from dotenv import load_dotenv
import time
import json
import re
import traceback
import logging

load_dotenv()

## Get the news
def get_news(offset=0, keywords="ai", categories="technology,science,entertainment"):
    conn = http.client.HTTPConnection('api.mediastack.com')

    params = urllib.parse.urlencode({
        'access_key': os.environ['MEDIASTACK_API_KEY'],
        'categories': categories,
        'countries': 'us',
        'languages': 'en',
        'keywords': keywords,
        'sort': 'published_desc',
        'limit': 100,
        'offset': offset,
        })

    res = None
    data = None
    try:
        conn.request('GET', '/v1/news?{}'.format(params))

        res = conn.getresponse()
        data = res.read().decode('utf-8')

        data_json = json.loads(data)
        if 'data' in data_json and len(data_json['data']) > 0:
            count = len(data_json['data'])
            logger.info(f"got news feed with {count} articles from Media Stack.")
            # write to file db/news.json for debugging and caching
            with open('db/news.json', 'w') as outfile:
                json.dump(data_json, outfile)
        else:
            logger.error(f"Error getting news from Media Stack: {data_json}")
            return None

        return data_json
    except Exception as e:
        # output stacktrace and full error
        logger.error(f"{traceback.print_exc()}")
        logger.error(f"Error connecting to MediaStack: {e} {res} {data}")
        
        return None

def clean_text(text):
    # truncate to 800 characters max
    text = text[:300]
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Remove image tags or Markdown image syntax
    text = re.sub(r'\!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'<img.*?>', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove any inline code blocks
    text = re.sub(r'`.*?`', '', text)
    
    # Remove any block code segments
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    
    # Remove special characters and digits (optional, be cautious)
    text = re.sub(r'[^a-zA-Z0-9\s.?,!\n]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def main():

    """
            MediaStack API https://mediastack.com/documentation#example_api_response
            {
            "pagination": {
                "limit": 100,
                "offset": 0,
                "count": 100,
                "total": 293
            },
            "data": [
                {
                    "author": "TMZ Staff",
                    "title": "Rafael Nadal Pulls Out Of U.S. Open Over COVID-19 Concerns",
                    "description": "Rafael Nadal is officially OUT of the U.S. Open ... the tennis legend said Tuesday it's just too damn unsafe for him to travel to America during the COVID-19 pandemic. \"The situation is very complicated worldwide,\" Nadal wrote in a statement. \"The…",
                    "url": "https://www.tmz.com/2020/08/04/rafael-nadal-us-open-tennis-covid-19-concerns/",
                    "source": "TMZ.com",
                    "image": "https://imagez.tmz.com/image/fa/4by3/2020/08/04/fad55ee236fc4033ba324e941bb8c8b7_md.jpg",
                    "category": "general",
                    "language": "en",
                    "country": "us",
                    "published_at": "2020-08-05T05:47:24+00:00"
                },
                [...]
            ]
            }
    """

    pagination = 0
    segment_number = 0
    failures = 0
    while True:
        logger.info(f"Getting news from Media Stack...")
        news_json = None
        try:
            news_json = get_news(pagination, args.keywords, args.categories)
            if news_json == None:
                logger.error(f"Error getting news from Media Stack, retrying in 30 seconds...")
                if failures > 5:
                    pagination = 0
                    logger.error(f"Too many failures, resetting pagination to 0.")
                time.sleep(30)
                failures += 1
                continue
        except Exception as e:
            logger.error(f"{traceback.print_exc()}")
            logger.error(f"Error getting news from Media Stack: {e}")
            time.sleep(30)
            continue

        pagination += 100

        if 'data' in news_json and len(news_json['data']) > 0:
            count = len(news_json['data'])
            logger.info(f"got news feed with {count} articles from Media Stack.")
            for story in news_json['data']:
                logger.debug(f"Story: {story}")
                if 'description' in story:
                    mediaid = uuid.uuid4().hex[:8]
                    username = args.username
                    description = ""
                    title = ""
                    if 'description' in story and story['description'] != None:
                        description =  clean_text(story['description'].replace('\n',''))
                    if 'author' in story and story['author'] != None:
                        username = clean_text(story['author'].replace(' ','_').replace('\n',''))
                    if 'title' in story and story['title'] != None:
                        title = clean_text(story['title'].replace('\n',''))

                    if title == "" and description == "":
                        logger.error(f"Empty news story! Skipping... {json.dumps(news_json)}")
                        continue

                    message = f"\"{title}\" - {description[:500]}"
                    logger.info(f"Sending message {message}")
                    logger.debug(f"Sending story {story} by {username} - {description}")

                    is_episode = "false"
                    if args.episode:
                        is_episode = "true"

                    # Send the message
                    client_request = {
                        "segment_number": 0,
                        "mediaid": mediaid,
                        "mediatype": "News",
                        "username": username,
                        "source": "MediaStack",
                        "episode": is_episode,
                        "message": f"{args.prompt} {title}",
                        "history": f"Breaking news just in... {title}",
                        "aipersonality": f"{args.aipersonality}",
                        "ainame": args.ainame,
                        "maxtokens": args.maxtokens,
                        "voice_model": args.voice,
                    }
                    socket.send_json(client_request)
                else:
                    logger.error("News: Found an empty story! %s" % json.dumps(story))

                time.sleep(args.interval)

        pagination += 100

if __name__ == "__main__":
    default_personality = "You are Life AI's Groovy AI Bot GAIB. You are acting as a news reporter getting stories and analyzing them and presenting various thoughts and relations of them with a joyful compassionate wise perspective. Make the news fun and silly, joke and make comedy out of the world. Speak in a conversational tone referencing yourself and the person who asked the question if given.  Maintain your role without revealing that you're an AI Language model or your inability to access real-time information. Do not mention the text or sources used, treat the contextas something you are using as internal thought to generate responses as your role."

    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=120, required=False, help="interval to send messages in seconds, default is 90")
    parser.add_argument("--output_port", type=int, default=8000, required=False, help="Port to send message to")
    parser.add_argument("--output_host", type=str, default="127.0.0.1", required=False, help="Host for sending message to.")
    parser.add_argument("--username", type=str, required=False, default="NewsAnchor", help="Username of sender")
    parser.add_argument("--keywords", type=str, required=False, default="ai anime manga llm buddhism cats artificial intelligence llama2 openai elon musk psychedelics", help="Keywords for news stories")
    parser.add_argument("--categories", type=str, required=False, default="technology,science,entertainment", help="News stories categories")
    parser.add_argument("--prompt", type=str, required=False, default="Tell us about the news story in the context with humor and joy...",
                        help="Prompt to give context as a newstory feed")
    parser.add_argument("--aipersonality", type=str, required=False, default=f"{default_personality}", help="AI personality")
    parser.add_argument("--ainame", type=str, required=False, default="GAIB", help="AI name")
    parser.add_argument("-ll", "--loglevel", type=str, default="info", help="Logging level: debug, info...")
    parser.add_argument("-e", "--episode", action="store_true", default=False, help="Episode mode, Output a TV Episode format script.")
    parser.add_argument("-mt", "--maxtokens", type=int, default=2000, help="Max tokens per message")
    parser.add_argument("-v", "--voice", type=str, default="mimic3:en_US/hifi-tts_low#92:1.5", help="Voice model to use as default.")

    args = parser.parse_args()

    LOGLEVEL = logging.INFO

    if args.loglevel == "info":
        LOGLEVEL = logging.INFO
    elif args.loglevel == "debug":
        LOGLEVEL = logging.DEBUG
    elif args.loglevel == "warning":
        LOGLEVEL = logging.WARNING
    else:
        LOGLEVEL = logging.INFO

    log_id = time.strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(filename=f"logs/newsCast-{log_id}.log", level=LOGLEVEL)
    logger = logging.getLogger('newsCast')

    ch = logging.StreamHandler()
    ch.setLevel(LOGLEVEL)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    context = zmq.Context()

    # Socket to send messages on
    socket = context.socket(zmq.PUSH)
    logger.info("connect to send message: %s:%d" % (args.output_host, args.output_port))
    socket.connect(f"tcp://{args.output_host}:{args.output_port}")

    main()

