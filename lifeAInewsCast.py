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
def get_news(offset=0, keywords="ai anime buddhism cats", categories="technology,science,entertainment"):
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
            print(f"got news feed with {count} articles from Media Stack.")
        else:
            print(f"Error getting news from Media Stack: {data_json}")
            return None

        return data_json
    except Exception as e:
        print(f"Error connecting to MediaStack: {e} {res} {data}")
        # output stacktrace and full error
        traceback.print_exc()
        return None

def clean_text(text):
    # This regular expression pattern will match any character that is NOT a lowercase or uppercase letter or a space.
    pattern = re.compile(r'[^a-zA-Z\s1-9\.,\?!\-]')
    # re.sub will replace these characters with an empty string, effectively removing them.
    cleaned_text = re.sub(pattern, '', text)
    # truncate to 800 characters max
    cleaned_text = cleaned_text[:800]
    return cleaned_text

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
        print(f"Getting news from Media Stack...")
        news_json = get_news(pagination, args.keywords, args.categories)
        if news_json == None:
            print(f"Error getting news from Media Stack, retrying in 30 seconds...")
            if failures > 5:
                pagination = 0
                print(f"Too many failures, resetting pagination to 0.")
            time.sleep(30)
            failures += 1
            continue

        segment_number += 1
        pagination += 100

        if 'data' in news_json and len(news_json['data']) > 0:
            count = len(news_json['data'])
            print(f"got news feed with {count} articles from Media Stack.")
            for story in news_json['data']:
                print(f"Story: {story}")
                if 'description' in story:
                    mediaid = uuid.uuid4().hex[:8]
                    username = args.username
                    description = ""
                    title = ""
                    if 'description' in story and story['description'] != None:
                        description =  clean_text(story['description'])
                    if 'author' in story and story['author'] != None:
                        username = story['author'].replace(' ','_')
                    if 'title' in story and story['title'] != None:
                        title = story['title']

                    if title == "" and description == "":
                        print(f"Empty news story! Skipping... {json.dumps(news_json)}")
                        continue

                    #scrub description very well for any odd characters or non speaking words
                    

                    message = f"{args.prompt}{username} reported \"{title}\" - {description}\n\n"
                    print(f"Sending message {message}")

                    # Send the message
                    client_request = {
                        "segment_number": segment_number,
                        "mediaid": mediaid,
                        "mediatype": "news",
                        "username": username,
                        "source": "lifeAI",
                        "message": message,
                        "aipersonality": args.aipersonality,
                        "ainame": args.ainame,
                        "history": [],
                    }
                    socket.send_json(client_request)
                else:
                    print("News: Found an empty story! %s" % json.dumps(story))

                time.sleep(args.interval)

        pagination += 100

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=60, required=False, help="interval to send messages in seconds, default is 120")
    parser.add_argument("--output_port", type=int, default=1500, required=False, help="Port to send message to")
    parser.add_argument("--output_host", type=str, default="127.0.0.1", required=False, help="Host for sending message to.")
    parser.add_argument("--username", type=str, required=False, default="NewsAnchor", help="Username of sender")
    parser.add_argument("--keywords", type=str, required=False, default="ai anime buddhism cats", help="Keywords for news stories")
    parser.add_argument("--categories", type=str, required=False, default="technology,science,entertainment", help="News stories categories")
    parser.add_argument("--prompt", type=str, required=False, default="News Story just in... ",
                        help="Prompt to give context as a newstory feed")
    parser.add_argument("--aipersonality", type=str, required=False, default="GAIB the AI Bot of Life AI, I am sending an interesting news article for analysis.", help="AI personality")
    parser.add_argument("--ainame", type=str, required=False, default="GAIB", help="AI name")
    parser.add_argument("-ll", "--loglevel", type=str, default="info", help="Logging level: debug, info...")

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
    logging.basicConfig(filename=f"logs/newCast-{log_id}.log", level=LOGLEVEL)
    logger = logging.getLogger('GAIB')

    ch = logging.StreamHandler()
    ch.setLevel(LOGLEVEL)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    context = zmq.Context()

    # Socket to send messages on
    socket = context.socket(zmq.PUSH)
    print("connect to send message: %s:%d" % (args.output_host, args.output_port))
    socket.connect(f"tcp://{args.output_host}:{args.output_port}")

    main()

