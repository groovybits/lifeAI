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

    conn.request('GET', '/v1/news?{}'.format(params))

    res = conn.getresponse()
    data = res.read()

    print(f"Got back {data[:120]} from Media Stack")

    return data.decode('utf-8')

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
    while True:
        segment_number += 1
        news = get_news(0, args.keywords, args.categories)
        news_json = json.loads(news)

        if 'data' in news_json and len(news_json['data']) > 0:
            count = len(news_json['data'])
            print(f"got news feed with {count} articles from Media Stack.")
            for story in news_json['data']:
                print(f"Story: {story}")
                if 'description' in story:
                    message_id = uuid.uuid4().hex[:8]
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
                    

                    message = f"{args.prompt}\n\n{username} reported that {title}:\n\nStory: {description}"
                    print(f"Sending message {message}")

                    # Send the message
                    tti_socket.send_string(str(segment_number), zmq.SNDMORE)
                    tti_socket.send_string(message_id, zmq.SNDMORE)
                    tti_socket.send_string("news", zmq.SNDMORE)
                    tti_socket.send_string(args.username, zmq.SNDMORE)
                    tti_socket.send_string("MediaStackNews", zmq.SNDMORE)
                    tti_socket.send_string(message)
                else:
                    print("News: Found an empty story! %s" % json.dumps(story))

                time.sleep(args.interval)

        pagination += 100

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=120, required=False, help="interval to send messages in seconds, default is 120")
    parser.add_argument("--output_port", type=int, default=1500, required=False, help="Port to send message to")
    parser.add_argument("--output_host", type=str, default="127.0.0.1", required=False, help="Host for sending message to.")
    parser.add_argument("--username", type=str, required=False, default="NewsAnchor", help="Username of sender")
    parser.add_argument("--keywords", type=str, required=False, default="ai anime buddhism cats", help="Keywords for news stories")
    parser.add_argument("--categories", type=str, required=False, default="technology,science,entertainment", help="News stories categories")
    parser.add_argument("--prompt", type=str, required=False, default="News Story:",
                        help="Prompt to give context as a newstory feed")
    args = parser.parse_args()

    context = zmq.Context()

    # Socket to send messages on
    tti_socket = context.socket(zmq.PUSH)
    print("connect to send message: %s:%d" % (args.output_host, args.output_port))
    tti_socket.connect(f"tcp://{args.output_host}:{args.output_port}")

    main()

