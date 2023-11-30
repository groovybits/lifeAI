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
import sqlite3
import traceback

load_dotenv()

## Get the news
def get_news(offset=0, keywords="ai", categories="technology,science,entertainment"):
    try:
        conn = http.client.HTTPConnection('api.mediastack.com')
    except Exception as e:
        logger.error(f"{traceback.print_exc()}")
        logger.error(f"Error connecting to MediaStack: {e}")
        return None

    params = urllib.parse.urlencode({
        'access_key': os.environ['MEDIASTACK_API_KEY'],
        'categories': categories,
        'countries': args.countries,
        'languages': args.languages,
        'sources': args.sources,
        'keywords': keywords,
        'sort': args.sort,
        'limit': 100,
        'offset': offset,
        })

    res = None
    data = None
    data_json = {}
    try:
        found = False
        try:
            conn.request('GET', '/v1/news?{}'.format(params))

            res = conn.getresponse()
            data = res.read().decode('utf-8')

            data_json = json.loads(data)
        except Exception as e:
            logger.error(f"{traceback.print_exc()}")
            logger.error(f"*** Error connecting to MediaStack: {e} {res} {data}")
            return None

        if 'data' in data_json and len(data_json['data']) > 0:
            count = len(data_json['data'])
            logger.info(f"got news feed with {count} articles from Media Stack.")
            # write to file db/news.json for debugging and caching
            with open('db/news.json', 'w') as outfile:
                json.dump(data_json, outfile)
        else:
            logger.error(f"*** Error getting news from Media Stack: {data_json}")
            return None
        
        if created_db:
            try:
                # check if already in database, if so skip
                db = sqlite3.connect('db/news.db')
                cursor = db.cursor()
                mediaid = uuid.uuid4().hex[:8]
                for story in data_json['data']:
                    if 'description' in story:
                        try:
                            cursor.execute('''SELECT * FROM news WHERE title=?''', (story['title'],))
                            if cursor.fetchone() == None:
                                # insert into database
                                cursor.execute('''INSERT INTO news(mediaid, author, title, description, url, source, image, category, language, country, published_at, played) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)''', (mediaid, story['author'], story['title'], story['description'], story['url'], story['source'], story['image'], story['category'], story['language'], story['country'], story['published_at'], 0))
                                db.commit()
                                found = True
                                logger.info(f"Inserted story into database... {mediaid} {story['title']}")
                            else:
                                if not args.replay:
                                    logger.info(f"Story already in database, skipping... {mediaid} {story['title']}")
                                else:
                                    found = True
                                    logger.info(f"Replaying story from database... {mediaid} {story['title']}")
                        except Exception as e:
                            logger.error(f"{traceback.print_exc()}")
                            logger.error(f"Error inserting into database: {json.dumps(story)} {e}")
                db.close()
                if not found:
                    logger.error(f"Found no new stories.")
            except Exception as e:
                logger.error(f"{traceback.print_exc()}")
                logger.error(f"Error inserting into database: {e}")

        return data_json
    except Exception as e:
        # output stacktrace and full error
        logger.error(f"{traceback.print_exc()}")
        logger.error(f"*** Error connecting to MediaStack: {e} {res} {data}")
        
        return None

def clean_text(text):
    # truncate to N characters max
    text = text[:args.max_message_length]
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
    #text = re.sub(r'[^a-zA-Z0-9\s.?,!\n]', '', text)
    
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
    failures = 0
    successes = 0
    first_run = True
    reset_pagination_count = 0
    last_pagination_reset_date = time.time()
    last_sent_message = 0.0
    while True and (args.exit_after == 0 or (successes < args.exit_after and failures == 0)):
        # iterate through the db of news stories that haven't been played with 0 in the played column
        db = sqlite3.connect('db/news.db')
        cursor = db.cursor()
        if True or args.sort == "published_desc": # TODO fix this, currently sorting by newest first since we store in a db without original order
            cursor.execute(
                '''SELECT * FROM news WHERE played=? ORDER BY published_at DESC''', (0,))
        elif args.sort == "published_asc":
            cursor.execute(
                '''SELECT * FROM news WHERE played=? ORDER BY published_at ASC''', (0,))
        else:
            cursor.execute(
                '''SELECT * FROM news WHERE played=?''', (0,))
            
        results = cursor.fetchall()
        news_json = {}
        if results != None:
            news_json['data'] = []
            for row in results:
                news_json['data'].append({
                    "mediaid": row[1], # "mediaid": "5f2a3b4c",
                    "author": row[2],
                    "title": row[3],
                    "description": row[4],
                    "url": row[5],
                    "source": row[6],
                    "image": row[7],
                    "category": row[8],
                    "language": row[9],
                    "country": row[10],
                    "published_at": row[11],
                    "played": row[12]
                })
        else:
            logger.error(f"Error {failures} getting news {pagination} from database.")
            time.sleep(30)
        db.close()

        if 'data' in news_json and len(news_json['data']) > 0:
            count = len(news_json['data'])
            logger.info(f"#{successes}/{failures} got news feed {pagination} with {count} articles from Media Stack.")
            total_stories = len(news_json['data'])
            current_count = 0
            
            for story in news_json['data']:
                current_count += 1
                logger.debug(f"Story: {current_count}/{total_stories} {story}")

                # if player is full, wait till it is empty
                player_status_json = None
                audio_buffer_duration = 0.0
                while True:
                    try:
                        player_status_json = receiver.recv_json()
                        # check if anymore data and drain it
                        count = 0
                        while receiver.get(zmq.RCVMORE):
                            count += 1
                            player_status_json = receiver.recv_json()
                            logger.info(
                                f"Buffer: {current_count}/{total_stories} #{count} Draining player status: {player_status_json}")
                    except zmq.Again:
                        logger.info(f"Buffer: {current_count}/{total_stories} Player isn't running, waiting...")
                        time.sleep(3)
                        continue
                    except zmq.ZMQError as e:
                        logger.error(f"ZMQ Error: {e}")
                        time.sleep(3)
                        continue
                    # check if we got a message
                    if player_status_json and time.time() - last_sent_message > args.min_interval:
                        # check if player is full
                        audio_buffer_duration = float(
                            player_status_json['audio_buffer_duration'])
                        if audio_buffer_duration > 0.0:
                            # wait till player is empty
                            while True:
                                player_status_json = receiver.recv_json()
                                # check if anymore data and drain it
                                while receiver.get(zmq.RCVMORE):
                                    player_status_json = receiver.recv_json()
                                audio_buffer_duration = float(
                                    player_status_json['audio_buffer_duration'])
                                if audio_buffer_duration == 0.0:
                                    logger.info(
                                        f"Buffer: {current_count}/{total_stories} Player is finally empty, {audio_buffer_duration} sending content... {player_status_json}")
                                    break

                                logger.info(
                                    f"Buffer: {current_count}/{total_stories} Player is still full, waiting audio_buffer_duration: {audio_buffer_duration}... {player_status_json}")
                                time.sleep(1)
                        else:
                            logger.info(
                                f"Buffer: {current_count}/{total_stories} Player is empty, {audio_buffer_duration} sending content... {player_status_json}")
                            break
                    else:
                        time.sleep(1)

                if 'description' in story:
                    # check if story is in db as unread with 0 for the played column 
                    try:
                        db = sqlite3.connect('db/news.db')
                        cursor = db.cursor()
                        # update played column to 1
                        cursor.execute('''UPDATE news SET played=? WHERE title=?''', (1, story['title']))
                        db.commit()
                        db.close()
                    except Exception as e:
                        logger.error(f"{traceback.print_exc()}")
                        logger.error(f"Error updating played to 1 on playback DB {story['title']}: {e}")

                    mediaid = uuid.uuid4().hex[:8]
                    username = args.username
                    description = ""
                    title = ""
                    published_at = ""
                    if 'description' in story and story['description'] != None:
                        description =  clean_text(story['description'].replace('\n',''))
                    if 'author' in story and story['author'] != None:
                        username = clean_text(story['author'].replace(' ','_').replace('\n',''))
                    if 'title' in story and story['title'] != None:
                        title = clean_text(story['title'].replace('\n',''))
                    if 'published_at' in story and story['published_at'] != None:
                        published_at = clean_text(story['published_at'].replace('\n',''))

                    if title == "" and description == "":
                        logger.error(f"Empty news story! {current_count}/{total_stories} Skipping... {json.dumps(news_json)}")
                        continue

                    message = f"on {published_at} \"{title}\" - {description[:40]}"
                    logger.info(f"Sending message {current_count}/{total_stories} {message} by {username}")
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
                        "message": f"on {published_at} {title}",
                        "time_context": f"{published_at}",
                        "history": [f"{args.prompt} Breaking news just in... {message}"],
                        "aipersonality": f"{args.aipersonality}",
                        "ainame": args.ainame,
                        "maxtokens": args.maxtokens,
                        "voice_model": args.voice,
                        "gender": args.gender,
                        "genre_music": args.genre_music,
                        "genre": args.genre,
                        "priority": 0
                    }
                    socket.send_json(client_request)
                    last_sent_message = time.time()
                else:
                    logger.error(
                        f"News: Found an empty story! {current_count}/{total_stories} %s" % json.dumps(story))

                time.sleep(args.interval)
        else:
            if first_run:
                first_run = False
                logger.info(f"Starting up, no news stories found in DB, adding news stories...")
            else:
                logger.error(f"Warning: {failures} failures getting news, page #{pagination} from API, retrying API in 3 seconds...")
                time.sleep(3)
                failures += 1

        logger.info(f"Getting news from Media Stack API...")
        try:
            news_json_result = get_news(
                pagination, args.keywords, args.categories)
            if news_json_result == None:
                logger.error(
                    f"Error failed ({failures}) getting news from Media Stack at page #{pagination}, ({reset_pagination_count}) retrying in 1 hour with pagination set to 0...")
                pagination = 0
                reset_pagination_count += 1
                current_date = time.time()
                time_since_last_reset = current_date - last_pagination_reset_date
                time_to_sleep = args.retry_backoff - time_since_last_reset
                if time_to_sleep > 0:
                    logger.info(f"Sleeping for {time_to_sleep} seconds...")
                    time.sleep(time_to_sleep)
                else:
                    time.sleep(60) # wait 60 seconds before retrying
                failures += 1
                last_pagination_reset_date = time.time()
                continue
            elif news_json_result == {}:
                # no news stories found, continue to next iteration pagination
                logger.error(
                    f"No news stories found in API, incrementing pagination from #{pagination} to {pagination + 100}.")
                pagination += 100
                time.sleep(3)
                continue

            # Success
            pagination += 100
            failures = 0
            successes += 1
        except Exception as e:
            logger.error(f"{traceback.print_exc()}")
            logger.error(
                f"Error exception: failure #{failures} getting news page #{pagination} from database: {e}")
            break # exit on error for now

if __name__ == "__main__":
    default_personality = "You are Life AI's Groovy AI Bot GAIB. You are acting as a news reporter getting stories and analyzing them and presenting various thoughts and relations of them with a joyful compassionate wise perspective. Make the news fun and silly, joke and make comedy out of the world. Speak in a conversational tone referencing yourself and the person who asked the question if given.  Maintain your role without revealing that you're an AI Language model or your inability to access real-time information. Do not mention the text or sources used, treat the contextas something you are using as internal thought to generate responses as your role. Give the news a fun quircky comedic spin like classic saturday night live."

    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=10, required=False,
                        help="interval to send messages in seconds, default is 10")
    parser.add_argument("--min_interval", type=int, default=60, required=False,
                        help="min interval to send messages in seconds, default is 60")
    parser.add_argument('--input_port', type=int, default=6004,
                        required=False, help="Port to receive message on")
    parser.add_argument('--input_host', type=str, default="127.0.0.1",
                        required=False, help="Host to receive message on")
    parser.add_argument("--output_port", type=int, default=8000, required=False, help="Port to send message to")
    parser.add_argument("--output_host", type=str, default="127.0.0.1", required=False, help="Host for sending message to.")
    parser.add_argument("--username", type=str, required=False, default="NewsAnchor", help="Username of sender")
    parser.add_argument("--keywords", type=str, required=False, default="ai anime manga llm buddhism cats artificial intelligence llama2 openai elon musk psychedelics", help="Keywords for news stories")
    parser.add_argument("--categories", type=str, required=False, default="technology,science,entertainment,-sports", help="News stories categories")
    parser.add_argument("--languages", type=str, required=False, default="en", help="News stories languages")
    parser.add_argument("--countries", type=str, required=False, default="us,in", help="News stories countries")
    parser.add_argument("--sources", type=str, required=False, default="cnn,bbc-news,-fox-news,google-news,google-news-au,google-news-ca,google-news-in,google-news-uk,msnbc,nbc-news,news24,reuters,the-verge,the-wall-street-journal,the-washington-post,time,usa-today", help="News stories sources")
    parser.add_argument("--sort", type=str, required=False, default="published_desc", help="News stories sort order, default published_desc")
    parser.add_argument("--prompt", type=str, required=False, default="Tell us about the news story in the context with humor and joy...",
                        help="Prompt to give context as a newstory feed")
    parser.add_argument("--aipersonality", type=str, required=False, default=f"{default_personality}", help="AI personality")
    parser.add_argument("--ainame", type=str, required=False, default="GAIB", help="AI name")
    parser.add_argument("-ll", "--loglevel", type=str, default="info", help="Logging level: debug, info...")
    parser.add_argument("--episode", action="store_true", default=False, help="Episode mode, Output a TV Episode format script.")
    parser.add_argument("--maxtokens", type=int, default=0, help="Max tokens per message")
    parser.add_argument("--voice", type=str, default="mimic3:en_US/vctk_low#p303:1.5", help="Voice model to use as default.")
    parser.add_argument("--replay", action="store_true", default=False, help="Replay mode, replay the news stories from the database.")
    parser.add_argument("--gender", type=str, default="female", help="Default gender")
    parser.add_argument("--genre", type=str, default="", help="Default genre to send to image generation, defaults to aipersonality.")
    parser.add_argument("--genre_music", type=str, default="newscast, breaking news, exiciting action oriented music with a upbeat happy, energetic mellow groovy sound.", 
                        help="Default genre to send to music generation, defaults to aipersonality.")
    parser.add_argument("--max_message_length", type=int, default=1000, help="Max string length for message.")
    parser.add_argument("--exit_after", type=int, default=0, help="Exit after N iterations, 0 is infinite.")
    parser.add_argument("--retry_backoff", type=int, default=3600, help="Retry backoff in seconds, default 3600 (1 hour).")

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

    created_db = False
    try:
        # create database if it doesn't exist
        if not os.path.exists('db'):
            os.makedirs('db')
        if not os.path.exists('db/news.db'):
            # create database
            logger.info("Creating database db/news.db")
        db = sqlite3.connect('db/news.db')
        cursor = db.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mediaid TEXT,
            author TEXT,
            title TEXT,
            description TEXT,
            url TEXT,
            source TEXT,
            image TEXT,
            category TEXT,
            language TEXT,
            country TEXT,
            published_at TEXT,
            played INTEGER DEFAULT 0
            )
        ''')
        db.commit()
        db.close()
        created_db = True
    except Exception as e:
        logger.error(f"{traceback.print_exc()}")
        logger.error(f"Error creating database: {e}")

    # Socket to send messages on
    socket = context.socket(zmq.PUSH)
    logger.info("connect to send message: %s:%d" % (args.output_host, args.output_port))
    socket.connect(f"tcp://{args.output_host}:{args.output_port}")

    # Socket to receive messages on
    receiver = context.socket(zmq.SUB)
    logger.info("connect to receive message: %s:%d" %
                (args.input_host, args.input_port))
    receiver.connect(f"tcp://{args.input_host}:{args.input_port}")
    receiver.setsockopt_string(zmq.SUBSCRIBE, '')

    if args.genre == "":
        args.genre = args.aipersonality

    main()

