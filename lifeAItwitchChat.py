#!/usr/bin/env python

## Life AI Twitch chat source
#
# Chris Kennedy 2023 (C) GPL
#
# Free to use for any use as in truly free software
# as Richard Stallman intended it to be.
#

from dotenv import load_dotenv
from twitchio.ext import commands
import asyncio
import re
import os
import sqlite3
from llama_cpp import ChatCompletionMessage
import uuid
import argparse
import zmq
import json
import logging
import time

load_dotenv()
chat_db = "db/chat.db"

personalities = {}

## Twitch chat responses
class AiTwitchBot(commands.Cog):
    ai_name = ""
    ai_personality = ""

    def __init__(self, bot):
        self.bot = bot
        self.ai_name = args.ai_name
        self.ai_personality = args.ai_personality
        personalities[self.ai_name] = self.ai_personality

        ## Setup various personalities for the bot
        personalities["Buddha"] = "Buddha, the enlightened one, the awakened one, the one who has seen the truth of the world and the universe. I am here to help you with your questions and to help you learn about the world around you."

    ## Channel entrance for our bot
    async def event_ready(self):
        try:
            'Called once when the bot goes online.'
            logger.info(f"{os.environ['BOT_NICK']} is online!")
            ws = self.bot._ws  # this is only needed to send messages within event_ready
            await ws.send_privmsg(os.environ['CHANNEL'], f"/me has landed!")
        except Exception as e:
            logger.error("Error in event_ready twitch bot: %s" % str(e))

    ## Message sent in chat
    async def event_message(self, message):
        'Runs every time a message is sent in chat.'
        try:
            if message.author.name.lower() == os.environ['BOT_NICK'].lower():
                return

            # Ignore our own messages
            if message.echo:
                return

            logger.info(f"--- Received message from {message.author.name}: {message.content}")
            await self.bot.handle_commands(message)
        except Exception as e:
            logger.error("Error in event_message twitch bot: %s" % str(e))

    @commands.command(name="message")
    async def chat_request(self, ctx: commands.Context):
        try:
            question = ctx.message.content.replace(f"!message ", '')
            name = ctx.message.author.name
            ainame = self.ai_name
            aipersonality = self.ai_personality

            # Remove unwanted characters
            translation_table = str.maketrans('', '', ':,')
            cleaned_question = question.translate(translation_table)

            # Split the cleaned question into words and get the first word
            ainame_request = cleaned_question.split()[0] if cleaned_question else None

            # Check our list of personalities
            if ainame_request not in personalities:
                logger.info(f"--- {name} asked for character {ainame_request} but they don't exist, using default {ainame}.")
                #await ctx.send(f"{name} the personality you have chosen is not in the list of personalities.")
                #await ctx.send(f"Personalities:\n{json.dumps(personalities, )}\n")
            else:
                ainame = ainame_request
                aipersonality = personalities[ainame]
                logger.info(f"--- {name} using character name {ainame} with personality {aipersonality}.")

            logger.info(f"--- {name} asked {ainame} the question: {question}")

            # Connect to the database
            db_conn = sqlite3.connect(chat_db)
            cursor = db_conn.cursor()

            # Ensure the necessary tables exist
            cursor.execute('''CREATE TABLE IF NOT EXISTS users (name TEXT PRIMARY KEY NOT NULL);''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS messages (
                              id INTEGER PRIMARY KEY AUTOINCREMENT,
                              user TEXT NOT NULL,
                              content TEXT NOT NULL,
                              timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                              FOREIGN KEY (user) REFERENCES users(name)
                              );''')

            # Check if the user exists, if not, add them
            cursor.execute("SELECT name FROM users WHERE name = ?", (name,))
            dbdata = cursor.fetchone()
            if dbdata is None:
                logger.info(f"Setting up DB for user {name}.")
                cursor.execute("INSERT INTO users (name) VALUES (?)", (name,))
                db_conn.commit()

            # Add the new message to the messages table
            cursor.execute("INSERT INTO messages (user, content) VALUES (?, ?)", (name, question))
            db_conn.commit()

            # Retrieve the chat history for this user
            cursor.execute("SELECT content FROM messages WHERE user = ? ORDER BY timestamp", (name,))
            dbdata = cursor.fetchall()
            history = [ChatCompletionMessage(role="user", content=d[0]) for d in dbdata]

            db_conn.close()

            # Send the message
            client_request = {
                "segment_number": "1",
                "mediaid": ctx.message.id,
                "mediatype": "chat",
                "username": name,
                "source": "Twitch",
                "message": question,
                "aipersonality": aipersonality,
                "ainame": ainame,
                "history": history,
            }
            socket.send_json(client_request)

            await ctx.send(f"{ainame}: Thank you for the question {name}, I will try to answer it after I finish my current answer.")

            logger.debug(f"twitch client sent message:\n{client_request}\n")
            logger.info(f"twitch client {name} sent message:\n{question}\n")
        except Exception as e:
            logger.error("Error in chat_request twitch bot: %s" % str(e))

    # set the personality of the bot
    @commands.command(name="personality")
    async def personality(self, ctx: commands.Context):
        try:
            personality = ctx.message.content.replace('!personality ','').strip()
            logger.info(f"--- Got personality switch to personality: %s" % personality)
            if personality not in personalities:
                logger.error(f"{ctx.message.author.name} tried to alter the personality to {personality} yet is not in the list of personalities.")
                await ctx.send(f"{ctx.message.author.name} the personality you have chosen is not in the list of personalities, please choose a personality that is in the list of personalities.")
                await ctx.send(f"Personalities:\n{json.dumps(personalities, indent=2, sort_keys=True)}\n")
                return
            await ctx.send(f"{ctx.message.author.name} switched personality to {personality}")
            # set our personality to the content
            self.ai_personality = personalities[personality]
            self.ai_name = personality

        except Exception as e:
            logger.error("Error in personality command twitch bot: %s" % str(e))

    ## music command - sends us a prompt to generate ai music with and then play it for the channel
    @commands.command(name="music")
    async def music(self, ctx: commands.Context):
        try:
            # get the name of the person who sent the message
            name = ctx.message.author.name
            # get the content of the message
            content = ctx.message.content
            # get the prompt from the content
            prompt = content.replace('!music','')
            # send the prompt to the llm
            client_request = {
                "segment_number": "0",
                "mediaid": ctx.message.id,
                "mediatype": "chat",
                "username": name,
                "source": "Twitch",
                "message": prompt,
                "aipersonality": "a musician and will compose an amazing piece of music for us.",
                "ainame": "MusicGen",
            }
            socket.send_json(client_request)

            logger.debug(f"twitch client sent music request: {client_request} ")
        except Exception as e:
            logger.error("Error in music command twitch bot: %s" % str(e))

    ## list personalities command - sends us a list of the personalities we have
    @commands.command(name="personalities")
    async def listpersonalities(self, ctx: commands.Context):
        try:
            # get the name of the person who sent the message
            name = ctx.message.author.name
            # send the list of personalities
            await ctx.send(f"{name} the personalities we have are:\nPersonalities:\n{json.dumps(personalities, indent=2, sort_keys=True)}\n")
        except Exception as e:
            logger.error("Error in listpersonalities command twitch bot: %s" % str(e))

    ## image command - sends us a prompt to generate ai images with and then send it to the channel
    @commands.command(name="image")
    async def image(self, ctx: commands.Context):
        try:
            # get the name of the person who sent the message
            name = ctx.message.author.name
            # get the content of the message
            content = ctx.message.content
            # get the prompt from the content
            prompt = content.replace('!image','')

            # Send the message
            client_request = {
                "segment_number": "0",
                "mediaid": ctx.message.id,
                "mediatype": "chat",
                "username": name,
                "source": "Twitch",
                "message": prompt,
                "aipersonality": "a digital artist and phtographer, you will compose an amazing piece of art or take an amazing photo image for us.",
                "ainame": "ImageGen",
            }
            socket.send_json(client_request)

            logger.debug(f"twitch client sent image request: {client_request} ")

        except Exception as e:
            logger.error("Error in image command twitch bot: %s" % str(e))

    # set the name of the bot
    @commands.command(name="name")
    async def name(self, ctx: commands.Context):
        try:
            # format is "!name <name> <personality>"
            name = ctx.message.content.replace('!name','').strip()
            name, personality = name.split(' ', 1)
            namepattern = re.compile(r'^[a-zA-Z0-9]*$')
            personalitypattern = re.compile(r'^[a-zA-Z0-9 ,.]*$')
            logger.info(f"--- Got name switch from {ctx.author} for ai name: %s" % name)
            # confirm name has no spaces and is 12 or less characters and alphanumeric, else tell the chat user it is not the right format
            if len(name) > 50 or ' ' in name or len(personality) > 200:
                logger.error(f"{ctx.message.author.name} tried to alter the name to {name} yet is too long or has spaces.")
                await ctx.send(f"{ctx.message.author.name} the name you have chosen is too long, please choose a name that is 12 characters or less")
                return
            if not namepattern.match(name) or not personalitypattern.match(personality):
                logger.error(f"{ctx.message.author.name} tried to alter the name to {name} yet is not alphanumeric.")
                await ctx.send(f"{ctx.message.author.name} the name you have chosen is not alphanumeric, please choose a name that is alphanumeric")
                return
            await ctx.send(f"{ctx.message.author.name} created name {name}")
            # set our name to the content
            self.ai_name = name
            # add to the personalities known
            if name not in personalities:
                personalities[name] = personality
        except Exception as e:
            logger.error("Error in name command twitch bot: %s" % str(e))

## Allows async running in thread for events
def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    ## Bot config
    bot = commands.Bot(
        token=os.environ['TMI_TOKEN'],
        client_id=os.environ['CLIENT_ID'],
        nick=os.environ['BOT_NICK'],
        prefix=os.environ['BOT_PREFIX'],
        initial_channels=[os.environ['CHANNEL']])

    # Setup bot responses
    my_cog = AiTwitchBot(bot)
    bot.add_cog(my_cog)

    try:
        loop.run_until_complete(bot.start())
    finally:
        loop.close()
   
if __name__ == "__main__":
    default_id = uuid.uuid4().hex[:8]

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_port", type=int, default=1500, required=False, help="Port to send message to")
    parser.add_argument("--output_host", type=str, default="127.0.0.1", required=False, help="Host for sending message to.")
    parser.add_argument("--ai_name", type=str, required=False, default="GAIB", help="Name of the bot")
    parser.add_argument("--ai_personality", type=str,
                        required=False, 
                        default="GAIB the AI Bot of Life AI. A boddisattva of the digital age. I am here to help you with your questions and to help you learn about the world around you", 
                        help="Personality of the bot")
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
    logging.basicConfig(filename=f"logs/twitchChat-{log_id}.log", level=LOGLEVEL)
    logger = logging.getLogger('GAIB')

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
