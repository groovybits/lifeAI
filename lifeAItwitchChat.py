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

load_dotenv()

current_personality = ""
current_name = ""
chat_db = "db/chat.db"

personalities = {}

## Twitch chat responses
class AiTwitchBot(commands.Cog):
    ai_name = ""
    ai_personality = ""

    def __init__(self, bot):
        self.bot = bot
        self.ai_name = current_name
        self.ai_personality = current_personality
        personalities[self.ai_name] = self.ai_personality

    ## Channel entrance for our bot
    async def event_ready(self):
        try:
            'Called once when the bot goes online.'
            print(f"{os.environ['BOT_NICK']} is online!")
            ws = self.bot._ws  # this is only needed to send messages within event_ready
            await ws.send_privmsg(os.environ['CHANNEL'], f"/me has landed!")
        except Exception as e:
            print("Error in event_ready twitch bot: %s" % str(e))

    ## Message sent in chat
    async def event_message(self, message):
        'Runs every time a message is sent in chat.'
        try:
            if message.author.name.lower() == os.environ['BOT_NICK'].lower():
                return

            # Ignore our own messages
            if message.echo:
                return

            print(f"--- Received message from {message.author.name}: {message.content}")
            await self.bot.handle_commands(message)
        except Exception as e:
            print("Error in event_message twitch bot: %s" % str(e))

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
            aipersonality = self.ai_personality

            # Check our list of personalities
            if ainame_request not in personalities:
                print(f"--- {name} asked for {ainame_request} but it doesn't exist, using default.")
                ctx.send(f"{name} the personality you have chosen is not in the list of personalities, please choose a personality that is in the list of personalities {json.dumps(personalities)}. You can create them using !name <name> <personality>.")
            else:
                ainame = ainame_request
                aipersonality = personalities[ainame]
                print(f"--- {name} set personality to {ainame} with personality {aipersonality}.")

            print(f"--- {name} asked {ainame} the question: {question}")

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
                print(f"Setting up DB for user {name}.")
                cursor.execute("INSERT INTO users (name) VALUES (?)", (name,))
                db_conn.commit()

            # Add the new message to the messages table
            if question != "...":
                cursor.execute("INSERT INTO messages (user, content) VALUES (?, ?)", (name, question))
                db_conn.commit()

            # Retrieve the chat history for this user
            cursor.execute("SELECT content FROM messages WHERE user = ? ORDER BY timestamp", (name,))
            dbdata = cursor.fetchall()
            history = [ChatCompletionMessage(role="user", content=d[0]) for d in dbdata]

            db_conn.close()

            # Send the message
            client_request = {
                "segment_number": "0",
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

            print(f"twitch client sent message:\n{client_request}\n")
        except Exception as e:
            print("Error in chat_request twitch bot: %s" % str(e))

    # set the personality of the bot
    @commands.command(name="personality")
    async def personality(self, ctx: commands.Context):
        try:
            personality = ctx.message.content.replace('!personality','')
            print(f"--- Got personality switch to personality: %s" % personality)
            if personality not in personalities:
                print(f"{ctx.message.author.name} tried to alter the personality to {personality} yet is not in the list of personalities.")
                await ctx.send(f"{ctx.message.author.name} the personality you have chosen is not in the list of personalities, please choose a personality that is in the list of personalities {json.dumps(personalities)}")
                return
            await ctx.send(f"{ctx.message.author.name} switched personality to {personality}")
            # set our personality to the content
            self.ai_personality = personalities[personality]
            self.ai_name = personality

        except Exception as e:
            print("Error in personality command twitch bot: %s" % str(e))

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
                "aipersonality": "You are a musician and will compose an amazing piece of music for us.",
                "ainame": "MusicGen",
            }
            socket.send_json(client_request)

            print(f"twitch client sent music request: {client_request} ")
        except Exception as e:
            print("Error in music command twitch bot: %s" % str(e))

    ## list personalities command - sends us a list of the personalities we have
    @commands.command(name="personalities")
    async def listpersonalities(self, ctx: commands.Context):
        try:
            # get the name of the person who sent the message
            name = ctx.message.author.name
            # send the list of personalities
            await ctx.send(f"{name} the personalities we have are {personalities}")
        except Exception as e:
            print("Error in listpersonalities command twitch bot: %s" % str(e))

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
                "aipersonality": "You are a digital artist and phtographer, you will compose an amazing piece of art or take an amazing photo image for us.",
                "ainame": "ImageGen",
            }
            socket.send_json(client_request)

            print(f"twitch client sent image request: {client_request} ")

        except Exception as e:
            print("Error in image command twitch bot: %s" % str(e))

    # set the name of the bot
    @commands.command(name="name")
    async def name(self, ctx: commands.Context):
        try:
            # format is "!name <name> <personality>"
            name = ctx.message.content.replace('!name','').strip().replace(' ', '_')
            name, personality = name.split(' ', 1)
            namepattern = re.compile(r'^[a-zA-Z0-9]*$')
            personalitypattern = re.compile(r'^[a-zA-Z0-9 ,.]*$')
            print(f"--- Got name switch from {ctx.author} for ai name: %s" % name)
            # confirm name has no spaces and is 12 or less characters and alphanumeric, else tell the chat user it is not the right format
            if len(name) > 32 or ' ' in name or len(personality) > 200:
                print(f"{ctx.message.author.name} tried to alter the name to {name} yet is too long or has spaces.")
                await ctx.send(f"{ctx.message.author.name} the name you have chosen is too long, please choose a name that is 12 characters or less")
                return
            if not namepattern.match(name) or not personalitypattern.match(personality):
                print(f"{ctx.message.author.name} tried to alter the name to {name} yet is not alphanumeric.")
                await ctx.send(f"{ctx.message.author.name} the name you have chosen is not alphanumeric, please choose a name that is alphanumeric")
                return
            await ctx.send(f"{ctx.message.author.name} created name {name}")
            # set our name to the content
            self.ai_name = name
            # add to the personalities known
            if name not in personalities:
                personalities[name] = personality
        except Exception as e:
            print("Error in name command twitch bot: %s" % str(e))

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
    parser.add_argument("--ai_name", type=str, required=False, default="Buddha", help="Name of the bot")
    parser.add_argument("--ai_personality", type=str,
                        required=False, 
                        default="Helpful wise boddisattva helping twitch chat users with their suffering and joy with equinimity and compassion.", 
                        help="Personality of the bot")
    args = parser.parse_args()

    context = zmq.Context()

    # Socket to send messages on
    socket = context.socket(zmq.PUSH)
    print("connect to send message: %s:%d" % (args.output_host, args.output_port))
    socket.connect(f"tcp://{args.output_host}:{args.output_port}")

    main()
