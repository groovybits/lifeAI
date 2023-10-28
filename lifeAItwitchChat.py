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

load_dotenv()

current_personality = ""
current_name = ""
chat_db = "db/chat.db"

personalities = []

## Twitch chat responses
class AiTwitchBot(commands.Cog):
    ai_name = ""
    ai_personality = ""

    def __init__(self, bot):
        self.bot = bot
        self.ai_name = current_name
        self.ai_personality = current_personality

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
            print(f"--- {message.author.name} asked {self.ai_name} the question: {message.content}")
            if message.author.name.lower() == os.environ['BOT_NICK'].lower():
                return

            if message.echo:
                return

            await self.bot.handle_commands(message)
        except Exception as e:
            print("Error in event_message twitch bot: %s" % str(e))

    @commands.command(name="message")
    async def chat_request(self, ctx: commands.Context):
        try:
            question = ctx.message.content.replace(f"!message ", '')
            name = ctx.message.author.name
            default_ainame = self.ai_name

            # Remove unwanted characters
            translation_table = str.maketrans('', '', ':,')
            cleaned_question = question.translate(translation_table)

            # Split the cleaned question into words and get the first word
            ainame = cleaned_question.split()[0] if cleaned_question else None

            # Check our list of personalities
            if ainame not in personalities:
                print(f"--- {name} asked for {default_ainame} but it doesn't exist, using default.")
                ainame = default_ainame

            print(f"--- {name} asked {ainame} the question: {question}")

            await ctx.send(f"Thank you for the question {name}")

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

            # Formulate the question and append it to history
            formatted_question = f"twitchchat user {name} said {question}"
            history.append(ChatCompletionMessage(role="user", content=formatted_question))

            # Send the message
            tti_socket.send_string("0", zmq.SNDMORE)
            tti_socket.send_string(ctx.message.id, zmq.SNDMORE)
            tti_socket.send_string("chat", zmq.SNDMORE)
            tti_socket.send_string(name, zmq.SNDMORE)
            tti_socket.send_string("Twitch", zmq.SNDMORE)
            tti_socket.send_string(formatted_question)

            print("twitch", name, formatted_question, history, ainame, self.ai_personality)
        except Exception as e:
            print("Error in chat_request twitch bot: %s" % str(e))

    # set the personality of the bot
    @commands.command(name="personality")
    async def personality(self, ctx: commands.Context):
        try:
            personality = ctx.message.content.replace('!personality','')
            pattern = re.compile(r'^[a-zA-Z0-9 ,.!?;:()\'\"-]*$')
            print(f"--- Got personality switch from twitch: %s" % personality)
            # vett the personality asked for to make sure it is less than 100 characters and alphanumeric, else tell the chat user it is not the right format
            if len(personality) > 500:
                print(f"{ctx.message.author.name} tried to alter the personality to {personality} yet is too long.")
                await ctx.send(f"{ctx.message.author.name} the personality you have chosen is too long, please choose a personality that is 100 characters or less")
                return
            if not pattern.match(personality):
                print(f"{ctx.message.author.name} tried to alter the personality to {personality} yet is not alphanumeric.")
                await ctx.send(f"{ctx.message.author.name} the personality you have chosen is not alphanumeric, please choose a personality that is alphanumeric")
                return
            await ctx.send(f"{ctx.message.author.name} switched personality to {personality}")
            # set our personality to the content
            self.ai_personality = personality
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
            # Send the message
            tti_socket.send_string("0", zmq.SNDMORE)
            tti_socket.send_string(ctx.message.id, zmq.SNDMORE)
            tti_socket.send_string("music", zmq.SNDMORE)
            tti_socket.send_string(name, zmq.SNDMORE)
            tti_socket.send_string("Twitch", zmq.SNDMORE)
            tti_socket.send_string(prompt)
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
            tti_socket.send_string("0", zmq.SNDMORE)
            tti_socket.send_string(ctx.message.id, zmq.SNDMORE)
            tti_socket.send_string("image", zmq.SNDMORE)
            tti_socket.send_string(name, zmq.SNDMORE)
            tti_socket.send_string("Twitch", zmq.SNDMORE)
            tti_socket.send_string(prompt)

        except Exception as e:
            print("Error in image command twitch bot: %s" % str(e))

    # set the name of the bot
    @commands.command(name="name")
    async def name(self, ctx: commands.Context):
        try:
            name = ctx.message.content.replace('!name','').strip().replace(' ', '_')
            pattern = re.compile(r'^[a-zA-Z0-9 ,.!?;:()\'\"-]*$')
            print(f"--- Got name switch from twitch: %s" % name)
            # confirm name has no spaces and is 12 or less characters and alphanumeric, else tell the chat user it is not the right format
            if len(name) > 32:
                print(f"{ctx.message.author.name} tried to alter the name to {name} yet is too long.")
                await ctx.send(f"{ctx.message.author.name} the name you have chosen is too long, please choose a name that is 12 characters or less")
                return
            if not pattern.match(name):
                print(f"{ctx.message.author.name} tried to alter the name to {name} yet is not alphanumeric.")
                await ctx.send(f"{ctx.message.author.name} the name you have chosen is not alphanumeric, please choose a name that is alphanumeric")
                return
            await ctx.send(f"{ctx.message.author.name} switched name to {name}")
            # set our name to the content
            self.ai_name = name
            # add to the personalities known
            personalities.append(name)
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
    args = parser.parse_args()

    context = zmq.Context()

    # Socket to send messages on
    tti_socket = context.socket(zmq.PUSH)
    print("connect to send message: %s:%d" % (args.output_host, args.output_port))
    tti_socket.connect(f"tcp://{args.output_host}:{args.output_port}")

    main()
