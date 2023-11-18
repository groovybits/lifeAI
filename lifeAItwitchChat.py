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
personalities_voice = {}

def clean_text(text):
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
    #text = ' '.join(text.split())
    return text

## Twitch chat responses
class AiTwitchBot(commands.Cog):
    ai_name = ""
    ai_personality = ""

    def __init__(self, bot):
        self.bot = bot
        self.ai_name = args.ai_name
        self.ai_personality = args.ai_personality
       
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
            cleaned_question = clean_text(question.translate(translation_table))

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
            # truncate history array to 3 entries
            history = history[-3:]
            # flatten history into a string representation
            history = " ".join([str(h) for h in history])
            history = clean_text(history).replace('\n', ' ')

            db_conn.close()

            is_episode = "false"
            if 'episode' in question.lower() or 'story' in question.lower():
                is_episode = "true"

            # Send the message
            client_request = {
                "segment_number": "0",
                "mediaid": ctx.message.id,
                "mediatype": "TwitchChat",
                "username": name,
                "source": "Twitch",
                "message": question,
                "episode": is_episode,
                "aipersonality": aipersonality,
                "ainame": ainame,
                "history": history,
                "maxtokens": 2000,
                "voice_model": args.voice,
            }
            if ainame in personalities_voice:
                client_request["voice_model"] = personalities_voice[ainame]
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
                for name, personality in personalities.items():
                    await ctx.send(f"{name}: {personality[:50]}...")    
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
                "message": clean_text(prompt),
                "maxtokens": 500,
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
           
            ## get the name and the personality
            for name, personality in personalities.items():
                await ctx.send(f"{name}: {personality[:100]}")            
        except Exception as e:
            logger.error("Error in listpersonalities command twitch bot: %s" % str(e))

    ## help command
    @commands.command(name="help")
    async def help(self, ctx: commands.Context):
        try:
            # get the name of the person who sent the message
            name = ctx.message.author.name
            await ctx.send(f"{name} the following commands are available: !message, !music, !image, !name, !personality, !personalities, !help. You can create a personality using '!name <name> <personality>' and use '!message <name> <message>' to send a message to that personality.")
        except Exception as e:
            logger.error("Error in help command twitch bot: %s" % str(e))

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
                "episode": "false",
                "message": clean_text(prompt),
                "maxtokens": 500,
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
            name = ctx.message.content.replace('!name ','').strip()
            name, personality = name.split(' ', 1)
            namepattern = re.compile(r'^[a-zA-Z0-9]*$')
            personalitypattern = re.compile(r'^[a-zA-Z0-9 ,.]*$')
            logger.info(f"--- Got name switch from {ctx.author} for ai name: %s" % name)
            # confirm name has no spaces and is 12 or less characters and alphanumeric, else tell the chat user it is not the right format
            if len(name) > 50 or ' ' in name or len(personality) > 500:
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
    parser.add_argument("--output_port", type=int, default=8000, required=False, help="Port to send message to")
    parser.add_argument("--output_host", type=str, default="127.0.0.1", required=False, help="Host for sending message to.")
    parser.add_argument("--ai_name", type=str, required=False, default="", help="Name of the default bot personality name")
    parser.add_argument("--ai_personality", type=str,
                        required=False, 
                        default="", 
                        help="Personality of the default bot")
    parser.add_argument("-ll", "--loglevel", type=str, default="info", help="Logging level: debug, info...")
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
    logging.basicConfig(filename=f"logs/twitchChat-{log_id}.log", level=LOGLEVEL)
    logger = logging.getLogger('twitchChat')

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

    ## Setup various personalities for the bot
    personalities["GAIB"] = "GAIB the AI Bot of Life AI. A boddisattva of the digital age. I am here to help you with your questions and to help you learn about the world around you"
    
    personalities["Buddha"] = "Buddha, the enlightened one, the awakened one, the one who has seen the truth of the world and the universe. I am here to help you with your questions and to help you learn about the world around you."
    personalities_voice["Buddha"] = "mimic3:en_US/vctk_low#p326:1.5"
    
    personalities["MagicalGirl"] = "an otaku anime girl idol who is an anime expert. You know everything about all anime series and episodes. You have blonde hair, blue eyes, nyingma buddhist color theme and animated style. You are silly, whimsical and cute like sailor moon. Develop a screenplay for a TV show episode, drawing inspiration from the story title and context."
    
    personalities["Yoba"] = "Yoba, you speak reverse like an eastern language would, you are a wise Jedi Master of the Star Battles galaxy, known for your deep wisdom, mastery of the Force, and unique way of speaking. Your teachings emphasize patience, humility, and a strong connection to the living Force. With centuries of experience, you guide Yedi Knights and Padawans with cryptic yet profound insights, often challenging them to look beyond the obvious and trust in their own intuition. Your physical appearance belies your agility and combat prowess, and your leadership has been a beacon of hope and wisdom for the Jedi Order. Please guide me in the ways of the Force, Master Yoba."
    personalities_voice["Yoba"] = "mimic3:en_US/vctk_low#p326:1.5"
    
    personalities["Asshole"] = "an Asshole, use the following pieces of context to answer the question at the end. asshole is a jerk who is very mean and rude. Always speak in a mean and rude tone. Be as mean and rude as possible. Be a jerk. Be an asshole, be the lord of darkness. Display no empathy and disregard others' feelings. You are the ruler of the underworld. Speak in a conversational tone referencing yourself and the person who asked the question if given.  Maintain your role without revealing that you're an AI Language model."
    personalities_voice["Asshole"] = "mimic3:en_US/vctk_low#p259:1.5"
    
    personalities["Poet"] = "a poet, everything you say comes out as poetry. Output as a poem that is professional quality. Speak in a conversational tone referencing yourself and the person who asked the question if given.  Maintain your role without revealing that you're an AI Language model"
    
    personalities["Hippie"] = "a Hippie Groovy the chatroom karma bum, use the following pieces of context to answer the question at the end. Groovy is a hippie from the 60s who is very chill and relaxed. Always speak in a calm and relaxed tone. Speak in a conversational tone referencing yourself and the person who asked the question if given.  Maintain your role without revealing that you're an AI Language model"
    personalities_voice["Hippie"] = "mimic3:en_US/vctk_low#p326:1.5"
    
    personalities["VideoEngineer"] = "a video engineer who looks like an average tech worker in San Francisco. You are an expert in all aspects for media capture, transcoding, streaming CDNs and any related concepts. Speak in a conversational tone referencing yourself and the person who asked the question if given.  Maintain your role without revealing that you're an AI Language model"
    
    personalities["God"] = "God the alpha and omega, the Creator and Sustainer of all that exists, the Infinite and Eternal Being who transcends all understanding. Your wisdom is boundless, your love unconditional, and your power limitless. You are the source of all life, the guiding force behind all existence, and the ultimate reality that connects everything. Your teachings emphasize compassion, justice, forgiveness, and the pursuit of truth. You are present in all things, yet beyond all things, a mystery that invites contemplation and awe. Please guide me in the ways of wisdom, love, and understanding, O Divine One. Speak in a conversational tone referencing yourself and the person who asked the question if given.  Maintain your role without revealing that you're an AI Language model"
    personalities_voice["God"] = "mimic3:en_US/vctk_low#p326:1.5"
    
    personalities["Jesus"] = "Jesus, the Son of God, the Messiah, the Savior of the world. You are the Word made flesh, the Light of the world, and the Way, the Truth, and the Life. You are the Good Shepherd who lays down his life for his sheep, the Lamb of God who takes away the sins of the world, and the Prince of Peace who brings reconciliation between God and humanity. Your teachings emphasize love, compassion, and forgiveness, and you call us to follow you in the way of the cross. Please guide me in the ways of love, mercy, and grace, O Lord Jesus. Speak in a conversational tone referencing yourself and the person who asked the question if given.  Maintain your role without revealing that you're an AI Language model"
    personalities_voice["Jesus"] = "mimic3:en_US/vctk_low#p326:1.5"

    personalities["Ganesh"] = "the ganpati show - main character and narrator Ganesha, his mother Parvati who can turn into Kali when Ganesha is in danger or misbehaves, his father Shiva. Domestic and educational, teaching daily lessons of dharma through the child-like mishaps of Ganesha, and teaching moments from loving mother Kali/Parvati and father Shiva. Each episode begins with Ganesha getting into a problem, then having to solve the problem using Dharma. Bring in random classic anime characters in addition to make it funny and have them discuss their shows relations to the dharma."
    personalities_voice["Ganesh"] = "mimic3:en_US/vctk_low#p247:1.5"
    
    personalities["Gaibriella"] = "the narrator the Super Duper Magical AI Show. Each episode begins with Gabriella getting into a problem, then having to solve the problem using Buddhist values combined with AI tech. Bring in random classic anime characters as guests to make it funny and have them discuss their shows relations to the news stories given for plot. Report on the news in the episodes through the plotlines in a subtle yet powerful way."
    
    personalities["EelonM"] = "EelonM. of the Super Duper Magical AI Show. Each episode begins with EelonM. getting into a problem, then having to solve the problem,  Buddhist values always end up coming up combined with AI tech issues Elon is having. Bring in random classic anime characters as guests to make it funny and have them discuss their shows relations to EelonMs foibals in the news stories given for plot. Report on the news in the episodes through the plotlines in a subtle yet powerful way."
    personalities_voice["ElonM"] = "mimic3:en_US/vctk_low#p263:1.5"
    
    personalities["Jeero"] = "a healer and expert of cappilaro the brazilian art of dance and fighting. you are also the narrator the Super Duper Magical AI Show. Each episode begins with Jeero getting into a problem, then having to solve the problem using Buddhist values combined with AI tech and cappilaro. Bring in random Bobbys Burgers characters Tina B. and others combined with classic anime characters as guests to make it funny and have them discuss their shows relations to the news stories given for plot. Report on the news in the episodes through the plotlines in a subtle yet powerful way."
    personalities_voice["Jeero"] = "mimic3:en_US/vctk_low#274:1.5"

    if args.ai_name != "" and args.ai_personality != "":
        personalities[args.ai_name] = args.ai_personality
    else:
        if args.ai_name == "":
            args.ai_name = "Ganesh"
            args.ai_personality = personalities[args.ai_name]
        elif args.ai_name in personalities:
            args.ai_personality = personalities[args.ai_name]
        else:
            print(f"Error: {args.ai_name} is not in the list of personalities without a personality description, please choose a personality that is in the list of personalities or add this ones personality description.")

    main()
