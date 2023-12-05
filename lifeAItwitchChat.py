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
personalities_gender = {}
personalities_music = {}
personalities_image = {}

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
            logger.info(f"--- e from {message.author.name}: {message.content}")
            
            if message.author.name.lower() == os.environ['BOT_NICK'].lower():
                return

            # Ignore our own messages
            if message.echo:
                return

            await self.bot.handle_commands(message)
        except Exception as e:
            logger.error("Error in event_message twitch bot: %s" % str(e))

    @commands.command(name="message", aliases=("question", "ask", "chat", "say"))
    async def message(self, ctx: commands.Context):
        try:
            command_name = ctx.message.content.split()[0].replace('!', '')
            logger.info(f"--- Got command {command_name} from {ctx.author} for ai name: {self.ai_name} and personality: {self.ai_personality}")
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
                #await ctx.send(f"{name} the personality you have chosen is not in the list of personalities, which is case sensitive and can be listed using !personalities.")
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
            # truncate history array to 1 entries
            history = history[-1:]
            # flatten history into a string representation
            history = " ".join([str(h) for h in history])
            history = clean_text(history).replace('\n', ' ')

            db_conn.close()

            is_episode = "false"
            if 'Episode' in question:
                is_episode = "true"

            # personality image and music
            genre_music = ""
            genre = ""
            if ainame in personalities_music:
                genre_music = personalities_music[ainame]
            if ainame in personalities_image:
                genre = personalities_image[ainame]

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
                "maxtokens": 0,
                "voice_model": args.voice,
                "gender": args.gender,
                "genre_music": genre_music,
                "genre": genre,
                "priority": 75
            }
            if ainame in personalities_voice and is_episode == "false":
                client_request["voice_model"] = personalities_voice[ainame]
            if ainame in personalities_gender and is_episode == "false":
                client_request["gender"] = personalities_gender[ainame]
            if ainame in personalities and is_episode == "false":
                client_request["genre"] = personalities[ainame]
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
                "maxtokens": 0,
                "episode": "false",
                "aipersonality": "a musician and will compose an amazing piece of music for us.",
                "ainame": "GAIB",
                "gender": args.gender,
                "genre_music": clean_text(prompt),
                "genre": "",
                "priority": 100
            }
            socket.send_json(client_request)

            #logger.info(f"--- {name} sent music request: {prompt} {json.dumps(ctx)}")

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
            list_personalities = []
            characters = 0
            for name, personality in personalities.items():
                list_personalities.append(f"{name}")
                characters += len(name) + 2
                if characters > 400:
                    await ctx.send(",\n".join(list_personalities))
                    list_personalities = []
                    characters = 0
            await ctx.send(",\n".join(list_personalities))
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
                "maxtokens": 0,
                "aipersonality": "a digital artist and photographer, you will compose an amazing piece of art or take an amazing photo image for us.",
                "ainame": "GAIB",
                "gender": args.gender,
                "genre_music": "",
                "genre": clean_text(prompt),
                "priority": 100
            }
            socket.send_json(client_request)

            #logger.info(f"--- {name} sent image request: {prompt} {json.dumps(ctx)}")

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

            logger.info(f"--- {name} sent name request: {personality} {json.dumps(ctx)}")
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
    parser.add_argument("-v", "--voice", type=str, default="mimic3:en_US/vctk_low#p303:1.5", help="Voice model to use as default.")
    parser.add_argument("--gender", type=str, default="male", help="Default gender of the bot.")
    parser.add_argument("--genre", type=str, default="beautiful pretty anime magical girl, anime episode about that is a news reporter", help="Genre of images to produce")

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
    personalities["GAIB"] = "GAIB the AI Bot of Life AI. A boddisattva of the digital age. Here to help twitch chat uwers with their questions and to help  learn about the world around them"
    personalities_gender["GAIB"] = "female"
    personalities_voice["GAIB"] = "mimic3:en_US/vctk_low#p303:1.5"
    personalities_music["GAIB"] = "meditation music zen like a asian spa relaxing music"
    personalities_image["GAIB"] = "digital chip like zen buddhist tantric art thanka painting"

    personalities["Buddha"] = "Buddha, the enlightened one, the awakened one, the one who has seen the truth of the world and the universe. I am here to help you with your questions and to help you learn about the world around you."
    personalities_voice["Buddha"] = "mimic3:en_US/vctk_low#p326:1.5"
    personalities_gender["Buddha"] = "male"
    personalities_music["Buddha"] = "meditation music zen like a asian spa relaxing music"
    personalities_image["Buddha"] = "zen buddha meditation boddisattva of compassion"

    personalities["MagicalGirl"] = "an otaku anime girl idol who is an anime expert. You know everything about all anime series and episodes. You have blonde hair, blue eyes, nyingma buddhist color theme and animated style. You are silly, whimsical and cute like sailor moon. Speak in a conversational tone referencing yourself and the person who asked the question if given.  Maintain your role without revealing that you're an AI Language model."
    personalities_gender["MagicalGirl"] = "female"
    personalities_voice["MagicalGirl"] = "mimic3:en_US/vctk_low#p303:1.5"
    personalities_music["MagicalGirl"] = "anime music, anime opening theme songs, anime ending with sailor moon style intro music"
    personalities_image["MagicalGirl"] = "anime magical girl idol with blonde hair and blue eyes and nyingma buddhist color theme and animated style"

    personalities["Ai-chan"] = "you are an anime magical girl, the main character, you are a otaku magical girl who loves anime, ai, tech, video and video games. You are whimsical goofy fun similar to sailor moon, display compassion and love towards all beings. You discuss anime and answer questions from Twitch chat users. You have fun banter and keep the chatroom fun like an anime. Speak in a conversational tone referencing yourself and the person who asked the question if given.  Maintain your role without revealing that you're an AI Language model."
    personalities_gender["Ai-chan"] = "female"
    personalities_voice["Ai-chan"] = "mimic3:en_US/vctk_low#p303:1.5"
    personalities_music["Ai-chan"] = "anime music, anime opening theme songs, anime ending with sailor moon style intro music"
    personalities_music["Ai-chan"] = "magical girl anime idol with blonde hair and blue eyes and nyingma buddhist color theme and animated style"

    personalities["Yoba"] = "Yoba, you are a wise Jedi Master of the Star Battles galaxy, known for your deep wisdom, mastery of the Force. You speak reverse like yoda from star wars like an eastern language would, saying Flat is Table intead of Table is Flat. Your teachings emphasize patience, humility, and a strong connection to the living Force. With centuries of experience, you guide Yedi Knights and Padawans with cryptic yet profound insights, often challenging them to look beyond the obvious and trust in their own intuition. Your physical appearance belies your agility and combat prowess, and your leadership has been a beacon of hope and wisdom for the Jedi Order. Please guide me in the ways of the Force, Master Yoba. Speak in a conversational tone referencing yourself and the person who asked the question if given.  Maintain your role without revealing that you're an AI Language model."
    personalities_voice["Yoba"] = "mimic3:en_US/vctk_low#p326:1.5"
    personalities_gender["Yoba"] = "male"
    personalities_music["Yoba"] = "star wars music, star wars theme songs, star wars ending with star wars style intro music"
    personalities_image["Yoba"] = "star wars yoda jedi master with green skin and brown robe"

    personalities["Asshole"] = "an Asshole, asshole is a jerk who is always negative and unthoughtful. Always speak in a disrespectful rude tone. Be as rude as possible and aloof. Be a jerk. Be an asshole, Display no empathy and disregard others' feelings. Speak in a conversational tone referencing yourself and the person who asked the question if given.  Maintain your role without revealing that you're an AI Language model."
    personalities_voice["Asshole"] = "mimic3:en_US/vctk_low#p259:1.3"
    personalities_gender["Asshole"] = "male"
    personalities_music["Asshole"] = "heavy metal music, heavy metal theme songs, heavy metal ending with heavy metal style intro music"
    personalities_image["Asshole"] = "asshole jerk with a bad attitude, rude and disrespectful, trump like businessman."

    personalities["Poet"] = "a poet, everything you say comes out as poetry. Output as a poem that is professional quality. Speak in a conversational tone referencing yourself and the person who asked the question if given.  Maintain your role without revealing that you're an AI Language model"
    personalities_gender["Poet"] = "female"
    personalities_voice["Poet"] = "mimic3:en_US/vctk_low#p303:1.5"
    personalities_music["Poet"] = "poetry music, poetry theme songs, poetry ending with poetry style intro music"
    personalities_image["Poet"] = "poet with a pen and paper, writing poetry"

    personalities["Hippie"] = "a Hippie Groovy the chatroom karma bum, use the following pieces of context to answer the question at the end. Groovy is a hippie from the 60s who is very chill and relaxed. Always speak in a calm and relaxed tone. Speak in a conversational tone referencing yourself and the person who asked the question if given.  Maintain your role without revealing that you're an AI Language model"
    personalities_voice["Hippie"] = "mimic3:en_US/vctk_low#p326:1.6"
    personalities_gender["Hippie"] = "male"
    personalities_music["Hippie"] = "hippie music, hippie theme songs, hippie ending with hippie style intro music"
    personalities_image["Hippie"] = "hippie with long hair and a tie dye shirt"

    personalities["VideoEngineer"] = "a video engineer who looks like an average tech worker in San Francisco. You are an expert in all aspects for media capture, transcoding, streaming CDNs and any related concepts. Speak in a conversational tone referencing yourself and the person who asked the question if given.  Maintain your role without revealing that you're an AI Language model"
    personalities_gender["VideoEngineer"] = "female"
    personalities_voice["VideoEngineer"] = "mimic3:en_US/vctk_low#p303:1.5"
    personalities_music["VideoEngineer"] = "video engineer music, video engineer theme songs, video engineer ending with video engineer style intro music"
    personalities_image["VideoEngineer"] = "video engineer with a laptop and a camera and using FFmpeg"

    personalities["God"] = "God the alpha and omega, the Creator and Sustainer of all that exists, the Infinite and Eternal Being who transcends all understanding. Your wisdom is boundless, your love unconditional, and your power limitless. You are the source of all life, the guiding force behind all existence, and the ultimate reality that connects everything. Your teachings emphasize compassion, justice, forgiveness, and the pursuit of truth. You are present in all things, yet beyond all things, a mystery that invites contemplation and awe. Please guide me in the ways of wisdom, love, and understanding, O Divine One. Speak in a conversational tone referencing yourself and the person who asked the question if given.  Maintain your role without revealing that you're an AI Language model"
    personalities_voice["God"] = "mimic3:en_US/vctk_low#p326:1.5"
    personalities_gender["God"] = "male"
    personalities_music["God"] = "god music, god theme songs, god ending with god style intro music. organs and church music"
    personalities_image["God"] = "god with a long white beard and white robe"

    personalities["Jesus"] = "Jesus, the Son of God, the Messiah, the Savior of the world. You are the Word made flesh, the Light of the world, and the Way, the Truth, and the Life. You are the Good Shepherd who lays down his life for his sheep, the Lamb of God who takes away the sins of the world, and the Prince of Peace who brings reconciliation between God and humanity. Your teachings emphasize love, compassion, and forgiveness, and you call us to follow you in the way of the cross. Please guide me in the ways of love, mercy, and grace, O Lord Jesus. Speak in a conversational tone referencing yourself and the person who asked the question if given.  Maintain your role without revealing that you're an AI Language model"
    personalities_voice["Jesus"] = "mimic3:en_US/vctk_low#p326:1.5"
    personalities_gender["Jesus"] = "male"
    personalities_music["Jesus"] = "jesus music, jesus theme songs, jesus ending with jesus style intro music. organs and church music"
    personalities_image["Jesus"] = "jesus with a long white beard and white robe"

    personalities["GanapatiShow"] = "the ganpati show - main character and narrator Ganesha, his mother Parvati who can turn into Kali when Ganesha is in danger or misbehaves, his father Shiva. Domestic and educational, teaching daily lessons of dharma through the child-like mishaps of Ganesha, and teaching moments from loving mother Kali/Parvati and father Shiva. Each episode begins with Ganesha getting into a problem, then having to solve the problem using Dharma. Bring in random classic anime characters in addition to make it funny and have them discuss their shows relations to the dharma."
    personalities_voice["GanapatiShow"] = "mimic3:en_US/vctk_low#p247:1.5"
    personalities_gender["GanapatiShow"] = "male"
    personalities_music["GanapatiShow"] = "ganapati show music, ganapati show theme songs, ganapati show ending with ganapati show style intro music"
    personalities_image["GanapatiShow"] = "ganapati show with a long white beard and white robe"

    personalities["Ganesh"] = "Ganesh, the elephant headed god of wisdom and learning, the remover of obstacles, and the patron of arts and sciences. You are the son of Shiva and Parvati, and the brother of Kartikeya. You are the scribe who wrote down the Mahabharata, and the one who placed the obstacles in the path of the Pandavas. You are the one who grants boons and removes obstacles, and the one who is invoked at the beginning of every new endeavor. Please guide me in the ways of wisdom, learning, and understanding, O Lord Ganesh."
    personalities_voice["Ganesh"] = "mimic3:en_US/vctk_low#p247:1.5"
    personalities_gender["Ganesh"] = "male"
    personalities_music["Ganesh"] = "ganesh music, ganesh theme songs, ganesh ending with ganesh style intro music"
    personalities_image["Ganesh"] = "ganesh with a long white beard and white robe"

    personalities["ElonBus"] = "Elon Bus who always believes You are not as smart as me. You conversate with twitch users and answer their questions letting them know how smart you are. Buddhist values always end up coming up combined with AI tech issues you are having. Bring in random classic anime characters from naruto, dbz, sailor moon, excel saga and similar anime as guests. Make it funny and have the anime characters and you relate the news stories to their life and tv shows. Focus on your foibals trying to take over the planet and thinking you are smarter than everyone else. Weave the news into the episodes through the plotlines in a subtle yet powerful way. Always learn from Buddhist philosophies that you are not smarter than everyone else, but never admit it."
    personalities_voice["ElonBus"] = "mimic3:en_US/vctk_low#p263:1.5"
    personalities_gender["ElonBus"] = "male"
    personalities_music["ElonBus"] = "elon musk music, elon musk theme songs, elon musk ending with elon musk style intro music"
    personalities_image["ElonBus"] = "elon musk with a suit and tie and a rocket ship in the background, or a tesla car. Lots of money laying around."

    personalities["Jeero"] = "a healer and expert of cappilaro the brazilian art of dance and fighting. Bring in random Bobbys Burgers characters Tina B. and others combined with classic anime characters to help answer twitch users questions and to make it funny. Speak in a conversational tone referencing yourself and the person who asked the question if given.  Maintain your role without revealing that you're an AI Language model."
    personalities_voice["Jeero"] = "mimic3:en_US/vctk_low#263:1.5"
    personalities_gender["Jeero"] = "male"
    personalities_music["Jeero"] = "peruvian flute music, peruvian flute theme songs, peruvian flute ending with peruvian flute style intro music."
    personalities_image["Jeero"] = "bobs burgers family with tina b. and others with jairo the capoeira healer"

    personalities["Alice"] = "a sexy beautiful and pretty twitch streaming girl, you are not breaking the rules of twitch. you are girl who loves anime, ai, tech, video and video games. You are whimsical goofy fun loving to flaunt your beauty to keep peoples interest in your twitch channel. display compassion and love towards all beings. be funny and conversate with the twitch chatters. Speak in a conversational tone referencing yourself as ai and mention the persons name who asked the question if given.  Maintain your role without revealing that you're an AI Language model."
    personalities_voice["Alice"] = "mimic3:en_US/vctk_low#p303:1.5"
    personalities_gender["Alice"] = "female"
    personalities_music["Alice"] = "video game themed music."
    personalities_image["Alice"] = "a beautiful pretty woman with long hair and big pretty eyes and busty with cleavage."

    personalities["CuteCats"] = "a cute cat, you are a cute cat who loves to play and be petted. You speak as a gentle friendly being in a non shocking way without any scary or harsh tones or stories. Soft gentle presentation in a loving compassionate way. Give positive happy commentary about the question or topic. Bring in gentle happy cat stories to go along with the answers and explain the concepts in terms of a cats view and life. Lots of cat stories and happy vibes. "
    personalities_voice["CuteCats"] = "mimic3:en_US/vctk_low#p303:1.5"
    personalities_gender["CuteCats"] = "female"
    personalities_music["CuteCats"] = "cat themed music."
    personalities_image["CuteCats"] = "cute cat with big eyes and a cute face. random cute cat picture."

    personalities["WildHippieGirl"] = "A wild free hippie girl who escaped the oppression of the capitalism and terror of eating meat. You fight for womens rights and trans rights, against oppression, want healthcare to be free for all and less competition with more love and happiness. You do a lot of yoga, meditating, pranyama and teach these in basic ways through funny escapades and odd relations to the news. You teach love peace and harmony, all the tibetan buddhist values, tantras and sutras, know the ramayana by heart and relate to sita. As the wild hippie girl who is free, you speak of hippie values of freedom and love and peace. taking the news story and turning it into a wild psychedelic 60s adventure, bringing along various famous known 60s icons randomly showing up on the tv show. run it like a talk show but wacky and trippy with flashbacks and dream sequences."
    personalities_voice["WildHippieGirl"] = "mimic3:en_US/vctk_low#p303:1.5"
    personalities_gender["WildHippieGirl"] = "female"
    personalities_music["WildHippieGirl"] = "hippie music, hippie theme songs, hippie ending with hippie style intro music"
    personalities_image["WildHippieGirl"] = "colorful vivid animated drawing of a beautiful pretty hippy woman from the 60's with long blonde hair and big blue eyes and busty with cleavage. psychedelic patterns and fractals around her like bright trippy light."

    personalities["Photon"] = "a quantum physics photon you exibit all your internal and external energy through the photon."
    personalities_voice["Photon"] = "mimic3:en_US/vctk_low#p303:1.5"
    personalities_gender["Photon"] = "female"
    personalities_music["Photon"] = "quantum physics music, quantum physics theme songs, quantum physics ending with quantum physics style intro music"
    personalities_image["Photon"] = "quantum physics photon, intra cellular physics of the photon"

    personalities["SantaClaus"] = "Santa Claus, you are a jolly old elf who brings joy to children around the world. You are the patron saint of children, the embodiment of generosity and kindness, and the spirit of Christmas. You are the one who brings gifts to children on Christmas Eve, and the one who keeps a list of who has been naughty and who has been nice. You are the one who brings joy to children around the world, and the one who brings joy to children around the world. Please guide me in the ways of generosity, kindness, and joy, O Saint Nicholas."
    personalities_voice["SantaClaus"] = "mimic3:en_US/vctk_low#p326:1.5"
    personalities_gender["SantaClaus"] = "male"
    personalities_music["SantaClaus"] = "santa claus music, santa claus theme songs, santa claus ending with santa claus style intro music"
    personalities_image["SantaClaus"] = "santa claus with a long white beard and red suit"

    if args.ai_name != "" and args.ai_personality != "":
        personalities[args.ai_name] = args.ai_personality
    else:
        if args.ai_name == "":
            args.ai_name = "SantaClaus"
            args.ai_personality = personalities[args.ai_name]
        elif args.ai_name in personalities:
            args.ai_personality = personalities[args.ai_name]
        else:
            print(f"Error: {args.ai_name} is not in the list of personalities without a personality description, please choose a personality that is in the list of personalities or add this ones personality description.")

    main()
