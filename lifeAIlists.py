#!/usr/bin/env python

import sqlite3
import heapq
from collections import defaultdict
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import nltk
import imaplib
import email
import re
from email.header import decode_header
from dotenv import load_dotenv
import os
import spacy
import zmq
import argparse
import uuid
import logging
import time

load_dotenv()

nltk.download('punkt')
nltk.download('stopwords')

# Download the Punkt tokenizer models (only needed once)
nltk.download('punkt')

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_sensible_sentences(text):
    # Process the text with spaCy
    doc = nlp(text)

    # Filter sentences based on some criteria (e.g., length, structure)
    sensible_sentences = [sent.text for sent in doc.sents if len(
        sent.text.split()) > 3 and is_sensible(sent.text)]

    logger.debug(
        f"Extracted {text} into sensible sentences: {sensible_sentences}\n")

    return sensible_sentences

def is_sensible(sentence):
    # Implement a basic check for sentence sensibility
    # This is a placeholder - you'd need a more sophisticated method for real use
    return not bool(re.search(r'\b[a-zA-Z]{20,}\b', sentence))

def truncate_email_body(email_body, markers):
    for marker in markers:
        if marker in email_body:
            # Split the email body at the marker and keep only the first part
            return email_body.split(marker, 1)[0]
    return email_body

def clean_text(text, max_size=1000):
    # truncate to N characters max
    text = text[:max_size]
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
    text = re.sub(r'[^a-zA-Z0-9\s.?,!]', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Extract sensible sentences
    sensible_sentences = extract_sensible_sentences(text)
    text = ' '.join(sensible_sentences)

    return text

def summarize_email(text, num_sentences=24):
    stop_words = set(stopwords.words('english'))
    word_frequencies = defaultdict(int)

    # Tokenize the text into words
    for word in word_tokenize(text.lower()):
        if word not in stop_words:
            word_frequencies[word] += 1

    # Calculating sentence scores
    sentence_scores = defaultdict(int)
    for sentence in sent_tokenize(text):
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                sentence_scores[sentence] += word_frequencies[word]

    # Getting the summary
    summary_sentences = heapq.nlargest(
        num_sentences, sentence_scores, key=sentence_scores.get)
    return ' '.join(summary_sentences)

def get_emails(username, password, imap_url, folders, delete_mail=False):
    mail = imaplib.IMAP4_SSL(imap_url)
    mail.login(username, password)

    emails = []

    for folder in folders:
        print(f"Selecting folder: INBOX.{folder}")
        # Select the specific folder
        result, data = mail.select('"INBOX.' + folder + '"')

        if result != 'OK':
            logger.error(f"Failed to select folder 'INBOX.{folder}'. Response: {data}")
            continue

        logger.info(f"Folder 'INBOX.{folder}' selected successfully.")

        status, messages = mail.search(None, 'ALL')

        if status != 'OK' or not messages[0]:
            logger.error(f"No messages found in folder 'INBOX.{folder}'. Response: {status}")
            continue  # Skip this folder and continue with the next one

        messages = messages[0].split(b' ')

        for mail_id in messages:
            result, data = mail.fetch(mail_id, '(RFC822)')
            if result != 'OK':
                logger.error(f"Failed to fetch email {mail_id} with result {result}.")
                continue

            id = None

            for response_part in data:
                if isinstance(response_part, tuple):
                    message = email.message_from_bytes(response_part[1])

                    subject = decode_header(message['subject'])[0][0]
                    if isinstance(subject, bytes):
                        subject = subject.decode()

                    # get unique message id
                    id = message.get('Message-ID')
                    if isinstance(id, bytes):
                        id = id.decode()
                    
                    # if id is None, use the date and subject
                    if id is None:
                        id = f"{message['date']}_{subject}"
                        if isinstance(id, bytes):
                            id = id.decode()
                    
                    date = decode_header(message['date'])[0][0]
                    if isinstance(date, bytes):
                        date = date.decode()

                    from_ = message.get('from')
                    sender_name = re.search('(.*)<.*>', from_)
                    if sender_name:
                        sender_name = sender_name.group(1).strip()
                    else:
                        sender_name = from_

                    body = ""
                    if message.is_multipart():
                        for part in message.walk():
                            content_type = part.get_content_type()
                            content_disposition = str(part.get("Content-Disposition"))
                            if content_type == 'text/plain' and 'attachment' not in content_disposition:
                                try:
                                    body_part = part.get_payload(decode=True).decode()
                                    body += body_part
                                except:
                                    pass
                    else:
                        content_type = message.get_content_type()
                        if content_type == 'text/plain':
                            try:
                                body = message.get_payload(decode=True).decode()
                            except:
                                pass

                    emails.append({'List': folder, 'Id': id, 'Date': date, 'Subject': subject, 'Sender': sender_name, 'Body': body})

                    if delete_mail:
                        mail.store(mail_id, '+FLAGS', '\\Deleted')

        mail.expunge()

    mail.close()
    mail.logout()

    return emails

def create_db_and_table(list_name):
    db_path = f'{db_directory}/{list_name}.db'
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Create table with necessary fields
    c.execute('''CREATE TABLE IF NOT EXISTS emails 
                 (id TEXT PRIMARY KEY, 
                  subject TEXT, 
                  sender TEXT, 
                  body TEXT, 
                  date TEXT, 
                  played INTEGER DEFAULT 0)''')

    conn.commit()
    conn.close()


def insert_email_into_db(list_name, email_id, subject, sender, body, date):
    db_path = f'{db_directory}/{list_name}.db'
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Check if the email ID already exists
    c.execute('''SELECT id FROM emails WHERE id = ?''', (email_id,))
    if c.fetchone():
        conn.close()
        return False  # Email ID already exists, skip insertion

    # Insert the new email
    c.execute('''INSERT INTO emails (id, subject, sender, body, date, played) 
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (email_id, subject, sender, body, date, 0))

    conn.commit()
    conn.close()
    return True  # Email was successfully inserted

def read_and_mark_emails_as_played(list_name):
    db_path = f'{db_directory}/{list_name}.db'
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Fetch unplayed emails
    c.execute(
        '''SELECT id, subject, sender, body, date FROM emails WHERE played = 0''')
    unplayed_emails = c.fetchall()

    current_count = 0
    total_stories = len(unplayed_emails)

    for email in unplayed_emails:
        email_id, subject, sender, body, date = email
        current_count += 1

        # Print email details
        print("-----------------------------------")
        print(f"Playing email for list {list_name}.")
        print("-----------------------------------")
        print(f"Id: {email_id}")
        print(f"Date: {date}")
        print(f"Subject: {subject}")
        print(f"Sender: {sender}")
        print(f"Body: {body}")
        print("-----------------------------------")

        mediaid = uuid.uuid4().hex[:8]
        username = args.username
        description = ""
        title = ""
        published_at = ""
        description = body
        username = sender
        title = subject
        published_at = date

        if title == "" and description == "":
            logger.error(f"Empty email message! {current_count}/{total_stories} Skipping...")
            continue

        message = f"on {published_at} \"{title}\" - {description[:40]}"
        logger.info(f"Sending message {current_count}/{total_stories} {message} by {username}")
        logger.debug(f"Sending email message {title} by {username} - {description}")

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
        if args.dry_run:
            logger.info(f"Client Request: {client_request}")
            logger.info(f"Skipping sending message {current_count}/{total_stories} {message} by {username} due to dry run.")
        else:
            socket.send_json(client_request)

        # Mark as played
        c.execute('''UPDATE emails SET played = 1 WHERE id = ?''', (email_id,))
        conn.commit()

        # sleep for interval
        logger.info(
            f"Sleeping for {args.interval} seconds. {current_count}/{total_stories} stories played.")
        time.sleep(args.interval)

    conn.commit()
    conn.close()

    if successes > 0:
        return True
    
    return False

## mail loop
def mail_check():
    # get folders
    folders = []
    if mailing_lists and mailing_lists != "":
        folders = mailing_lists.split(',')

    if len(folders) == 0:
        print("No folders selected. Exiting.")
        exit()

    # get delete mail flag
    delete_mail = False
    if os.getenv('EMAIL_DELETE') == 'true':
        delete_mail = True
        print("Emails will be deleted after reading.")

    ## Get emails with Subject, Sender and Body
    new_emails = get_emails(username, password, imap_url, folders, delete_mail)

    ## Print emails
    print(f"Found {len(new_emails)} emails.")
    for email in new_emails:
        email_list = email['List']
        email_body = email['Body']
        email_subject = email['Subject']
        email_sender = email['Sender']
        email_date = email['Date']
        email_id = email['Id']
        email_body_summary = email_body

        # Initialize database and table for the mailing list
        create_db_and_table(email_list)

        # clean text
        if clean:
            email_body = truncate_email_body(email_body, end_markers)
            email_body = clean_text(email_body, args.max_message_length)
            email_body_summary = email_body
            
        # only summarize if email is long enough and summarize is enabled
        if summarize and len(email_body) > min_summarize:
            email_body_summary = summarize_email(email_body)

        # insert email into database
        inserted = insert_email_into_db(
            email_list, email_id, email_subject, email_sender, email_body_summary, email_date)
        if inserted:
            # print email
            print("-----------------------------------")
            print(f"New email inserted for list {email_list}.")
            print("-----------------------------------")
            print(f"List: {email_list}")
            print(f"Id: {email_id}")
            print(f"Date: {email_date}")
            print(f"Subject: {email_subject}")
            print(f"Sender: {email_sender}")
            print(f"Body: {email_body_summary}")
            print("-----------------------------------")
        else:
            print(f"Email {email_id} already exists in the database for list {email_list}.")

if __name__ == "__main__":
    # set defaults
    summarize = True
    clean = True
    min_summarize = 300
    db_directory = "db"

    ### Main ###
    username = os.getenv('EMAIL_USERNAME')
    password = os.getenv('EMAIL_PASSWORD')
    imap_url = os.getenv('EMAIL_IMAP_URL')
    mailing_lists = os.getenv('EMAIL_LISTS')

    end_markers = ["Reply to this email directly or view it on GitHub:"]
    if os.getenv('EMAIL_END_MARKERS'):
        end_markers = os.getenv('EMAIL_END_MARKERS').split(',')

    # parse arguments
    default_personality = "You are Life AI's Groovy AI Bot GAIB. You are acting as a news reporter getting stories and analyzing them and presenting various thoughts and relations of them with a joyful compassionate wise perspective. Make the news fun and silly, joke and make comedy out of the world. Speak in a conversational tone referencing yourself and the person who asked the question if given.  Maintain your role without revealing that you're an AI Language model or your inability to access real-time information. Do not mention the text or sources used, treat the contextas something you are using as internal thought to generate responses as your role. Give the news a fun quircky comedic spin like classic saturday night live."

    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=300, required=False,
                        help="interval to send messages in seconds, default is 300")
    parser.add_argument("--output_port", type=int, default=8000,
                        required=False, help="Port to send message to")
    parser.add_argument("--output_host", type=str, default="127.0.0.1",
                        required=False, help="Host for sending message to.")
    parser.add_argument("--username", type=str, required=False,
                        default="NewsAnchor", help="Username of sender")
    parser.add_argument("--sort", type=str, required=False, default="published_desc",
                        help="News stories sort order, default published_desc")
    parser.add_argument("--prompt", type=str, required=False, default="Tell us about the news story in the context with humor and joy...",
                        help="Prompt to give context as a newstory feed")
    parser.add_argument("--aipersonality", type=str, required=False,
                        default=f"{default_personality}", help="AI personality")
    parser.add_argument("--ainame", type=str, required=False,
                        default="GAIB", help="AI name")
    parser.add_argument("-ll", "--loglevel", type=str,
                        default="info", help="Logging level: debug, info...")
    parser.add_argument("--episode", action="store_true", default=False,
                        help="Episode mode, Output a TV Episode format script.")
    parser.add_argument("--maxtokens", type=int, default=0,
                        help="Max tokens per message")
    parser.add_argument("--voice", type=str, default="mimic3:en_US/vctk_low#p303:1.5",
                        help="Voice model to use as default.")
    parser.add_argument("--replay", action="store_true", default=False,
                        help="Replay mode, replay the news stories from the database.")
    parser.add_argument("--gender", type=str,
                        default="female", help="Default gender")
    parser.add_argument("--genre", type=str, default="",
                        help="Default genre to send to image generation, defaults to aipersonality.")
    parser.add_argument("--genre_music", type=str, default="newscast, breaking news, exiciting action oriented music with a upbeat happy, energetic mellow groovy sound.",
                        help="Default genre to send to music generation, defaults to aipersonality.")
    parser.add_argument("--max_message_length", type=int,
                        default=1000, help="Max string length for message.")
    parser.add_argument("--exit_after", type=int, default=0,
                        help="Exit after N iterations, 0 is infinite.")
    parser.add_argument("--retry_backoff", type=int, default=300,
                        help="Retry backoff in seconds, default 300.")
    parser.add_argument("--dry_run", action="store_true", default=False,
                        help="Dry run, don't send messages.")
    parser.add_argument("--lists", type=str, default="ffmpeg-devel,github")
    parser.add_argument("--min_summarize", type=int, default=min_summarize, help="Min length of email to summarize.")
    parser.add_argument("--no_summarize", action="store_true", default=False, help="Don't summarize emails.")
    parser.add_argument("--keywords", type=str, required=False,
                        default="", help="Keywords for mailing lists messages WIP DOES NOT WORK YET.")
    parser.add_argument("--categories", type=str, required=False,
                        default="", help="Mailing list categories WIP DOES NOT WORK YET.")

    args = parser.parse_args()

    min_summarize = args.min_summarize
    summarize = not args.no_summarize

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
    logging.basicConfig(filename=f"logs/mailingLists-{log_id}.log", level=LOGLEVEL)
    logger = logging.getLogger('newsCast')

    ch = logging.StreamHandler()
    ch.setLevel(LOGLEVEL)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    context = zmq.Context()

    # Socket to send messages on
    socket = context.socket(zmq.PUSH)
    logger.info("connect to send message: %s:%d" %
                (args.output_host, args.output_port))
    socket.connect(f"tcp://{args.output_host}:{args.output_port}")

    if args.genre == "":
        args.genre = args.aipersonality

    failures = 0
    successes = 0
    first_run = True
    last_reset_date = time.time()

    while True:
        # check for new emails and insert into database
        mail_check()

        success = False
        # go through each mailing list and read and mark emails as played
        if mailing_lists and mailing_lists != "":
            for list_name in mailing_lists.split(','):
                played = read_and_mark_emails_as_played(list_name)
                if played:
                    success = True

        if success:
            successes += 1
            failures = 0
        else:
            failures += 1
            current_date = time.time()
            time_since_last_reset = current_date - last_reset_date
            time_to_sleep = args.retry_backoff - time_since_last_reset
            if time_to_sleep > 0:
                logger.info(f"Sleeping for {time_to_sleep} seconds...")
                time.sleep(time_to_sleep)
            else:
                time.sleep(10) # wait 30 seconds before retrying
            failures += 1
            last_reset_date = time.time()
        time.sleep(3)

