#!/usr/bin/env python

## Life AI Document Injection module
#
# Chris Kennedy 2023 (C) GPL
#
# Free to use for any use as in truly free software
# as Richard Stallman intended it to be.
#

import zmq
import argparse
import warnings
import logging
import time
import json
import logging
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
import chromadb
from constants import CHROMA_SETTINGS
from langchain.vectorstores import Chroma
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import GPT4All
from langchain.chains import RetrievalQA
import re

load_dotenv()

warnings.simplefilter(action='ignore', category=Warning)

def clean_text(text):
    # truncate to 800 characters max
    text = text[:args.max_size]
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
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path="db")
    db = Chroma(persist_directory="db", embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
    retriever = db.as_retriever(search_kwargs={"k": 1})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [StreamingStdOutCallbackHandler()]
    llm = GPT4All(model="models/ggml-all-MiniLM-L6-v2-f16.bin", max_tokens=args.max_tokens, backend='gptj', n_batch=8, callbacks=callbacks, verbose=False)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    while True:
        header_message = receiver.recv_json()

        # context
        history = ""
        if 'history' in header_message:
            history = json.dumps(header_message['history'])
        else:
            history = ""
        message = ""
        if 'message' in header_message:
            message = header_message['message']
        else:
            message = ""

        message = clean_text(message)

        logger.debug(f"received message: {message} in context: {history} {json.dumps(header_message)}\n")

        # look up in chroma db
        logger.info(f"looking up {message} in chroma db...\n")
        res = qa(message)
        if res is None:
            logger.error(f"Error getting answer from Chroma DB: {res}")
            return None
        if 'result' not in res:
            logger.error(f"Error getting answer from Chroma DB: {res}")
            return None
        if 'source_documents' not in res:
            logger.error(f"Error getting answer from Chroma DB: {res}")
            return None
        logger.debug(f"got answer: {res['result']} in context: {history}\n")
        answer, docs = res['result'], res['source_documents']
        for document in docs:
            logger.debug(f"got document: {document.metadata}\n")
            source_doc = document.metadata["source"]
            context_add = f" {document.page_content}"
            history += context_add

        logger.info(f"got answer: {answer} in context: {history}\n")
        header_message['history'] = clean_text(history)

        # send the processed message
        logger.info(f"sending message: {message} in context: {history} {json.dumps(header_message)}\n")

        sender.send_json(header_message)
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, default=8000, required=False, help="Port for receiving text input")
    parser.add_argument("--output_port", type=int, default=1500, required=False, help="Port for sending image output")
    parser.add_argument("--input_host", type=str, default="127.0.0.1", required=False, help="Port for receiving text input")
    parser.add_argument("--output_host", type=str, default="127.0.0.1", required=False, help="Port for sending image output")
    parser.add_argument("-ll", "--loglevel", type=str, default="info", help="Logging level: debug, info...")
    parser.add_argument("--max_size", type=int, default=800, required=False, help="Maximum size of text to process")
    parser.add_argument("--max_tokens", type=int, default=4096, required=False, help="Maximum tokens to process")
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
    logging.basicConfig(filename=f"logs/docInject-{log_id}.log", level=LOGLEVEL)
    logger = logging.getLogger('docInject')

    ch = logging.StreamHandler()
    ch.setLevel(LOGLEVEL)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    logger.info("connected to ZMQ in: %s:%d\n" % (args.input_host, args.input_port))
    receiver.bind(f"tcp://{args.input_host}:{args.input_port}")
    #receiver.subscribe("")

    sender = context.socket(zmq.PUSH)
    logger.info("binded to ZMQ out: %s:%d\n" % (args.output_host, args.output_port))
    sender.bind(f"tcp://{args.output_host}:{args.output_port}")

    main()

