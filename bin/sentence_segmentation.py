#!/usr/bin/env python

import argparse
import spacy
import re

## python -m spacy download en_core_web_sm

def extract_sensible_sentences(text):
    # Load the spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Process the text with spaCy
    doc = nlp(text)

    # Filter sentences based on some criteria (e.g., length, structure)
    sensible_sentences = [sent.text for sent in doc.sents if len(sent.text.split()) > 3 and is_sensible(sent.text)]

    return sensible_sentences

def is_sensible(sentence):
    # Implement a basic check for sentence sensibility
    # This is a placeholder - you'd need a more sophisticated method for real use
    return not bool(re.search(r'\b[a-zA-Z]{20,}\b', sentence))

# Initialize the argument parser
parser = argparse.ArgumentParser(description="Extract sensible sentences from a text.")
parser.add_argument("text", type=str, help="The text to be processed")

# Parse the arguments
args = parser.parse_args()

# Extract sensible sentences from the input text
sensible_sentences = extract_sensible_sentences(args.text)
print("\n".join(sensible_sentences))

