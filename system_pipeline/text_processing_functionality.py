import nltk
import string
import re


def get_tokenized_data(file_location):
    tokens = []

    f = open(file_location, "r")
    file_data = f.read()
    file_data = file_data.replace("-", " ")

    file_data = file_data.lower()
    file_data = re.sub("[\(\[].*?[\)\]]", "", file_data)
    file_data = file_data.splitlines()
    file_data = ' '.join(file_data).rstrip()

    tokenized_data = nltk.word_tokenize(file_data)
    for idx, token in enumerate(tokenized_data):

        token = re.sub("['…`´’]", "", token)

        if token not in string.punctuation and len(token) > 0:
            tokens.append(token.replace("'", ""))

    return tokens
