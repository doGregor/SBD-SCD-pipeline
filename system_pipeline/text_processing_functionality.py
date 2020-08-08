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
            tokens.append(token)

    return tokens


def get_predictions_as_sentences(predictions):
    aligned_full_sentences = []
    for aligned_prediction in predictions:
        tokens = aligned_prediction[0]
        sc_tag = aligned_prediction[1]

        single_sentence = []
        for idx, token in enumerate(tokens):
            if (token == "s" or token == "t" or token == "re" or token == "m" or
                    token == "d" or token == "ll" or token == "ve"):
                del single_sentence[-1]
                new_token = tokens[idx - 1] + token
                single_sentence.append(new_token)
            else:
                single_sentence.append(token)

        str_single_sentence = ' '.join(single_sentence[:-1])
        str_single_sentence += single_sentence[-1]
        aligned_full_sentences.append((str_single_sentence, sc_tag))

    return aligned_full_sentences
