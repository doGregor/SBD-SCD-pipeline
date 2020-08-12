"""
This module implements functionality to preprocess the input-files contents and
to align the output with the predictions and save it.
"""

import nltk
import string
import re


def get_tokenized_data(file_location):
    """
    Function reads txt file from path and returns content as tokens.
    :param file_location: path to a txt file
    :return: list of nltk tokenized word-tokens
    """
    tokens = []

    f = open(file_location, "r")
    file_data = f.read()

    file_data = file_data.lower()
    file_data = re.sub("[\(\[].*?[\)\]]", "", file_data)
    file_data = file_data.splitlines()
    file_data = ' '.join(file_data).rstrip()

    tokenized_data = nltk.word_tokenize(file_data)
    for idx, token in enumerate(tokenized_data):

        token = re.sub("['…`´’-]", "", token)

        if token not in string.punctuation and len(token) > 0:
            tokens.append(token)

    return tokens


def get_predictions_as_sentences(predictions):
    """
    Function produces output that is ready to write to output file.
    :param predictions: list of tuples where tuple[0] is list of tokens of one sentence
                        and tuple[1] is the speaker change label
    :return: list of sentences with speaker change labels
    """
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


def save_predictions(predictions, filename):
    """
    Saves predictions to output_dir.
    :param predictions: list of (sentence, SC_label) tuples
    :param filename: file name for output directory
    :return: returns nothing
    """
    output_path = "output_dir" + "/" + filename
    with open(output_path, "w+") as output_file:
        for item in predictions:
            output_file.write(str(item))
            output_file.write("\n")
