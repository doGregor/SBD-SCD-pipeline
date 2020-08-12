import os
import nltk
import itertools

window_size = 7
directories = ["train", "test", "val"]

for directory in directories:
    directory = "../data/" + directory
    output_name = "out_" + directory + ".txt"

    speaker_change_label = "SC "
    output_file = open(output_name, "w")
    for file in os.listdir(directory):
        with open(directory+"/"+file) as curr_file:
            curr_label = None
            for idx, line in enumerate(curr_file):
                line.rstrip()
                label = line[-6:]
                line = line[:-6]
                if idx == 0:
                    to_append = speaker_change_label + line
                    curr_label = label
                elif curr_label != label:
                    curr_label = label
                    to_append = speaker_change_label + line
                else:
                    to_append = line

                output_file.write(to_append)
                output_file.write("\n")
            curr_file.close()
    output_file.close()

    data = open(output_name, "r")
    tokens = []
    pos_tags = []

    for line in data:
        tokens_element = []
        pos_tags_element = []
        if line[:3] == "SC ":
            line_words = nltk.word_tokenize(line[3:].rstrip())
            for idx, token in enumerate(line_words):
                if (token == "'s" or token == "'t"
                        or token == "'re" or token == "'m"
                        or token == "'d" or token == "'ll"
                        or token == "'ve") and idx != 0:
                    del tokens_element[-1]
                    new_token = line_words[idx - 1] + token
                    tokens_element.append(new_token)
                else:
                    tokens_element.append(token)
                    if idx == 0:
                        pos_tags_element.append("SC")
                    else:
                        pos_tags_element.append("O")
        else:
            line_words = nltk.word_tokenize(line.rstrip())
            for idx, token in enumerate(line_words):
                if (token == "'s" or token == "'t"
                        or token == "'re" or token == "'m"
                        or token == "'d" or token == "'ll"
                        or token == "'ve") and idx != 0:
                    del tokens_element[-1]
                    new_token = line_words[idx - 1] + token
                    tokens_element.append(new_token)
                else:
                    tokens_element.append(token)
                    if idx == 0:
                        pos_tags_element.append("NSC")
                    else:
                        pos_tags_element.append("O")

        tokens.append(tokens_element)
        pos_tags.append(pos_tags_element)

    write_to = "pos_" + directory + ".txt"
    pos_sc_train = open(write_to, "w")

    for idx in range(0, len(tokens), window_size):
        if (idx + window_size - 1) < len(tokens):
            curr_tokens = list(itertools.chain.from_iterable(tokens[idx:idx + window_size]))
            curr_pos_tags = list(itertools.chain.from_iterable(pos_tags[idx:idx + window_size]))
            for i, t in enumerate(curr_tokens):
                pos_sc_train.write(t + " " + curr_pos_tags[i])
                pos_sc_train.write("\n")
            if len(curr_pos_tags) > 350:
                print(len(curr_pos_tags))
        else:
            curr_tokens = list(itertools.chain.from_iterable(tokens[idx:]))
            curr_pos_tags = list(itertools.chain.from_iterable(pos_tags[idx:]))
            for i, t in enumerate(curr_tokens):
                pos_sc_train.write(t + " " + curr_pos_tags[i])
                pos_sc_train.write("\n")
            if len(curr_pos_tags) > 350:
                print(len(curr_pos_tags))

        pos_sc_train.write("\n")
