import numpy as np
import nltk
import itertools
import os

for file in os.listdir("./data_all"):
    if file.endswith(".txt"):
        print("Current file:", file)
        file_path = "./data_all/" + str(file)

        current_file = open(file_path, "r")

        tokens = []
        pos_tags = []
        write_to = "./output_all/" + "pos_" + str(file)
        window_size = 7
        for line in current_file:
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
                        pos_tags_element.append("O")
            # consistency check
            if len(tokens_element) > 0:
                tokens.append(tokens_element)
                pos_tags.append(pos_tags_element)
            else:
                print("Unusable sequence")
                print(tokens_element, pos_tags_element)
        current_file.close()

        print("input read")

        pos_sc_file = open(write_to, "w")
        for idx in range(0, len(tokens), window_size):
            if (idx + window_size) < len(tokens) - 1:
                curr_tokens = list(itertools.chain.from_iterable(tokens[idx:idx + window_size]))
                curr_pos_tags = list(itertools.chain.from_iterable(pos_tags[idx:idx + window_size]))
                tags = np.asarray(curr_pos_tags)
                for i, t in enumerate(curr_tokens):
                    pos_sc_file.write(t + " " + curr_pos_tags[i])
                    pos_sc_file.write("\n")
                if len(curr_pos_tags) > 512:
                    print(len(curr_pos_tags))
                pos_sc_file.write("\n")
            else:
                curr_tokens = list(itertools.chain.from_iterable(tokens[idx:]))
                curr_pos_tags = list(itertools.chain.from_iterable(pos_tags[idx:]))
                tags = np.asarray(curr_pos_tags)
                for i, t in enumerate(curr_tokens):
                    pos_sc_file.write(t + " " + curr_pos_tags[i])
                    pos_sc_file.write("\n")
                if len(curr_pos_tags) > 512:
                    print(len(curr_pos_tags))
                pos_sc_file.write("\n")

        pos_sc_file.close()
