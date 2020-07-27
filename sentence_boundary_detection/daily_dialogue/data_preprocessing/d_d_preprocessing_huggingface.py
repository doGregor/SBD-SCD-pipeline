import pandas as pd
import nltk
import string
import re
import numpy as np
import os

config = {
    "path_to_data": "./data",
    "sequence_length": 64,
    "path_to_save": "./data_out",
    "file_name_to_save": "test"
}

NUMBERS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


def convert_to_df_with_pos_tags(token_list):
    df = pd.DataFrame({'token': token_list})
    print("token column initialized")
    df["pos"] = "O"
    print("O tag added")
    mask_1 = (df['token'] == ".")
    print("mask created")
    df["pos"][mask_1] = ""
    df["token"][mask_1] = ""
    mask_1.drop(mask_1.tail(1).index, inplace=True)
    df["pos"][np.flatnonzero(mask_1)+1] = "BOS"
    print("mask applied")
    df["pos"][0] = "BOS"
    return df


def save_output(data, path=config["path_to_save"], filename=config["file_name_to_save"]):
    path = os.path.join(path, filename+".csv")
    data.to_csv(path_or_buf=path, index=False, header=True)


def save_txt(data, path=config["path_to_save"], filename=config["file_name_to_save"]):
    path = os.path.join(path, filename + ".txt")
    data.to_csv(path_or_buf=path, sep=" ", index=False, header=False)
    f = open(path, "r+")
    file_data = f.read()
    file_data = file_data.splitlines()
    f.truncate(0)
    for line in file_data:
        line = line.rstrip()
        if not line:
            f.write("\n")
        else:
            f.write(line+"\n")
    f.close()


def get_cleaned_data(f_path):
    print("reading df")
    df = pd.read_csv(f_path)
    print("dropping NA")
    df = df.dropna()
    print("creating sentences")
    sentences = []
    for g, df_p in df.groupby(np.arange(df.shape[0]) // config["sequence_length"]):
        if len(df_p["pos"].tolist()) is config["sequence_length"]:
            df_p = df_p.append(pd.Series(), ignore_index=True)
            sentences.append(df_p)
    print("sentences created")
    print("number of samples: ", len(sentences))
    return sentences


def return_tokeized_lecture_data():

    for file in os.listdir(config["path_to_data"]):
        tokens = []
        print("current file: ", file)
        curr_path = config["path_to_data"] + "/" + file
        f = open(curr_path, "r")
        file_data = f.read()
        file_data = file_data.replace("-", " ")

        file_data = file_data.lower()
        file_data = re.sub("[\(\[].*?[\)\]]", "", file_data)
        file_data = file_data.replace("__eou__", "")
        file_data = file_data.rstrip()
        file_data = file_data.splitlines()
        file_data = ' '.join(file_data).replace("  ", " ")

        tokenized_data = nltk.word_tokenize(file_data)
        for idx, token in enumerate(tokenized_data):

            token = token.replace("`", "")
            token = token.replace("´", "")
            token = token.replace("’", "")
            token = token.replace("'", "")

            if len(token) is 2 and token[-1] is ".":
                tokens.append(token[0])
                tokens.append(".")
                continue
            '''
            if len(token) > 1 and token[0] in NUMBERS and token[-1] is ".":
                tokens.append(token[:-1])
                tokens.append(".")
                continue
            '''

            if token == "." or token == "!" or token == "?":
                tokens.append(".")
            else:
                if token not in string.punctuation and len(token) > 0:
                    if tokenized_data[idx] == "'s" or tokenized_data[idx] == "'t"\
                            or tokenized_data[idx] == "'re" or tokenized_data[idx] == "'m"\
                            or tokenized_data[idx] == "'d" or tokenized_data[idx] == "'ll"\
                            or tokenized_data[idx] == "'ve":
                        del tokens[-1]
                        new_token = tokenized_data[idx - 1] + tokenized_data[idx]
                        tokens.append(new_token)
                    elif (token == "s" or token == "t" or token == "re"
                            or token == "m" or token == "d" or token == "ll"
                            or token == "ve")\
                            and (tokenized_data[idx - 1] == "'" or tokenized_data[idx - 1] == "’")\
                            and (len(tokenized_data[idx - 2]) > 1 or tokenized_data[idx - 2] == "i"):
                        del tokens[-1]
                        del tokens[-1]
                        new_token = tokenized_data[idx - 2] + "'" + token
                        tokens.append(new_token)
                    else:
                        tokens.append(token)

        df = convert_to_df_with_pos_tags(tokens)
        save_output(df, filename=file[:-4])
        f_path = config["path_to_save"] + "/" + file[:-4] + ".csv"
        sentences = get_cleaned_data(f_path)
        sentences = pd.concat(sentences)
        save_txt(sentences, filename=file[:-4])


if __name__ == '__main__':
    return_tokeized_lecture_data()