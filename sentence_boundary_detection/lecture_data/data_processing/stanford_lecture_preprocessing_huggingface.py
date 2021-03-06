import pandas as pd
import nltk
import string
import re
import numpy as np
import os
from random import shuffle

config = {
    "path_to_data": "./data",
    "path_to_save": "./output_data",
    "file_name_to_save": "all_stanford_data_txt",
    "sequence_length": 64,
    "train_size": 0.8
}
NUMBERS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


def return_tokenized_lecture_data():
    tokens = []

    for file in os.listdir(config["path_to_data"]):
        print("current file: ", file)
        curr_path = config["path_to_data"] + "/" + file
        f = open(curr_path, "r")
        file_data = f.read()
        file_data = file_data.replace("-", " ")

        file_data = file_data.lower()
        file_data = re.sub("[\(\[].*?[\)\]]", "", file_data)
        file_data = file_data.splitlines()
        file_data = ' '.join(file_data).replace("  ", " ")

        sent_tokenizer = nltk.tokenize.PunktSentenceTokenizer()
        for sentence in sent_tokenizer.tokenize(file_data):
            if sentence not in string.punctuation:
                tokenized_sentence = nltk.word_tokenize(sentence)
                len_sent = 0
                for token in tokenized_sentence:
                    if token not in string.punctuation:
                        len_sent = len_sent + 1
                if len_sent > 6 and len_sent <= 70:
                    for idx, token in enumerate(tokenized_sentence):

                        token = re.sub("[`´]", "", token)

                        if len(token) is 2 and token[-1] is ".":
                            tokens.append(token[0])
                            tokens.append(".")
                            continue
                        if len(token) > 1 and token[0] in NUMBERS and token[-1] is ".":
                            tokens.append(token[:-1])
                            tokens.append(".")
                            continue

                        if token == "." or token == "!" or token == "?":
                            tokens.append(".")
                        else:
                            if token not in string.punctuation:
                                if (token == "'s" or token == "'t" or token == "'re" or
                                        token == "'m" or token == "'d" or token == "'ll"):
                                    del tokens[-1]
                                    new_token = tokenized_sentence[idx - 1] + token
                                    tokens.append(new_token)
                                else:
                                    if len(token.replace("'", "")) > 0:
                                        tokens.append(token.replace("'", ""))
    return tokens


def convert_to_df_with_pos_tags(token_list):
    df = pd.DataFrame({'token': token_list})
    df["pos"] = "O"
    mask_1 = (df['token'] == ".")
    df["pos"][mask_1] = ""
    df["token"][mask_1] = ""
    mask_1.drop(mask_1.tail(1).index, inplace=True)
    df["pos"][np.flatnonzero(mask_1)+1] = "BOS"
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
    df = pd.read_csv(f_path)
    df_list = np.split(df, df[df.isnull().all(1)].index)
    useful_data = []
    for s_df in df_list:
        s_df = s_df.dropna()
        if s_df.shape[0] > 6:
            useful_data.append(s_df)
    df = pd.concat(df_list)
    df = df.dropna()
    sentences = []
    for g, df_p in df.groupby(np.arange(df.shape[0]) // config["sequence_length"]):
        if len(df_p["pos"].tolist()) is config["sequence_length"]:
            df_p = df_p.append(pd.Series(), ignore_index=True)
            sentences.append(df_p)
    print("number of samples: ", len(sentences))
    #shuffle(sentences)
    return sentences


if __name__ == '__main__':

    tokens = return_tokenized_lecture_data()
    df = convert_to_df_with_pos_tags(tokens)
    save_output(df)
    f_path = config["path_to_save"] + "/" + config["file_name_to_save"] + ".csv"
    sentences = get_cleaned_data(f_path)
    nrow = len(sentences)
    print(nrow)
    len = int(nrow*config["train_size"])
    print("Train: ", len)
    train_dev = int((nrow-len)/2)
    print("Test/Dev: ", nrow-len)
    df_train = sentences[:len]
    #shuffle(df1)
    df_test = sentences[len:len+train_dev]
    df_dev = sentences[len+train_dev:]
    df_train = pd.concat(df_train)
    df_test = pd.concat(df_test)
    df_dev = pd.concat(df_dev)
    save_txt(df_train, config["path_to_save"], "train_no_shuffle")
    save_txt(df_test, config["path_to_save"], "test_no_shuffle")
    save_txt(df_dev, config["path_to_save"], "dev_no_shuffle")
