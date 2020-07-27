import pandas as pd
import numpy as np
from random import shuffle
import itertools
import os

TRAIN = .8


def divide_chunks(lst, n):
    new_list = list((lst[i:i + n] for i in range(0, len(lst), n)))
    return new_list


def save_output(data, filename, path="./output"):
    path = os.path.join(path, filename+".csv")
    data.to_csv(path_or_buf=path, index=False, header=True)


lecture_data = pd.read_csv("./data/all_stanford_data_txt.csv")
lecture_data_sentence_list = np.split(lecture_data, lecture_data[lecture_data.isnull().all(1)].index)
num_lecture_sentences = len(lecture_data_sentence_list)
lectures_train = int(TRAIN*num_lecture_sentences)
lectures_dev_test = num_lecture_sentences - lectures_train
data_lecture_train = lecture_data_sentence_list[:lectures_train]
data_lecture_dev = lecture_data_sentence_list[lectures_train:lectures_train+int(lectures_dev_test/2)]
data_lecture_test = lecture_data_sentence_list[lectures_train+int(lectures_dev_test/2):]
print("Train lecture sentences: ", len(data_lecture_train))
print("Dev lecture sentences: ", len(data_lecture_dev))
print("Test lecture sentences: ", len(data_lecture_test))

daily_dialogue_train = pd.read_csv("./data/dialogues_train.csv")
daily_dialogue_train_sentence_list = np.split(daily_dialogue_train, daily_dialogue_train[daily_dialogue_train.isnull().all(1)].index)
daily_dialogue_dev = pd.read_csv("./data/dialogues_validation.csv")
daily_dialogue_dev_sentence_list = np.split(daily_dialogue_dev, daily_dialogue_dev[daily_dialogue_dev.isnull().all(1)].index)
daily_dialogue_test = pd.read_csv("./data/dialogues_test.csv")
daily_dialogue_test_sentence_list = np.split(daily_dialogue_test, daily_dialogue_test[daily_dialogue_test.isnull().all(1)].index)
print("Train daily dialogue sentences: ", len(daily_dialogue_train_sentence_list))
print("Dev daily dialogue sentences: ", len(daily_dialogue_dev_sentence_list))
print("Test daily dialogue sentences: ", len(daily_dialogue_test_sentence_list))

all_train = data_lecture_train + daily_dialogue_train_sentence_list
all_dev = data_lecture_dev + daily_dialogue_dev_sentence_list
all_test = data_lecture_test + daily_dialogue_test_sentence_list

chunks_train = divide_chunks(all_train, 10)
chunks_dev = divide_chunks(all_dev, 10)
chunks_test = divide_chunks(all_test, 10)

shuffle(chunks_train)
shuffle(chunks_dev)
shuffle(chunks_test)

train = itertools.chain.from_iterable(chunks_train)
dev = itertools.chain.from_iterable(chunks_dev)
test = itertools.chain.from_iterable(chunks_test)

train_df = pd.concat(train)
dev_df = pd.concat(dev)
test_df = pd.concat(test)

print("Tokens train: ", train_df.shape[0])
print("Tokens dev: ", dev_df.shape[0])
print("Tokens test: ", test_df.shape[0])

#save_output(train_df, "train")
#save_output(dev_df, "dev")
#save_output(test_df, "test")
