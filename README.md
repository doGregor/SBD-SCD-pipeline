# project_module_ss_2020
 Sentence Boundary Detection and Speaker Identification
 
## Sentence Boundary Detection
### Model: bert-base-uncased, Data: Stanford lecture data
`sentence_boundary_detection/lecture_data/data`
Raw data that was used and preprocessed csv files (train/dev/test):

Lecture|Source
-------|------
Stanford CS224N: Natural Language Processing with Deep Learning (Winter 2019)|https://www.youtube.com/playlist?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z
Stanford Lecture Collection Human Behavioral Biology|https://www.youtube.com/playlist?list=PL848F2368C90DDC3D
Stanford CS330: Multi-Task and Meta-Learning (2019)|https://www.youtube.com/playlist?list=PLoROMvodv4rMC6zfYmnD7UG3LVvwaITY5
Stanford CS234: Reinforcement Learning (Winter 2019)|https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u
Stanford CS229: Machine Learning (Autumn 2018)|https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU
Stanford CS224U: Natural Language Understanding (Spring 2019)|https://www.youtube.com/playlist?list=PLoROMvodv4rObpMCir6rNNUlFAn56Js20
Stanford CS221: AI Principles and Techniques (Autumn 2019)|https://www.youtube.com/playlist?list=PLoROMvodv4rO1NB9TD4iUZ3qghGEGtqNX

`sentence_boundary_detection/lecture_data/data_processing`
Contains a script that transforms the raw text files to train/dev/test data in CoNLL-2003 format (Tjong Kim Sang & De Meulder, 2003).

`sentence_boundary_detection/lecture_data/huggingface_bert_base_2020_06_08`
Contains a report of results that are achieved by fine-tuning the bert-base-uncased model, the original test data and the predicted test data.

### Model: bert-base-uncased, Data: Daily dialogue (https://www.aclweb.org/anthology/I17-1099/)
`sentence_boundary_detection/daily_dialogue/data`
Raw data that was used and csv files (train/dev/test)

## Speaker Identification
