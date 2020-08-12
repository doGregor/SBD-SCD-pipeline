# Project Module Information Science SS2020
 Sentence Boundary Detection and Speaker Change Detection
 
### How to run the pipeline
1. Install dependencies: Navigate to the projects root directory and run `pip install -r requirements.txt`
2. Download BERT models `bert_sbd_model` and `bert_sc_model` from --LINK-- (because of size) and move them
    in `system_pipeline` folder
3. Place the file that should be processed as `.txt`-file in the `system_pipeline/input_dir` folder and run
    `python3 pipeline.py`. The output is saved as txt-file in `system_pipeline/output_dir`. Note: you can also
    process multiple files
 
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

### Model: bert-base-uncased, Data: daily dialogue (https://www.aclweb.org/anthology/I17-1099/)
`sentence_boundary_detection/daily_dialogue/data`
Raw data that was used and csv files (train/dev/test)

`sentence_boundary_detection/daily_dialogue/results_huggingface_bert_base`
Contains a report of results that are achieved by fine-tuning the bert-base-uncased model, the original test data and the predicted test data.

### Model: bert-base-uncased, Data: mix of Stanford lecture data and daily dialogue data
#### details on data preprocessing can be found in `sentence_boundary_detection/mixed_data/results_huggingface_bert_base/general_information_data.pdf`
`sentence_boundary_detection/mixed_data/data`
Csv files (train/dev/test)

`sentence_boundary_detection/mixed_data/results_huggingface_bert_base`
Contains a report of results that are achieved by fine-tuning the bert-base-uncased model, information about the data, the original test data and the predicted test data.

## Speaker Change Detection
