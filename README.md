# Project Information Science 2020
 This work proposes a deep learning BERT-based approach for sentence boundary detection and speaker change detection
 in unpunctuated texts. We are able to report state-of-the-art results for solutions to both problems. For more
information we refer to the related paper (accepted at NLPBT workshop at the Web Conference 2021): https://www.researchgate.net/publication/350888509_Making_Sense_of_Subtitles_Sentence_Boundary_Detection_and_Speaker_Change_Detection_in_Unpunctuated_Texts

If you use parts of our code or in general adapt our approach we kindly ask you to cite our work as follows:
```
@inproceedings{donabauer_kruschwitz_corney_2021,
author = {Donabauer, Gregor and Kruschwitz, Udo and Corney, David},
title = {Making Sense of Subtitles: Sentence Boundary Detection and Speaker Change Detection in Unpunctuated Texts},
year = {2021},
booktitle = {Companion Proceedings of the Web Conference 2021 (WWW '21 Companion)},
publisher = {ACM},
venue = {Ljubljana, Slovenia},
doi = {10.1145/3442442.3451894},
isbn = {978-1-4503-8313-4/21/04},
address = {New York, NY}
}
```

### How to run the pipeline
1. Install dependencies: Navigate to the projects root directory and run `pip install -r requirements.txt`
2. Train BERT models `bert_sbd_model` and `bert_sc_model` as described below and move them
    in `system_pipeline` folder (if you want access to the ready-to-use models you are welcome to contact us)
3. Place the file(s) that should be processed as `.txt`-file(s) in the `system_pipeline/input_dir` folder and run
    `python3 pipeline.py`. The output is saved as txt-file in `system_pipeline/output_dir`. Note: you can also
    process multiple files
 
## Sentence Boundary Detection
`sentence_boundary_detection/lecture_data/`
For this sepcific dataset: utilized data, optional data preprocessing, detailed evaluation results, config for model training.

Raw data originates from:
Lecture|Source
-------|------
Stanford CS224N: Natural Language Processing with Deep Learning (Winter 2019)|https://www.youtube.com/playlist?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z
Stanford Lecture Collection Human Behavioral Biology|https://www.youtube.com/playlist?list=PL848F2368C90DDC3D
Stanford CS330: Multi-Task and Meta-Learning (2019)|https://www.youtube.com/playlist?list=PLoROMvodv4rMC6zfYmnD7UG3LVvwaITY5
Stanford CS234: Reinforcement Learning (Winter 2019)|https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u
Stanford CS229: Machine Learning (Autumn 2018)|https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU
Stanford CS224U: Natural Language Understanding (Spring 2019)|https://www.youtube.com/playlist?list=PLoROMvodv4rObpMCir6rNNUlFAn56Js20
Stanford CS221: AI Principles and Techniques (Autumn 2019)|https://www.youtube.com/playlist?list=PLoROMvodv4rO1NB9TD4iUZ3qghGEGtqNX

`sentence_boundary_detection/daily_dialogue/`
For this sepcific dataset: utilized data, optional data preprocessing, detailed evaluation results, config for model training.
Data originates from https://www.aclweb.org/anthology/I17-1099/

`sentence_boundary_detection/mixed_data/`
For this sepcific dataset: utilized data, optional data preprocessing, detailed evaluation results, config for model training.

`sentence_boundary_detection/le_2020_vs_BERT/`
Results of comparing BERT vs Le on the DailyDialogue dataset version of Le (2020). Results 5-fold cross-validation and training configuration for both models. NOTE: There are samples that are labeled with either no or more than one BOS tags since we use sequence labeling and no class labels. The fraction of those wrongly labeled samples is larger within the results using the apporach of Le (2020) in contrast to those of BERT. Since we cannot assign class labels to those samples, there is only a small difference in the resulting F1 scores.

`sentence_boundary_detection/evaluation/`
Script to evaluate predictions in CoNLL-2003 format with our labels.

`sentence_boundary_detection/training/`
Scripts that are necessary to train a model with a given config and given data (for details see bottom of the page)

## Speaker Change Detection
`speaker_change_detection/original_data/`
Results and preprocessing scripts for original data

`speaker_change_detection/training/`
Scripts, labels.txt and config file that are needed for model training. (for details see bottom of the page)

`speaker_change_detection/unpunctuated_data/`
Results and preprocessing scripts for unpunctuated data

`speaker_change_detection/evaluate/`
Scripts to evaluate the resulting .txt files in CoNLL-2003 format. Can be used if data preprocessing from `original_data` folder has been used.

Since the data used for SCD is too extensive you have to run our python scripts by yourself, to generate the train/dev/test files:
- download the data from Meng et al. (2017) and run their scripts as explained (https://sites.google.com/site/textscd/)
- run the files in the `speaker_change_detection/original_data/` folder in the order they are numbered (Important: you have to specify your folderlocation within the scripts)
- `01_preprocess_original_data.py` needs to be run 3 times: Input is in each run one of the folders (train/dev/val) that are resulting from the original Meng et al. data preprocessing
- `02_data_to_pos_SW.py` and `03_generate_ground_truth_SW.py` only need to be run if you want to try out and evaluate the sliding window proceeding (Input location to specify in code is a folder that contains the 3 files produced by `01_preprocess_original_data.py`.
- The files produced by `04_data_to_pos_NO_SW.py`should be renamed train.txt, dev.txt and test.txt and are used for SCD BERT model training. (Same input to specify in script as with `02_data_to_pos_SW.py`).
- `05_generate_ground_truth_NO_SW.py` is used to generate ground truth data that is needed to evaluate the predictions on the test data that are produced through model training (if `do_predict` is set to `True`).

## How to rerun the training:
Details (legacy): https://github.com/huggingface/transformers/tree/master/examples/legacy/token-classification

What you need (at least):
- config file (.json)
- data folder with 3 data files (.txt); have to be named train.txt, dev.txt, test.txt
- `run_ner.py` and `utils_ner.py` files
- file with your labels (.txt)
- empty output folder

More details, like utilized model, maximum sequence length or batch size are set within config file. If all required components are available run `python3 run_ner.py config.json`
All utilized datasets as well as the associated configs are available within the repository.
