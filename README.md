# Project Information Science 2020
 Sentence Boundary Detection and Speaker Change Detection
 
### How to run the pipeline
1. Install dependencies: Navigate to the projects root directory and run `pip install -r requirements.txt`
2. Download BERT models `bert_sbd_model` and `bert_sc_model` from [--LINK-- (because of size)] (will be published if paper is accepted) and move them
    in `system_pipeline` folder
3. Place the file that should be processed as `.txt`-file in the `system_pipeline/input_dir` folder and run
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
- run the `preprocessing_data.py` (for unpunctuated data) or `preprocessing_original_data.py` (for original data) (Note: you have to specify your folderlocation within the script)
- run the `data_to_pos.py` file (Note: you have to specify your folder location within the script)

## How to rerun the training:
Details: https://github.com/huggingface/transformers/tree/master/examples/token-classification

What you need (at least):
- config file (.json)
- data folder with 3 data files (.txt); have to be named train.txt, dev.txt, test.txt
- `run_ner.py` and `utils_ner.py` files
- file with your labels (.txt)
- empty output folder

More details, like utilized model, maximum sequence length or batch size are set within config file. If all required components are available run `python3 run_ner.py config.json`
All utilized datasets as well as the associated configs are available within the repository.
