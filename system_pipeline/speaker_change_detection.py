"""
This module contains functionality to detect speaker changes from text.
"""

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import itertools

CONFIG = {
    "cache": "./cache_dir",
    "sc_model": "./bert_sc_model",
}


class SCDetector():
    def __init__(self):
        self.label_list = ["SC", "O"]
        self.model = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=CONFIG["sc_model"],
                                                                     cache_dir=CONFIG["cache"])
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=CONFIG["sc_model"],
                                                       cache_dir=CONFIG["cache"])

    def align_sentences_with_sc_tags(self, labeled_output_tuples):
        """
        Aligns predicted speaker change labels with sentences.
        :param labeled_output_tuples: list of POS-tagged tuples
        :return: list of tuples where tuple[0] is list of sentence tokens and
                 tuple[1] is boolean speaker change label
        """
        sentence_list = []
        sentence = []
        sc_label = True
        for idx, prediction in enumerate(labeled_output_tuples):
            sentence.append(prediction[0])
            if prediction[0] == '.':
                if idx < len(labeled_output_tuples)-1:
                    sentence_list.append((sentence, sc_label))
                    sentence = []
                    if labeled_output_tuples[idx+1][1] == "SC":
                        sc_label = True
                    else:
                        sc_label = False
                else:
                    sentence_list.append((sentence, sc_label))
        return sentence_list

    def predict_speakers(self, sentence_list):
        """
        Performs POS-tagging to predict speaker changes.
        :param sentence_list: list of lists where each list contains tokens of a single sentence
                 with dot for detected sentence boundaries
        :return: returns list of tuples where tuple[0] is list of tokens of a single sentence
                 and tuple[1] is speaker change label
        """
        print("[INFO] Predicting and aligning speaker change information")
        window_size = 7
        if len(sentence_list) > window_size:
            output_sequences = []
            for idx in range(len(sentence_list)):
                if (idx + window_size - 1) < len(sentence_list):
                    current_sequence = list(itertools.chain.from_iterable(sentence_list[idx:idx + window_size]))
                    inputs = self.tokenizer.encode(current_sequence, return_tensors="pt")
                    outputs = self.model(inputs)[0]
                    predictions = torch.argmax(outputs, dim=2)

                    labeled_output = []
                    for token, prediction in zip(current_sequence, predictions[0].tolist()[1:-1]):
                        labeled_output.append((token, self.label_list[prediction]))
                    output_sequences.append(labeled_output)

            labeled_output = []
            for idx, sample in enumerate(output_sequences):
                if idx == 0:
                    sentences = 0
                    for idx_inner, pair in enumerate(sample):
                        if idx_inner == 0 or sample[idx_inner - 1][0] == ".":
                            sentences += 1
                        if sentences < 5:
                            labeled_output.append(pair)
                            
                elif idx == len(output_sequences)-1:
                    sentences = 0
                    for idx_inner, pair in enumerate(sample):
                        if idx_inner == 0 or sample[idx_inner - 1][0] == ".":
                            sentences += 1
                        if sentences > 3:
                            labeled_output.append(pair)
                else:
                    sentences = 0
                    for idx_inner, pair in enumerate(sample):
                        if idx_inner == 0 or sample[idx_inner - 1][0] == ".":
                            sentences += 1
                        if sentences == 4:
                            labeled_output.append(pair)

        else:
            sentence_list = list(itertools.chain.from_iterable(sentence_list))
            inputs = self.tokenizer.encode(sentence_list, return_tensors="pt")
            outputs = self.model(inputs)[0]
            predictions = torch.argmax(outputs, dim=2)

            labeled_output = []
            for token, prediction in zip(sentence_list, predictions[0].tolist()[1:-1]):
                labeled_output.append((token, self.label_list[prediction]))

        return self.align_sentences_with_sc_tags(labeled_output)
