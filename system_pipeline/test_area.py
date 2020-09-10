import os
from text_processing_functionality import get_tokenized_data
from sentence_boundary_detection import SBDDetector

'''
CONFIG = {
    "input": "./input_dir",
    "output": "./output_dir",
    "working": "./working_dir",
    "sequence_length": 64,

    "cache": "./cache_dir",
    "sbd_model": "./bert_sbd_model"
}

if __name__ == '__main__':


    ### IN SBD MODULE
        def test_sbd(self, tokens):
        print(tokens)
        from transformers import pipeline
        nlp = pipeline('ner', model=self.model, tokenizer=self.tokenizer, grouped_entities=False)
        return nlp(tokens)

    sbd_detector = SBDDetector()

    for file in os.listdir(CONFIG['input']):
        print("[INFO] Processing following file:", file)
        path_of_current_file = CONFIG['input'] + "/" + str(file)

        file_data = open(path_of_current_file, "r")
        data = ""
        for line in file_data:
            data += line.lower()

        output = sbd_detector.test_sbd(tokens=data)

        print(output)


        #tokens = get_tokenized_data(path_of_current_file)

        #predictions, label_ids, metrics = sbd_detector.test_sbd(tokens=tokens)
        #print(predictions, label_ids, metrics)
'''

