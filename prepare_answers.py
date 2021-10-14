import json
import yaml
import re
import torch

def prepare_answers(annotations):
    answers = [[a['answer'] \
                for a in ans_dict['answers']] \
               for ans_dict in annotations]
    prepared = []
    for sample_answers in answers:
        prepared_sample_answers = []
        for answer in sample_answers:
            answer = answer.lower()
            punctuation_dict = {'.': ' ', "'": '', '?': ' ', '_': ' ',
                                '-': ' ', '/': ' ', ',': ' '}
            rep = punctuation_dict
            rep = dict((re.escape(k), v) for k, v in rep.items())
            pattern = re.compile("|".join(rep.keys()))
            answer = pattern.sub(lambda m: rep[re.escape(m.group(0))], answer)
            prepared_sample_answers.append(answer)
        prepared.append(prepared_sample_answers)
    return prepared

def encode_answers(answers, answer_to_index):
    answer_vec = torch.zeros(len(answer_to_index))
    for answer in answers:
        index = answer_to_index.get(answer)
        if index is not None:
            answer_vec[index] += 1
    return answer_vec

if __name__ == '__main__':
    config = 'config/default.yaml'
    with open(config, 'r') as f: config = yaml.load(f)
    vocabs = config['annotations']['path_vocabs']
    with open(vocabs, 'r') as f: vocabs = json.load(f)
    answer_to_index = vocabs['answer']
    splits = ['train', 'val']
    for split in splits:
        annotations = config['annotations']['dir'] + '/' + split + '.json'
        with open(annotations, 'r') as f: annotations_json = json.load(f)
        answers = prepare_answers(annotations_json)
        answers_json = 'data/' + split + '_answers.json'
        with open(answers_json, 'w') as f: json.dump(answers, f, indent = 2)
        with open(answers_json, 'r') as f: answers_json = json.load(f)
        answers = [encode_answers(answer,
                                  answer_to_index) for answer in answers]
