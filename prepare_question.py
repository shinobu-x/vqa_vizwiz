import re
import json
import yaml
import torch

def prepare_questions(annotations):
    prepared = []
    questions = [q['question'] for q in annotations]
    for question in questions:
        question = question.lower()
        punctuation_dict = {'.': ' ', "'": '', '?': ' ', '_': ' ', '-': ' ',
                            '/': ' ', ',': ' '}
        conversational_dict = {"thank you": '', "thanks": '', "thank": '',
                               "please": '', "hello": '',
                               "hi ": ' ', "hey ": ' ', "good morning": '',
                               "good afternoon": '', "have a nice day": '',
                               "okay": '', "goodbye": ''}
        rep = punctuation_dict
        rep.update(conversational_dict)
        rep = dict((re.escape(k), v) for k, v in rep.items())
        pattern = re.compile("|".join(rep.keys()))
        question = pattern.sub(lambda m: rep[re.escape(m.group(0))], question)
        question = question.split(' ')
        question = list(filter(None, question))
        prepared.append(question)
    return prepared

def encode_question(question, token_to_index, max_length):
    question_vec = torch.zeros(max_length).long()
    length = min(len(question), max_length)
    for i in range(length):
        token = question[i]
        index = token_to_index.get(token, 0)
        question_vec[i] = index
    return question_vec, max(length, 1)

if __name__ == '__main__':
    config = 'config/default.yaml'
    with open(config, 'r') as f: config = yaml.load(f)
    vocabs = config['annotations']['path_vocabs']
    with open(vocabs, 'r') as f: vocabs = json.load(f)
    token_to_index = vocabs['question']
    max_question_length = config['annotations']['max_length']
    splits = ['train','val']
    for split in splits:
        annotations = config['annotations']['dir'] + '/' + split + '.json'
        with open(annotations, 'r') as f: annotations_json = json.load(f)
        questions = prepare_questions(annotations_json)
        question_json = 'data/' + split + '_questions.json'
        with open(question_json, 'w') as f: json.dump(questions, f, indent = 2)
        with open(question_json, 'r') as f: question_json = json.load(f)
        questions = [encode_question(question, token_to_index,
                                max_question_length) for question in questions]
