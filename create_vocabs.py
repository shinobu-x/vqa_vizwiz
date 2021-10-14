import argparse
import itertools
import json
import os
from collections import Counter
from itertools import takewhile
import yaml
from preprocessing.utils import prepare_questions, prepare_answers

def create_question_vocab(questions, min_count=0):
    words = itertools.chain.from_iterable([q for q in questions])
    counter = Counter(words)
    counted_words = counter.most_common()
    selected_words = list(takewhile(lambda x: x[1] >= min_count,
                                    counted_words))
    vocab = {t[0]: i for i, t in enumerate(selected_words, start=1)}
    return vocab

def create_answer_vocab(annotations, top_k):
    answers = itertools.chain.from_iterable(prepare_answers(annotations))
    counter = Counter(answers)
    counted_ans = counter.most_common(top_k)
    vocab = {t[0]: i for i, t in enumerate(counted_ans, start=0)}
    return vocab

parser = argparse.ArgumentParser()
parser.add_argument('--path_config', default='config/default.yaml', type=str,
                    help='path to a yaml config file')

def main():
    global args
    args = parser.parse_args()
    if args.path_config is not None:
        with open(args.path_config, 'r') as handle:
            config = yaml.load(handle)
    dir_path = config['annotations']['dir']
    train_path = os.path.join(dir_path,
                              config['training']['train_split'] + '.json')
    with open(train_path, 'r') as f:
        train_ann = json.load(f)
    questions = prepare_questions(train_ann)
    question_vocab = create_question_vocab(questions,
                                    config['annotations']['min_count_word'])
    answer_vocab = create_answer_vocab(train_ann,
                                       config['annotations']['top_ans'])
    vocabs = {
        'question': question_vocab,
        'answer': answer_vocab,
    }
    with open(config['annotations']['path_vocabs'], 'w') as f:
        json.dump(vocabs, f)
    print("vocabs saved in {}".format(config['annotations']['path_vocabs']))

if __name__ == '__main__':
    main()
