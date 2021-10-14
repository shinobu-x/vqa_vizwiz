import re
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

def encode_question(question, token_to_index, max_length):
    question_vec = torch.zeros(max_length).long()
    length = min(len(question), max_length)
    for i in range(length):
        token = question[i]
        index = token_to_index.get(token, 0)
        question_vec[i] = index
    return question_vec, max(length, 1)

def encode_answers(answers, answer_to_index):
    answer_vec = torch.zeros(len(answer_to_index))
    for answer in answers:
        index = answer_to_index.get(answer)
        if index is not None:
            answer_vec[index] += 1
    return answer_vec
