import json
import os
import os.path

import h5py
import torch
import torch.utils.data as data

from datasets.features import FeaturesDataset
from preprocessing.utils import prepare_questions, prepare_answers
from preprocessing.utils import encode_question, encode_answers


def get_loader(config, split):
    split = VQADataset(
        config,
        split
    )
    loader = torch.utils.data.DataLoader(
        split,
        batch_size=config['training']['batch_size'],
        shuffle=True if split == 'train' or split == 'trainval' else False,
        pin_memory=True,
        num_workers=config['training']['data_workers'],
        collate_fn=collate_fn,
    )
    return loader

def collate_fn(batch):
    batch.sort(key=lambda x: x['q_length'], reverse=True)
    return data.dataloader.default_collate(batch)

class VQADataset(data.Dataset):
    def __init__(self, config, split):
        super(VQADataset, self).__init__()

        with open(config['annotations']['path_vocabs'], 'r') as f:
            vocabs = json.load(f)

        annotations_dir = config['annotations']['dir']

        path_ann = os.path.join(annotations_dir, split + ".json")
        with open(path_ann, 'r') as f:
            self.annotations = json.load(f)

        self.max_question_length = config['annotations']['max_length']
        self.split = split

        self.vocabs = vocabs
        self.token_to_index = self.vocabs['question']
        self.answer_to_index = self.vocabs['answer']
        self.questions = prepare_questions(self.annotations)
        self.raw_questions = self.questions
        self.questions = [encode_question(q, self.token_to_index,
                        self.max_question_length) for q in self.questions]
        if self.split != 'test':
            self.answers = prepare_answers(self.annotations)
            self.raw_answers = self.answers
            self.answers = [encode_answers(a, self.answer_to_index) for a in
                            self.answers]
        if self.split == "train" or self.split == "trainval":
            self._filter_unanswerable_samples()
        type = config['images']['data']['type']
        if type == 'test':
            path = config['images']['data']['dir']['test']
        else:
            path = config['images']['data']['dir']['trainval']
        file = path + '/' + config['images']['data']['file']
        with h5py.File(file, 'r') as f:
            img_names = f['img_name'][()]
        self.name_to_id = {}
        #self.name_to_id = {name: i for i, name in enumerate(img_names)}
        for i, k in enumerate(img_names):
            d = {k.decode(): i}
            self.name_to_id.update(d)
        for k in self.name_to_id:
            if self.name_to_id.get(k) is None:
                print(k)
        print(len(self.name_to_id))
        self.img_names = [s['image'] for s in self.annotations]
        # load features
        self.features_type = config['images']['mode']
        self.features = FeaturesDataset(file, self.features_type)

    def _filter_unanswerable_samples(self):
        a = []
        q = []
        annotations = []
        for i in range(len(self.answers)):
            if len(self.answers[i].nonzero()) > 0:
                a.append(self.answers[i])
                q.append(self.questions[i])
                annotations.append(self.annotations[i])
        self.answers = a
        self.questions = q
        self.annotations = annotations

    @property
    def num_tokens(self):
        return len(self.token_to_index) + 1  # add 1 for <unknown> token at index 0

    def __getitem__(self, i):

        item = {}
        item['question'], item['q_length'] = self.questions[i]
        if self.split != 'test':
            item['answer'] = self.answers[i]
        img_name = self.img_names[i]
        feature_id = self.name_to_id[img_name]
        item['img_name'] = self.img_names[i]
        item['visual'] = self.features[feature_id]
        item['sample_id'] = i
        return item

    def __len__(self):
        return len(self.questions)
