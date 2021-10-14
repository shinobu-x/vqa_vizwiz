import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence


class Model(nn.Module):
    def __init__(self, config, embedding_tokens):
        super(Model, self).__init__()
        image_features = config['model']['pooling']['dim_v']
        question_features = config['model']['pooling']['dim_q']
        hidden_features = config['model']['pooling']['dim_h']
        probabilities = config['annotations']['top_ans']
        embedding_size = config['model']['seq2vec']['emb_size']
        glimpses = config['model']['attention']['glimpses']
        dropout_ratio = config['model']['seq2vec']['dropout']

        self.text = TextEncoder(embedding_tokens, embedding_size, question_features,
                                dropout_ratio)
        self.attention = Attention(image_features, question_features,
                                   hidden_features, glimpses, dropout_ratio)
        self.classifier = Classifier(glimpses * image_features + question_features,
                                     hidden_features, probabilities, dropout_ratio)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v, q, q_len):
        q = self.text(q, list(q_len.data))
        v = F.normalize(v, p=2, dim=1)
        attention_maps = self.attention(v, q)
        v = apply_attention(v, attention_maps)
        combined = torch.cat([v, q], dim=1)
        answer = self.classifier(combined)
        return answer

class TextEncoder(nn.Module):
    def __init__(self, num_tokens, embedding_size, question_features, drop=0.0):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(num_tokens, embedding_size, padding_idx=0)
        self.dropout = nn.Dropout(drop)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=question_features,
                            num_layers=1)
        self.question_features = question_features
        self._init_lstm(self.lstm.weight_ih_l0)
        self._init_lstm(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()
        init.xavier_uniform_(self.embedding.weight)

    def _init_lstm(self, weight):
        for w in weight.chunk(4, 0):
            init.xavier_uniform_(w)

    def forward(self, q, q_len):
        embedded = self.embedding(q)
        tanhed = self.tanh(self.dropout(embedded))
        packed = pack_padded_sequence(tanhed, q_len, batch_first=True)
        _, (_, c) = self.lstm(packed)
        return c.squeeze(0)

class Attention(nn.Module):
    def __init__(self, image_features, question_features, hidden_features, glimpses,
                 drop=0.0):
        super(Attention, self).__init__()
        self.conv_v = nn.Conv2d(image_features, hidden_features, 1, bias=False)
        self.fc_q = nn.Linear(question_features, hidden_features)
        self.conv_x = nn.Conv2d(hidden_features, glimpses, 1)
        self.dropout = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v, q):
        v = self.conv_v(self.dropout(v))
        q = self.fc_q(self.dropout(q))
        q = repeat_encoded_question(q, v)
        x = self.relu(v + q)
        x = self.conv_x(self.dropout(x))
        return x

class Classifier(nn.Sequential):
    def __init__(self, input_features, hidden_features, probabilities, drop=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('lin1', nn.Linear(input_features, hidden_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('lin2', nn.Linear(hidden_features, probabilities))

def repeat_encoded_question(q, v):
    n, c = q.size()
    spatial_size = v.dim() - 2
    q = q.view(n, c,
               *([1] * spatial_size)).expand_as(v)
    return q

def apply_attention(v, attention):
    n, c = v.size()[:2]
    glimpses = attention.size(1)
    v = v.view(n, 1, c, -1)
    attention = attention.view(n, glimpses, -1)
    attention = attention.view(n, glimpses, -1)
    attention = F.softmax(attention, dim=-1).unsqueeze(2)
    weighted = attention * v
    weighted_mean = weighted.sum(dim=-1)
    return weighted_mean.view(n, -1)
