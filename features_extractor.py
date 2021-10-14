import argparse
import time

import h5py
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.models as models
import yaml
from torch.autograd import Variable
from tqdm import tqdm

from datasets.images import ImageDataset, get_transform

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.model = models.resnet152(pretrained=True)
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        def save_attention_features(module, input, output):
            self.attention_features = output
        def save_original_features(module, input, output):
            self.original_features = output
        def save_features(module, input, output):
            self.features = output
        self.model.layer4.register_forward_hook(save_attention_features)
        self.model.avgpool.register_forward_hook(save_original_features)
        self.model.layer4.register_forward_hook(save_features)

    def forward(self, x):
        self.model(x)
        return self.original_features, self.attention_features, self.features

def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config', default='config/default.yaml',
                        type=str, help='path to a yaml config file')
    args = parser.parse_args()
    if args.path_config is not None:
        with open(args.path_config, 'r') as f:
            config = yaml.load(f)
            config = config['images']
    cudnn.benchmark = True
    model = FeatureExtractor().to(device)
    model.eval()
    # Resize, Crop, Normalize
    transform = get_transform(config['img_size'])
    if config['type'] == 'test':
        images = config['dir']['test']
    else:
        images = config['dir']['trainval']
    dataset = ImageDataset(images, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['workers'],
        shuffle=False,
        pin_memory=True,
    )
    file = config['data']['file']
    if config['data']['type'] == 'test':
        data_dir = config['data']['dir']['test']
    else:
        data_dir = config['data']['dir']['trainval']
    path = data_dir + '/' + file
    h5_file = h5py.File(path, 'w')
    dummy_input = Variable(torch.ones(1, 3, config['img_size'],
                            config['img_size']), requires_grad=True).to(device)
    _, attention_features, features  = model(dummy_input)
    attention_features_shape = (len(data_loader.dataset), attention_features.size(1),
                                attention_features.size(2),
                                attention_features.size(3))
    original_features_shape = (len(data_loader.dataset), attention_features.size(1))
    features_shape = (len(data_loader.dataset), features.size(1), features.size(2),
                      features.size(3))
    h5_attention = h5_file.create_dataset('attention',
                                    shape=attention_features_shape, dtype='float16')
    h5_original = h5_file.create_dataset('original',
                                    shape=original_features_shape, dtype='float16')
    h5_features = h5_file.create_dataset('features',
                                    shape=features_shape, dtype='float16')
    dt = h5py.special_dtype(vlen=str)
    img_names = h5_file.create_dataset('img_name',
                                    shape=(len(data_loader.dataset),), dtype=dt)
    begin = time.time()
    end = time.time()
    print('Extracting features ...')
    idx = 0
    delta = config['batch_size']
    for i, inputs in enumerate(tqdm(data_loader)):
        inputs_img = Variable(inputs['visual'].to(device), requires_grad=True)
        original_features, attention_features, features = model(inputs_img)
        original_features = original_features.view(-1, 2048)
        h5_original[idx:idx + delta] = \
                        original_features.data.cpu().numpy().astype('float16')
        h5_attention[idx:idx + delta, :, :] = \
                        attention_features.data.cpu().numpy().astype('float16')
        h5_features[idx:idx + delta, :, :] = \
                        features.data.cpu().numpy().astype('float16')
        img_names[idx:idx + delta] = inputs['name']
        idx += delta
    h5_file.close()
    end = time.time() - begin
    print('Finished in {}m and {}s'.format(int(end / 60), int(end % 60)))
    print('Created file : ' + data_dir)

if __name__ == '__main__':
    main()
