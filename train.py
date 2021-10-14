import argparse
import os.path
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import yaml
from torch.autograd import Variable
from tqdm import tqdm

import models
import utils
from datasets import vqa_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def train(model, loader, optimizer, tracker, epoch, split):
    model.train()
    tracker_class, tracker_params = \
                                tracker.MovingMeanMonitor, {'momentum': 0.99}
    tq = tqdm(loader, desc='{} E{:03d}'.format(split, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(split),
                                 tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(split),
                                tracker_class(**tracker_params))
    log_softmax = nn.LogSoftmax(dim=1).to(device)

    for item in tq:
        v = item['visual']
        q = item['question']
        a = item['answer']
        q_length = item['q_length']

        v = Variable(v.to(device))
        q = Variable(q.to(device))
        a = Variable(a.to(device))
        q_length = Variable(q_length.to(device))
        out = model(v, q, q_length)
        nll = -log_softmax(out)
        loss = (nll * a / 10).sum(dim=1).mean()
        acc = utils.vqa_accuracy(out.data, a.data).cpu()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_tracker.append(loss.item())
        acc_tracker.append(acc.mean())
        fmt = '{:.4f}'.format
        tq.set_postfix(loss=fmt(loss_tracker.mean.value),
                       acc=fmt(acc_tracker.mean.value))

def evaluate(model, loader, tracker, epoch, split):
    model.eval()
    tracker_class, tracker_params = tracker.MeanMonitor, {}
    predictions = []
    samples_ids = []
    accuracies = []
    tq = tqdm(loader, desc='{} E{:03d}'.format(split, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(split),
                                 tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(split),
                                tracker_class(**tracker_params))
    log_softmax = nn.LogSoftmax(dim=1).to(device)
    with torch.no_grad():
        for item in tq:
            v = item['visual']
            q = item['question']
            a = item['answer']
            sample_id = item['sample_id']
            q_length = item['q_length']
            v = Variable(v.to(device))
            q = Variable(q.to(device))
            a = Variable(a.to(device))
            q_length = Variable(q_length.to(device))
            out = model(v, q, q_length)
            nll = -log_softmax(out)
            loss = (nll * a / 10).sum(dim=1).mean()
            acc = utils.vqa_accuracy(out.data, a.data).cpu()
            _, answer = out.data.cpu().max(dim=1)
            predictions.append(answer.view(-1))
            accuracies.append(acc.view(-1))
            samples_ids.append(sample_id.view(-1).clone())
            loss_tracker.append(loss.item())
            acc_tracker.append(acc.mean())
            fmt = '{:.4f}'.format
            tq.set_postfix(loss=fmt(loss_tracker.mean.value),
                           acc=fmt(acc_tracker.mean.value))
        predictions = list(torch.cat(predictions, dim=0))
        accuracies = list(torch.cat(accuracies, dim=0))
        samples_ids = list(torch.cat(samples_ids, dim=0))
    eval_results = {
        'answers': predictions,
        'accuracies': accuracies,
        'samples_ids': samples_ids,
        'avg_accuracy': acc_tracker.mean.value,
        'avg_loss': loss_tracker.mean.value
    }
    return eval_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config', default='config/default.yaml',
                        type=str, help='path to a yaml config file')
    args = parser.parse_args()
    if args.path_config is not None:
        with open(args.path_config, 'r') as f:
            config = yaml.load(f)
    prefix = datetime.now().strftime("%y%m%d%H%M%S")
    results_dir = config['training']['results']
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    print('Model logs will be saved in {}'.format(results_dir))
    cudnn.benchmark = True
    # Generate datasets and loaders
    train_loader = vqa_dataset.get_loader(config, split='train')
    val_loader = vqa_dataset.get_loader(config, split='val')
    model = nn.DataParallel(models.Model(config,
                                train_loader.dataset.num_tokens)).to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                model.parameters()), config['training']['lr'])
    features_type = train_loader.dataset.features_type
    print(features_type)
    # Load model weights if necessary
    if config['model']['pretrained'] is not None:
        print("Loading Model from %s" % config['model']['pretrained'])
        pretrained = torch.load(config['model']['pretrained'])
        weights = pretrained['weights']
        model.load_state_dict(weights)
    tracker = utils.Tracker()
    min_loss = 10
    max_accuracy = 0
    best_accuracy = prefix + features_type + '-best_accuracy.pth'
    best_loss = prefix + features_type + '-best_loss.pth'
    path_best_accuracy = results_dir + '/' + best_accuracy
    path_best_loss = results_dir + '/' + best_loss
    for i in range(config['training']['epochs']):
        train(model, train_loader, optimizer, tracker, epoch=i,
              split=config['training']['split'])
        if config['training']['split'] == 'train':
            eval_results = evaluate(model, val_loader, tracker, epoch=i, split='val')
            log_data = {'epoch': i, 'tracker': tracker.to_dict(), 'config': config,
                        'weights': model.state_dict(), 'eval_results': eval_results,
                        'vocabs': train_loader.dataset.vocabs}
            if eval_results['avg_loss'] < min_loss:
                torch.save(log_data, path_best_loss)
                min_loss = eval_results['avg_loss']

            if eval_results['avg_accuracy'] > max_accuracy:
                torch.save(log_data, path_best_accuracy)
                max_accuracy = eval_results['avg_accuracy']
    # Save final model
    log_data = {'tracker': tracker.to_dict(), 'config': config,
                'weights': model.state_dict(), 'vocabs': train_loader.dataset.vocabs}
    final = prefix + '-final.pth'
    path_final_log = results_dir + '/' + final
    torch.save(log_data, path_final_log)

if __name__ == '__main__':
    main()
