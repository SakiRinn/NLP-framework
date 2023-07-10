import argparse
import json
import logging
import math
import os
import re
from time import localtime, strftime

import torch
from attrdict import AttrDict
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm, trange
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from datasets.goemotions import GoEmotions

import models
import utils


class Trainer:

    def __init__(self, config=None):
        if config is not None:
            with open(config) as f:
                self.opt = AttrDict(json.load(f))
        else:
            self.opt = self.init_args()

        self.logger, self.dirname = self.init_logger('train')
        self.print_args()
        self.opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
            if self.opt.device is None else torch.device(self.opt.device)

        self.train_loader = self.prepare_loader(mode='train')
        self.val_loader = self.prepare_loader(mode='val')
        self.test_loader = self.prepare_loader(mode='test')

        self.model = self.prepare_model()
        self.optimizer, self.scheduler = self.prepare_optimizer()

        self.opt.initializer = utils.get_initializers()[self.opt.initializer]
        self.init_params()
        self.print_params()

        if self.opt.resume_dir != '':
            self.load(self.opt.resume_dir)

        if self.opt.device.type == 'cuda':
            self.logger.info(
                f'Cuda memory allocated: {torch.cuda.memory_allocated(device=self.opt.device.index) / 1024**3:.2f} GB')

    def init_logger(self, name='print'):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logging.basicConfig(format='[%(asctime)s | %(levelname)s | %(name)s]  %(message)s',
                            datefmt='%Y.%m.%d-%H:%M:%S',
                            level=logging.INFO)
        dirname = f'{self.opt.model}_{self.opt.dataset}_{strftime("%m%d%H%M", localtime())}'
        os.makedirs(os.path.join('outputs', dirname), exist_ok=True)
        logger.addHandler(logging.FileHandler(os.path.join('outputs', dirname, f'{name}.log')))
        return logger, dirname

    @staticmethod
    def init_args():
        parser = argparse.ArgumentParser()

        # Base
        parser.add_argument('dataset', choices=utils.get_datasets().keys(), type=str)
        parser.add_argument("data_dir", type=str,
                            help='The directory of dataset file')
        parser.add_argument("--resume-dir", type=str,
                            default='', help="The folder you'll resume")
        parser.add_argument('--model', default='bert',
                            choices=utils.get_models().keys(), type=str)
        parser.add_argument('--optimizer', default='adamW',
                            choices=utils.get_optimizers().keys(), type=str)
        parser.add_argument('--initializer', default='xavier_uniform',
                            choices=utils.get_initializers().keys(), type=str)

        # Train
        parser.add_argument('--epoch', default=10, type=int,
                            help='Try larger number for non-BERT models')
        parser.add_argument('--batch-size', default=32,
                            type=int, help='Try 16, 32, 64 for BERT models')
        parser.add_argument('--lr', default=5e-5, type=float,
                            help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
        parser.add_argument('--dropout', default=0.1, type=float)
        parser.add_argument('--l2-reg', default=0.01, type=float)
        parser.add_argument('--max-grad-norm', default=1.0, type=float)
        parser.add_argument('--warmup-proportion', default=0.1, type=float)

        # Misc
        parser.add_argument('--device', default=None,
                            type=str, help='e.g. cuda:0')
        parser.add_argument('--seed', default=114514, type=int,
                            help='set seed for reproducibility')
        parser.add_argument('--log-step', default=25, type=int)
        parser.add_argument('--patience', default=5, type=int)

        # Model
        parser.add_argument('--embed-size', default=256, type=int)
        parser.add_argument('--hidden-size', default=768, type=int,
                            help='If using pretrained bert, it must be 768')
        parser.add_argument('--max-seq-len', default=50, type=int)
        parser.add_argument('--pretrained-model',
                            default='pretrained/bert-base-uncased', type=str)
        parser.add_argument('--pretrained-tokenizer',
                            default='pretrained/goemotions-tokenizer', type=str)

        # dataset == Goemotions
        parser.add_argument("--taxonomy", default='original', choices=['original', 'ekman', 'group'],
                            type=str, help="Taxonomy (original, ekman, group)")
        parser.add_argument("--threshold", default=0.3, type=float)

        return parser.parse_args()

    def print_args(self):
        self.logger.info('Training arguments:')
        for arg in vars(self.opt):
            self.logger.info(f'>  {arg}: {getattr(self.opt, arg)}')

    def init_params(self):
        for child in self.model.children():
            if isinstance(child, BertModel) or 'Loss' in child.__class__.__name__:
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def print_params(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        self.logger.info(f'Trainable parameters: {n_trainable_params}')
        self.logger.info(f'Non-trainable parameters: {n_nontrainable_params}')

    def prepare_loader(self, mode='all'):
        if isinstance(self.opt.dataset, str):
            self.opt.dataset = utils.get_datasets()[self.opt.dataset]
            self.labels = self.opt.dataset.get_labels(self.opt.data_dir)
            if self.opt.pretrained_tokenizer != '':
                self.tokenizer = BertTokenizer.from_pretrained(self.opt.pretrained_tokenizer)
            else:
                self.tokenizer = self.opt.dataset.build_tokenizer(self.opt.data_dir)
        dataset = self.opt.dataset(self.opt.data_dir, mode=mode)
        return DataLoader(dataset, batch_size=self.opt.batch_size, shuffle=(mode == 'train'))

    def prepare_model(self):
        self.opt.vocab_len = len(self.tokenizer)
        self.opt.num_labels = len(self.labels)
        self.opt.model = utils.get_models()[self.opt.model]
        pretrain = None
        if self.opt.model == models.BERT:
            pretrain = BertModel.from_pretrained(self.opt.pretrained_model)
        elif self.opt.model == models.LSTM:
            pretrain = None     # TODO
        return self.opt.model(self.opt, pretrain).to(self.opt.device)

    def prepare_optimizer(self):
        self.opt.step = len(self.train_loader) * self.opt.epoch
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.opt.optimizer = utils.get_optimizers()[self.opt.optimizer]
        optimizer = self.opt.optimizer(
            _params, lr=self.opt.lr, weight_decay=self.opt.l2_reg)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(
                self.opt.step * self.opt.warmup_proportion),
            num_training_steps=self.opt.step)
        return optimizer, scheduler

    def load(self, resume_dir):
        epochs = []
        model_name = ''
        for filename in os.listdir(resume_dir):
            match = re.match(r'^model_(\d+)_', filename)
            if match:
                epoch = int(match[1])
                if epoch > max(epochs):
                    model_name = filename
                epochs.append(epoch)
        self.model.load_state_dict(torch.load(
            os.path.join(resume_dir, model_name)))
        self.optimizer.load_state_dict(torch.load(
            os.path.join(resume_dir, "optimizer.pt")))
        self.scheduler.load_state_dict(torch.load(
            os.path.join(resume_dir, "scheduler.pt")))

    def save(self, info='save'):
        torch.save(
            self.model.state_dict(),
            os.path.join(self.dirname, f'model_{info}.pt'),
        )
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(self.dirname, f'optimizer.pt'),
        )
        torch.save(
            self.scheduler.state_dict(),
            os.path.join(self.dirname, f'scheduler.pt'),
        )
        self.logger.info(f'Save new checkpoints.')

    def train(self, train_loader, val_loader):
        max_val_acc, max_val_f1 = 0, 0
        max_val_epoch, global_step = 0, 0
        path = None

        for epoch in trange(self.opt.epoch, desc="Epoch"):
            self.logger.info(f'Epoch: {epoch}')
            n_correct, n_total, loss_total = 0, 0, 0
            self.model.train()

            for batch, gt in tqdm(train_loader, desc="Iteration"):
                global_step += 1
                self.optimizer.zero_grad()

                inputs = {k: v.to(self.opt.device) \
                          for k, v in self.opt.dataset.input_packets(batch).items()}
                del inputs['input']
                batch, gt = batch.to(self.opt.device), gt.to(self.opt.device)
                outputs = self.model(batch, gt, **inputs)
                loss = outputs[-1]
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()

                n_correct += (torch.argmax(outputs, -1) == gt).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    self.logger.info(
                        '>  loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            val_acc, val_f1 = self.evaluate(val_loader)
            self.logger.info(
                '>> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                max_val_epoch = epoch
                self.save(info=f'{epoch}e_{val_acc:.4f}acc')
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
            if epoch - max_val_epoch >= self.opt.patience:
                self.logger.info('Early stop!')
                break

        return path

    def evaluate(self, val_loader):
        total_loss, global_step = 0, 0
        self.model.eval()

        self.logger.info('Evaluating...')
        with torch.no_grad():
            for i, (batch, gt) in enumerate(val_loader):
                inputs = {k: v.to(self.opt.device) \
                          for k, v in self.opt.dataset.input_packets(batch)}
                del inputs['input']
                outputs = self.model(batch, gt, **inputs)
                pred = outputs[0]
                loss = outputs[-1]

                if isinstance(self.dataset, GoEmotions):
                    pred = torch.sigmoid(pred)
                else:
                    pred = torch.softmax(pred, dim=-1)
                pred = torch.where(pred > self.opt.threshold, 1, 0)

                total_loss += loss
                global_step += 1

                results = {"loss": total_loss / global_step}
                results.update(utils.compute_metrics(pred, gt))
        return results

    def run(self):
        best_path = self.train(self.train_loader, self.val_loader)
        self.model.load_state_dict(torch.load(best_path))
        results = self.evaluate(self.test_loader)
        for key in sorted(results.keys()):
            self.logger.info(f">  {key} = {results[key]:.4f}")


if __name__ == '__main__':
    trainer = Trainer(config=None)
    trainer.run()
