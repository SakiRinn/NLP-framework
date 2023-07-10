import argparse
import math
import os
import torch
from torch.utils.data.dataloader import DataLoader
import models
import utils
from transformers import BertModel


class Trainer:

    def __init__(self):
        self.opt = self.set_args()
        self.logger, self.dirname = utils.set_logger(self.opt, 'train')
        self.print_args()

        self.opt.dataset = utils.get_datasets()[self.opt.dataset]
        self.train_set = self.opt.dataset(self.opt.data_dir, mode='train')
        self.train_loader = DataLoader(
            dataset=self.train_set, batch_size=self.opt.batch_size, shuffle=True)
        self.test_set = self.opt.dataset(self.opt.data_dir, mode='test')
        self.test_loader = DataLoader(
            dataset=self.test_set, batch_size=self.opt.batch_size, shuffle=False)
        self.val_set = self.opt.dataset(self.opt.data_dir, mode='val')
        self.val_loader = DataLoader(
            dataset=self.val_set, batch_size=self.opt.batch_size, shuffle=False)

        self.opt.vocab_len = len(self.dataset.tokenizer.vocab)
        self.opt.num_labels = len(self.dataset.labels)
        self.opt.model = utils.get_models()[self.opt.model]
        pretrain = None
        if isinstance(self.opt.model, models.BERT):
            pretrain = BertModel.from_pretrained(self.opt.pretrained_bert_name)
        elif isinstance(self.opt.model, models.LSTM):
            pretrain = None     # TODO
        self.model = self.opt.model(self.opt, pretrain)

        self.opt.optimizer = utils.get_optimizers()[self.opt.optimizer]
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = self.opt.optimizer(
            _params, lr=self.opt.lr, weight_decay=self.opt.l2_reg)

        self.opt.initializer = utils.get_initializers()[self.opt.initializer]
        self.init_params()
        self.print_params()

        self.opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
            if self.opt.device is None else torch.device(self.opt.device)
        if self.opt.device.type == 'cuda':
            self.logger.info(
                f'Cuda memory allocated: {torch.cuda.memory_allocated(device=self.opt.device.index)}')

    @staticmethod
    def set_args():
        parser = argparse.ArgumentParser()

        # Base
        parser.add_argument("data_dir", type=str,
                            help='The directory of dataset file')
        parser.add_argument('--model', default='bert',
                            choices=utils.get_models().keys(), type=str, required=True)
        parser.add_argument('--dataset', default='goemotions', choices=utils.get_datasets().keys(),
                            type=str, required=True)
        parser.add_argument('--optimizer', default='adam',
                            choices=utils.get_optimizers().keys(), type=str, required=True)
        parser.add_argument('--initializer', default='xavier_uniform',
                            choices=utils.get_initializers().keys(), type=str, required=True)

        # Hyper-parameters
        parser.add_argument('--epoch', default=20, type=int,
                            help='Try larger number for non-BERT models')
        parser.add_argument('--batch-size', default=16,
                            type=int, help='Try 16, 32, 64 for BERT models')
        parser.add_argument('--lr', default=2e-5, type=float,
                            help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
        parser.add_argument('--dropout', default=0.1, type=float)
        parser.add_argument('--l2-reg', default=0.01, type=float)

        # Misc
        parser.add_argument('--device', default=None,
                            type=str, help='e.g. cuda:0')
        parser.add_argument('--seed', default=114514, type=int,
                            help='set seed for reproducibility')
        parser.add_argument('--log-step', default=10, type=int)
        parser.add_argument('--patience', default=5, type=int)

        # Model
        parser.add_argument('--embed-size', default=300, type=int)
        parser.add_argument('--hidden-size', default=300, type=int)
        parser.add_argument('--max-seq-len', default=50, type=int)
        parser.add_argument('--pretrained-bert-name',
                            default='bert-base-uncased', type=str)

        # dataset == Goemotions
        parser.add_argument("--taxonomy", default='original', choices=['original', 'ekman', 'group'],
                            type=str, required=True, help="Taxonomy (original, ekman, group)")

        return parser.parse_args()

    def print_args(self):
        self.logger.info('> Training arguments:')
        for arg in vars(self.opt):
            self.logger.info(f'\t{arg}: {getattr(self.opt, arg)}')

    def init_params(self):
        for child in self.model.children():
            if isinstance(child, BertModel):    # Skip bert params
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
        self.logger.info(f'> Trainable parameters: {n_trainable_params}')
        self.logger.info(f'> Non-trainable parameters: {n_nontrainable_params}')

    def train(self, optimizer, train_loader, val_loader):
        max_val_acc = 0
        max_val_f1 = 0
        max_val_epoch = 0
        global_step = 0
        path = None
        for i_epoch in range(self.opt.num_epoch):
            self.logger.info('=' * 50)
            self.logger.info(f'>  Epoch: {i_epoch}')
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, batch in enumerate(train_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [batch[col].to(self.opt.device)
                          for col in self.opt.inputs_cols]
                outputs = self.opt.model(inputs)
                targets = batch['polarity'].to(self.opt.device)

                loss = self.model.loss(outputs, targets)
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    self.logger.info(
                        '   loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            val_acc, val_f1 = self._evaluate_acc_f1(val_loader)
            self.logger.info(
                '>  val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                max_val_epoch = i_epoch
                filename = '{0}_{1}_{2}acc.pth'.format(
                    self.opt.model_name, self.opt.dataset, round(val_acc, 4))
                path = os.path.join(self.dirname, filename)
                torch.save(self.model.state_dict(), path)
                self.logger.info(f'>> Saved: {filename}')
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
            if i_epoch - max_val_epoch >= self.opt.patience:
                print('>  Early stop.')
                break

        return path

    def run(self):
        best_path = self.train(self.optimizer, self.train_loader, self.val_loader)
        self.model.load_state_dict(torch.load(best_path))
        test_acc, test_f1 = self._evaluate_acc_f1(self.test_loader)
        self.logger.info(f'>  test_acc: {test_acc:.4f}, test_f1: {test_f1:.4f}')
