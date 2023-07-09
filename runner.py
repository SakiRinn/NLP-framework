import argparse
import math
import os
import torch
import utils
from transformers import BertModel
from datasets.absa import ABSADataset


class Runner:

    def __init__(self):
        self.opt = self.set_args()
        self.logger = utils.set_logger(self.opt)
        self.print_args()

        self.opt.model = utils.get_models()[self.opt.model]
        self.opt.dataset = utils.get_datasets()[self.opt.dataset]
        self.opt.optimizer = utils.get_optimizers()[self.opt.optimizer]
        self.opt.initializer = utils.get_initializers()[self.opt.initializer]
        self.init_params()
        self.print_params()

        self.opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
            if self.opt.device is None else torch.device(self.opt.device)
        if self.opt.device.type == 'cuda':
            self.logger.info(
                f'Cuda memory allocated: {torch.cuda.memory_allocated(device=self.opt.device.index)}')

        if 'bert' in self.opt.model:
            tokenizer = utils.Tokenizer4Bert(
                self.opt.max_seq_len, self.opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(self.opt.pretrained_bert_name)
            self.model = self.opt.model(bert, self.opt).to(self.opt.device)
        else:
            tokenizer = utils.build_tokenizer(
                fnames=[self.opt.dataset_file['train'],
                        self.opt.dataset_file['test']],     # TODO
                max_seq_len=self.opt.max_seq_len,
                dat_fname=f'{self.opt.dataset.__class__.__name__}_tokenizer.dat')
            embedding_matrix = utils.build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=self.opt.embed_dim,
                dat_fname=f'{self.opt.embed_dim}_{self.opt.dataset.__class__.__name__}_embedding_matrix.dat')
            self.model = self.opt.model(
                embedding_matrix, self.opt).to(self.opt.device)
        self.trainset = ABSADataset(
            self.opt.dataset_file['train'], tokenizer)
        self.testset = ABSADataset(
            self.opt.dataset_file['test'], tokenizer)

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

        # For non-bert models
        parser.add_argument('--embed-dim', default=300, type=int)
        parser.add_argument('--hidden-dim', default=300, type=int)
        # For bert models
        parser.add_argument('--bert-dim', default=768, type=int)
        parser.add_argument('--pretrained-bert-name',
                            default='bert-base-uncased', type=str)
        parser.add_argument('--max-seq-len', default=85, type=int)

        # model == bert_lcf
        parser.add_argument('--local-context-focus', default='cdm',
                            type=str, help='local context focus mode, cdw or cdm')
        parser.add_argument('--SRD', default=3, type=int,
                            help='semantic-relative-distance, see the paper of LCF-BERT model')

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
            if type(child) != BertModel:        # Skip bert params
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
        self.logger.info(f'> n_trainable_params: {n_trainable_params}')
        self.logger.info(f'> n_nontrainable_params: {n_nontrainable_params}')

    def train(self, criterion, optimizer, train_dataloader, val_dataloader):
        max_val_acc = 0
        max_val_f1 = 0
        max_val_epoch = 0
        global_step = 0
        path = None
        for i_epoch in range(self.opt.num_epoch):
            self.logger.info('>' * 100)
            self.logger.info(f'epoch: {i_epoch}')
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, batch in enumerate(train_dataloader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = batch['polarity'].to(self.opt.device)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            val_acc, val_f1 = self._evaluate_acc_f1(val_dataloader)
            self.logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                max_val_epoch = i_epoch
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_{1}_val_acc_{2}'.format(self.opt.model_name, self.opt.dataset, round(val_acc, 4))
                torch.save(self.model.state_dict(), path)
                self.logger.info('>> saved: {}'.format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
            if i_epoch - max_val_epoch >= self.opt.patience:
                print('>> early stop.')
                break

        return path