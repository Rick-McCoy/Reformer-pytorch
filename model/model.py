'''
Implements Reformer
'''
import os
import time
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torch import nn
from pytorch_lightning.overrides.override_data_parallel import LightningDistributedDataParallel
from utils.utils import merge_hp
from model.decoder import Decoder
from model.embedding import Embeddings, PositionalEncoding
from model.labelsmoothing import LabelSmoothing
from datasets.dataloader import Dataloaders
from datasets.music import roll_to_midi

class Reformer(pl.LightningModule):
    def __init__(self, hp, args):
        super(Reformer, self).__init__()
        self.decoder = Decoder(hp, args)
        self.embed = nn.Sequential(
            Embeddings(hp, args), PositionalEncoding(hp, args)
        )
        self.proj = nn.Linear(hp.model.d_model, hp.data.vocab)
        self.criterion = LabelSmoothing(hp.train.smoothing)
        self.hp = hp
        self.args = args
        self.hparams = merge_hp(hp, args)
        self.dataloaders = Dataloaders(hp, args)

    def forward(self, x):
        embed = self.embed(x)
        output = self.proj(self.decoder(embed, embed))
        return output

    def training_step(self, batch, batch_idx):
        source, target, _ = batch
        res = self.forward(source)
        loss = self.criterion(res, target)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        source, target, accuracy_mask = batch
        res = self.forward(source)
        loss = self.criterion(res, target)
        resmax = torch.argmax(res, dim=-1)
        conf = F.softmax(res, dim=-1).max(dim=-1)[0]
        acc = (resmax == target).flatten().masked_select(accuracy_mask.flatten()).float().mean()
        if batch_idx == 0 and self.hp.data.dataset == "music":
            try:
                song = roll_to_midi(resmax[0].cpu().numpy())
                song.write(os.path.join('samples', str(int(time.time())) + '.mid'))
            except AssertionError as error:
                print(error)
                print('Failed to generate sample')
        return {'val_loss': loss, 'val_acc': acc, 'val_confidence': conf}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        conf = torch.stack([x['val_confidence'] for x in outputs])
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        self.logger.experiment.add_histogram('val_confidence', conf, self.global_step)
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        source, target, accuracy_mask = batch
        res = self.forward(source)
        loss = self.criterion(res, target)
        resmax = torch.argmax(res, dim=-1)
        conf = F.softmax(res, dim=-1).max(dim=-1)[0]
        acc = (resmax == target).flatten().masked_select(accuracy_mask.flatten()).float().mean()
        if batch_idx == 0 and self.hp.data.dataset == "music":
            try:
                song = roll_to_midi(resmax[0].cpu().numpy())
                song.write(os.path.join('samples', str(int(time.time())) + '.mid'))
            except AssertionError as error:
                print(error)
                print('Failed to generate sample')
        return {'test_loss': loss, 'test_acc': acc, 'test_confidence': conf}

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        conf = torch.stack([x['test_confidence'] for x in outputs])
        self.logger.experiment.add_histogram('test_confidence', conf, self.global_step)
        tensorboard_logs = {'test_loss': avg_loss, 'test_acc': avg_acc}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hp.train.lr)

    def configure_ddp(self, model, device_ids):
        return LightningDistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=False
        )

    @pl.data_loader
    def train_dataloader(self):
        return self.dataloaders.get_dataloader("train")

    @pl.data_loader
    def val_dataloader(self):
        return self.dataloaders.get_dataloader("val")

    @pl.data_loader
    def test_dataloader(self):
        return self.dataloaders.get_dataloader("test")
