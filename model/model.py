import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torch import nn
from utils.utils import merge_hp
from model.decoder import Decoder
from model.embedding import Embeddings, PositionalEncoding
from model.labelsmoothing import LabelSmoothing
from datasets.dataloader import create_dataloader

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

    def forward(self, x, mask):
        embed = self.embed(x)
        output = self.proj(self.decoder(embed, embed, mask))
        return output

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        res = self.forward(x, mask)
        loss = self.criterion(res, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        res = self.forward(x, mask)
        loss = self.criterion(res, y)
        resmax = torch.argmax(res, dim=-1)
        conf = F.softmax(res, dim=-1).max(dim=-1)[0]
        acc = (resmax == y).float().mean()
        return {'val_loss': loss, 'val_acc': acc, 'val_confidence': conf}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        conf = torch.stack([x['val_confidence'] for x in outputs])
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        self.logger.experiment.add_histogram('val_confidence', conf, self.global_step)
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y, mask = batch
        res = self.forward(x, mask)
        loss = self.criterion(res, y)
        resmax = torch.argmax(res, dim=-1)
        conf = F.softmax(res, dim=-1).max(dim=-1)[0]
        acc = (resmax == y).float().mean()
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

    @pl.data_loader
    def train_dataloader(self):
        return create_dataloader(self.hp, self.args, "train")

    @pl.data_loader
    def val_dataloader(self):
        return create_dataloader(self.hp, self.args, "val")

    @pl.data_loader
    def test_dataloader(self):
        return create_dataloader(self.hp, self.args, "test")
