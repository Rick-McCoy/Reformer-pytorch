'''
Implements Reformer
'''
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torch import nn
from tqdm import tqdm
from pytorch_lightning.overrides.override_data_parallel import LightningDistributedDataParallel
from utils.utils import merge_hp, top_p_sample, save_to_midi
from model.decoder import Decoder
from model.embedding import Embeddings, PositionalEncoding
from model.labelsmoothing import LabelSmoothing
from datasets.dataloader import Dataloaders

class Reformer(pl.LightningModule):
    def __init__(self, hp, args):
        super(Reformer, self).__init__()
        self.decoder = Decoder(hp, args)
        self.positional_encoding = PositionalEncoding(hp, args)
        self.embed = nn.ModuleList([
            nn.Sequential(
                Embeddings(vocab, hp.model.d_model), self.positional_encoding
            ) for vocab in hp.data.vocab
        ])
        self.proj = nn.ModuleList([
            nn.Linear(hp.model.d_model, vocab, bias=False) for vocab in hp.data.vocab
        ])
        for embed, proj in zip(self.embed, self.proj):
            embed[0].embed.weight = proj.weight
        self.linear = nn.Linear(hp.model.d_model * len(hp.data.vocab), hp.model.d_model)
        self.criterion = nn.ModuleList(
            LabelSmoothing(hp.train.smoothing, vocab, 1) for vocab in hp.data.vocab
        )
        self.hp = hp
        self.args = args
        self.hparams = merge_hp(hp, args)
        self.dataloaders = Dataloaders(hp, args)

    def forward(self, x: torch.Tensor):
        embedding = torch.cat(
            [embed(x[..., i]) for embed, i in zip(self.embed, range(x.size(-1)))], dim=-1
        )
        linear = self.linear(embedding)
        decode = self.decoder(linear, linear)
        output = [proj(decode) for proj in self.proj]
        return output

    def training_step(self, batch, batch_idx):
        source, target, mask = batch
        res = self.forward(source)
        loss = [
            crit(r, target[..., i], mask) for crit, r, i\
                in zip(self.criterion, res, range(target.size(-1)))
        ]
        tensorboard_logs = {
            'total_train_loss': sum(loss),
        }
        tensorboard_logs.update({'train_loss_{}'.format(i): l for i, l in enumerate(loss)})
        return {'loss': sum(loss), 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        source, target, mask = batch
        res = self.forward(source)
        loss = [
            crit(r, target[..., i], mask) for crit, r, i\
                in zip(self.criterion, res, range(target.size(-1)))
        ]
        prob = [F.softmax(r, dim=-1) for r in res]
        conf, resmax = zip(*[torch.max(p, dim=-1) for p in prob])
        acc = [
            (rm == target[..., i]).flatten().masked_select(mask.flatten()).float().mean()\
                for rm, i in zip(resmax, range(target.size(-1)))
        ]
        if batch_idx == 0 and self.hp.data.dataset == "music":
            self.sample_from_prob([p[0] for p in prob])
        return {
            'val_loss': torch.stack(loss),
            'val_acc': torch.stack(acc),
            'val_confidence': torch.stack(conf)
        }

    def validation_end(self, outputs):
        loss = torch.stack([x['val_loss'] for x in outputs])
        acc = torch.stack([x['val_acc'] for x in outputs])
        conf = torch.stack([x['val_confidence'] for x in outputs])
        tensorboard_logs = {'avg_val_loss': loss.sum(dim=-1).mean(), 'avg_val_acc': acc.mean()}
        tensorboard_logs.update({'val_loss_{}'.format(i): l.mean() for i, l in enumerate(loss.t_())})
        tensorboard_logs.update({'val_acc_{}'.format(i): a.mean() for i, a in enumerate(acc.t_())})
        self.logger.experiment.add_histogram('val_confidence', conf, self.global_step)
        return {'val_loss': loss.sum(dim=-1).mean(), 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        source, target, mask = batch
        res = self.forward(source)
        loss = [
            crit(r, target[..., i], mask) for crit, r, i\
                in zip(self.criterion, res, range(target.size(-1)))
        ]
        prob = [F.softmax(r, dim=-1) for r in res]
        conf, resmax = zip(*[torch.max(p, dim=-1) for p in prob])
        acc = [
            (rm == target[..., i]).flatten().masked_select(mask.flatten()).float().mean()\
                for rm, i in zip(resmax, range(target.size(-1)))
        ]
        if batch_idx == 0 and self.hp.data.dataset == "music":
            self.autoregressive_sample(source)
        return {
            'test_loss': torch.stack(loss),
            'test_acc': torch.stack(acc),
            'test_confidence': torch.stack(conf)
        }

    def test_end(self, outputs):
        loss = torch.stack([x['test_loss'] for x in outputs])
        acc = torch.stack([x['test_acc'] for x in outputs])
        conf = torch.stack([x['test_confidence'] for x in outputs])
        tensorboard_logs = {'avg_test_loss': loss.sum(dim=-1).mean(), 'avg_test_acc': acc.mean()}
        tensorboard_logs.update({'test_loss_{}'.format(i): l.mean() for i, l in enumerate(loss.t_())})
        tensorboard_logs.update({'test_acc_{}'.format(i): a.mean() for i, a in enumerate(acc.t_())})
        self.logger.experiment.add_histogram('test_confidence', conf, self.global_step)
        return {'test_loss': loss.sum(dim=-1).mean(), 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hp.train.lr, amsgrad=True)

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

    def sample_from_prob(self, prob: torch.Tensor):
        sample = torch.cat([
            top_p_sample(prob=p) for p in prob
        ], dim=-1).cpu().numpy()
        save_to_midi(sample)

    def autoregressive_sample(self, primer: torch.Tensor, length=512):
        bucket_length = self.hp.model.bucket_length
        gen = primer[:, :bucket_length * 2]
        padding = torch.cat([
            gen.new_full((1, bucket_length * 2, 1), fill_value=vocab - 1)\
                for vocab in self.hp.data.vocab
        ], dim=-1)
        for _ in tqdm(range(length // (bucket_length * 2) - 1)):
            gen = torch.cat([gen, padding], dim=1)
            for j in tqdm(range(bucket_length * 2)):
                output = self.forward(gen)
                prob = [F.softmax(o[0, j - bucket_length - 1], dim=-1) for o in output]
                sample = torch.cat([top_p_sample(p) for p in prob], dim=-1)
                gen[0, j - bucket_length] = sample
        save_to_midi(gen[0].cpu().numpy())
