import yaml
import torch
import argparse
import platform

from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger

from utils.hparams import HParam
from model.model import Reformer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="name of the model for logging, saving checkpoint")
    parser.add_argument('-b', '--batch_size', type=int, required=True,
                        help="batch size to be used")
    parser.add_argument('-f', '--fast_dev_run', type=bool, required=False, default=False,
                        help="enable fast dev run for debugging purposes")
    parser.add_argument('-v', '--version', type=int, required=False, default=None,
                        help="version to resume checkpoint from, default new version")
    args = parser.parse_args()
    
    hp = HParam(args.config)
    with open(args.config, 'r') as f:
        hp_str = ''.join(f.readlines())
    if platform.system() == 'Windows':
        hp.train.num_workers = 0

    reformer = Reformer(hp, args)

    logger = TestTubeLogger(
        save_dir=hp.log.path,
        name=args.name,
        version=args.version,
    )

    trainer = Trainer(
        logger=logger,
        default_save_path=hp.log.path,
        # distributed_backend='ddp',
        fast_dev_run=args.fast_dev_run,
        gpus=1,
        accumulate_grad_batches=hp.train.accumulate,
        min_nb_epochs=hp.train.epochs,
        max_nb_epochs=hp.train.epochs,
        weights_summary='full'
    )

    trainer.fit(reformer)

    trainer.test(reformer)
