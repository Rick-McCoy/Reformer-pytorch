from torch.utils.tensorboard import SummaryWriter

from . import plotting as plt 


class MyWriter(SummaryWriter):
    def __init__(self, hp, logdir):
        super(MyWriter, self).__init__(logdir)
        self.hp = hp

    def log_training(self, train_loss, step):
        raise NotImplementedError

    def log_validation(self, step):
        raise NotImplementedError

    def log_sample(self, step):
        raise NotImplementedError
