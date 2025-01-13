import os

logger_types = {'wandb', 'std', 'tensorboard'}

class BaseLogger: 
    def log_hparams(self, hparam_dict):
        raise NotImplementedError()
    def log_history(self, param, value, step):
        raise NotImplementedError()
    def log_video(self, video_path):
        raise NotImplementedError()
    def log_image(self, image_path):
        raise NotImplementedError()
    def __repr__(self):
        return f"{self.__class__.__name__}:{self.args}"
    def make_logger(self, lg_cls, lg_args):
        return str_to_cls[lg_cls](*lg_args)
