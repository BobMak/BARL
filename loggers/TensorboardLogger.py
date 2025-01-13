from torch.utils.tensorboard import SummaryWriter
import loggers.BaseLogger as BaseLogger

class TensorboardLogger(BaseLogger):
    def __init__(self, log_dir, id=None):
        self.args = (log_dir,id)
        # Check for existence of log_dir:
        # get the length of folders with same name:
        folder_name = log_dir
        if id is None:
            i = 1
            while os.path.exists(folder_name):
                folder_name = f"{log_dir}_{i}"
                i += 1
        else:
            folder_name = f"{log_dir}_{id}"
        log_dir = folder_name
        self.writer = SummaryWriter(log_dir)
    def log_hparams(self, hparam_dict):
        for param, value in hparam_dict.items():
            self.writer.add_text(param, str(value), global_step=0)
    def log_history(self, param, value, step):
        self.writer.add_scalar(param, value, global_step=step)
    def log_video(self, video_path, name="video"):
        self.writer.add_video(name, video_path)
    def log_image(self, image_path, name="image"):
        self.writer.add_image(name, image_path)