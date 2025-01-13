import wandb
import loggers.BaseLogger as BaseLogger


class WandBLogger(BaseLogger):
    def __init__(self, entity, project):
        self.args = (entity, project)
        wandb.init(entity=entity, project=project)
    def log_hparams(self, hparam_dict):
        for param, value in hparam_dict.items():
            # check if not serializable:
            try:
                wandb.log({param: value})
            except Exception as e:
                print(f"Could not log {param}: {value}")
    def log_history(self, param, value, step):
        wandb.log({param: value}, step=step)
    def log_video(self, video_path, name="video"):
        wandb.log({name: wandb.Video(video_path)})
    def log_image(self, image_path, name="image"):
        wandb.log({name: wandb.Image(image_path)})