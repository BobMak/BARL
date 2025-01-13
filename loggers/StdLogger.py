import logging
from functools import lru_cache
import loggers.BaseLogger

class StdLogger(BaseLogger):
    def __init__(self, logger=None):
        self.args = (logger,)
        if logger is not None:
            self.log = logger
        else:
            self.log = logging.getLogger("Barl")
            self.log.setLevel(logging.INFO)
            st = logging.StreamHandler()
            st.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s:%(levelname)s: %(message)s')
            st.setFormatter(formatter)
            self.log.addHandler(st)
            # self.log.setFormatter(formatter)
    def log_hparams(self, hparam_dict):
        for param, value in hparam_dict.items():
            # self.log.info(param, value)
            self.log.info(f"{param}: {value}")
    def log_history(self, param, value, step):
        self.log.info(f"{param}: {value}")
    @lru_cache(None)
    def log_video(self, *args, **kwargs):
        self.log.warning("videos are not logged by std logger")