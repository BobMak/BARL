import unittest
import logging
import os
import shutil
from io import StringIO
import sys
import random as rnd

from skimage.future.graph.setup import base_path

sys.path.append('./')
from DQN import DQN

hparams = {'learning_rate': 0.01, 'batch_size': 32}


class TestSavingWBuffer(unittest.TestCase):
    def setUp(self):
        self.save_buffer = True
        self.rnd_str = f"{rnd.randint(0,65536):04x}"
        self.base_path = self.rnd_str
        self.env = "CartPole-v1"
    def tearDown(self):
        shutil.rmtree(self.base_path)
    def test_loaded(self):
        agent = DQN(self.env, base_path=self.rnd_str, chkp_interval=10, save_buffer=self.save_buffer,
                    save_checkpoints=True, chkp_latest_only=False)
        model_path = f"{self.rnd_str}/model"
        agent.save(model_path)
        loaded_agent = DQN.load(model_path)
        self.assertTrue(loaded_agent is not None)
        self.assertTrue(loaded_agent.env is not None)
        self.assertTrue(loaded_agent.env_id == self.env)
    def test_checkpoint(self):
        agent = DQN(self.env, base_path=self.rnd_str, chkp_interval=10, save_buffer=self.save_buffer, save_checkpoints=True, chkp_latest_only=False)
        model_path = f"{self.rnd_str}/{str(agent)}"
        agent.learn(11)
        chk_path = f"{model_path}_10"
        self.assertTrue(os.path.exists(chk_path))
        checkpoint_agent = DQN.load(chk_path)
        self.assertTrue(checkpoint_agent.tot_env_steps == 10)
    def test_last_checkpoint(self):
        agent = DQN(self.env, base_path=self.rnd_str, chkp_interval=10, save_buffer=self.save_buffer, save_checkpoints=True, chkp_latest_only=True)
        model_path = f"{self.rnd_str}/{str(agent)}"
        agent.learn(11)
        chk_path = f"{model_path}_latest"
        self.assertTrue(os.path.exists(chk_path))
        checkpoint_agent = DQN.load(chk_path)
        self.assertTrue(checkpoint_agent.tot_env_steps == 10)

    def test_modules(self):
        agent = DQN(self.env, base_path=self.rnd_str, chkp_interval=10, save_buffer=self.save_buffer,
                    save_checkpoints=True, chkp_latest_only=True)
        model_path = f"{self.rnd_str}/model"
        agent.learn(10)
        agent.save(model_path)
        loaded_agent = DQN.load(model_path)
        for m1, m2 in zip(loaded_agent.online_qs.modules(), agent.online_qs.modules()):
            if hasattr(m1, 'weight'):
                self.assertTrue((m1.weight == m2.weight).all())

class TestSavingWOBuffer(TestSavingWBuffer):
    def setUp(self):
        super().setUp()
        self.save_buffer = False


if __name__ == '__main__':
    unittest.main()
