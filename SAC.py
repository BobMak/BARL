from dataclasses import dataclass
import torch as th

from BaseAgent import BaseAgent


@dataclass
class SAC(BaseAgent):

    def __init__(self)