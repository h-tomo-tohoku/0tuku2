import os
import pickle
import numpy as np


class BaseModel:
    def __init__(self) -> None:
        self.params = None, None

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError

    def save_params(self, file_name=None):
        if file_name is None:
            file_name = self.__class__.__name__ + ".pkl"

        params = [p.astype(np.float16) for p in self.params]

        with open(file_name, "wb") as f:
            pickle.dump(params, f)

    def load_params(self, file_name=None):
        if file_name is None:
            file_name = self.__class__.__name__ + ".pkl"

        if "/" in file_name:
            file_name = file_name.replace("/", os.seq)

        if not os.path.exists(file_name):
            raise IOError("No file: " + file_name)

        params = [p.astype("f") for p in self.params]

        for i, param in enumerate(self.params):
            param[...] = params[i]
