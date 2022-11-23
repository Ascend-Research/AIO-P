from abc import ABC, abstractmethod

"""
Overhead file for task managers
Responsible for taking a CG-based model with modified head, e.g., from model_src/multitask/adapt_cg_framework.py
and training it/evaluating it on the appropriate data, etc.
"""

class BaseTaskManager(ABC):

    def __init__(self, log_f=print):

        self.model, self.best_model = None, None
        self.log_f = log_f
        self.train_dict, self.test_dict = {}, {}
        self.train_metric, self.test_metric = "train_acc", "test_acc"

    @abstractmethod
    def set_model(self, model):
        self.model = model

    @abstractmethod
    def train(self, eval_test=False):
        self.train_dict = {self.train_metric: -1}
        if eval_test:
            self.test_dict[self.test_metric] = self.eval()
        return {**self.train_dict, **self.test_dict}

    @abstractmethod
    def eval(self):
        return -1

    def get_best_model(self):
        if self.best_model is None:
            self.log_f("Model not trained/no best")
            return None
        return self.best_model
