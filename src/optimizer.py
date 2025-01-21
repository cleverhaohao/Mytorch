from task1_operators import *


class optimizer():
    def __init__(self, parameters: list[Tensor], lr: float):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        pass

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = None

class SGD(optimizer):
    def step(self):
        for parameter in self.parameters:
            parameter.data = parameter.data - self.lr * parameter.grad


