# SGD optimizer code

class SGD:
    def __init__(self, params, lr: float = 0.1):
        self.lr = lr
        self.params = list(params)

    def step(self):
        for param in self.params:
            param.data -= param.grad*self.lr

    def zero_grad(self):
        for param in self.params:
            param.grad = 0.0
            param.parents = set()
