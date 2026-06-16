# Linear regression model

from raw_learn.libs.tensorlib import Tensor
from raw_learn.optimizers import SGD

import sys
import random as rd

class LinearRegression:
    def __init__(self, lr: float=0.1):
        self.w = Tensor(rd.randint(1, 10), requires_grad=True)
        self.b = Tensor(rd.randint(1, 10), requires_grad=True)

        self.lr = lr

    def forward(self, x: float | Tensor):
        x = Tensor(x) if not isinstance(x, Tensor) else x
        return x*self.w+self.b

    def fit(self, x: list, y: list, epochs: int):
        n_samples = len(x)
        if n_samples <= 0 or len(y) <= 0:
            print("[ERROR] Invalid training data.")
            sys.exit()
        
        optimizer = SGD([self.w, self.b], lr=self.lr)
        tensor_n_samples = Tensor(n_samples)
        
        for i in range(epochs):
            optimizer.zero_grad()
            total_loss = Tensor(0.0)
            
            for X, Y in zip(x, y):
                pred = self.forward(X) 
                error = pred-Y

                loss = error**2
                total_loss += loss
            
            mse_loss = total_loss/tensor_n_samples
            mse_loss.backward()

            optimizer.step()

    def predict(self, x: float | Tensor):
        z = self.forward(x)
        return z.data
