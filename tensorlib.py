# Autograd scalar tensor

import sys
import math

operations = {
    "sum": ((lambda a, b: a+b), (lambda a, b, c: (c, c))),
    "sub": ((lambda a, b: a-b), (lambda a, b, c: (c, -c))),
    "mul": ((lambda a, b: a*b), (lambda a, b, c: (b*c, a*c))),
    "div": ((lambda a, b: a/b), (lambda a, b, c: ((1/b)*c, (-a/(b**2))*c))),
    "pow": (
        (lambda a, b: a**b),
        (lambda a, b, c: (
            (b*(a**(b-1)))*c, 
            ((a**b)*math.log(abs(a) if a != 0 else 1e-9))*c
        ))
    ),
}

class Tensor:
    def __init__(self, data: any, requires_grad: bool = False, parents=()):
        self.data = float(data)

        self.requires_grad = requires_grad
        self.grad = 0.0
        
        self.parents = set(parents) if parents else set()
        self._backward = lambda: None

    def config_operation(self, other, operator: str):
        if operator not in operations:
            print("[ERROR] Invalid given operator.")
            sys.exit()

        other = Tensor(other) if not isinstance(other, Tensor) else other
    
        requires_grad = self.requires_grad or other.requires_grad
        operate, derivate = operations[operator]

        out = Tensor(
            data=operate(self.data, other.data),
            requires_grad=requires_grad,
            parents=(self, other),
        )

        def _backward():
            dA, dB = derivate(self.data, other.data, out.grad)
            if self.requires_grad:
                self.grad += dA
            if other.requires_grad:
                other.grad += dB

        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.parents:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Tensor(Data={self.data}, Gradient={self.grad})"
    
    def __add__(self, other): return self.config_operation(other, "sum")
    def __sub__(self, other): return self.config_operation(other, "sub")
    def __mul__(self, other): return self.config_operation(other, "mul")
    def __truediv__(self, other): return self.config_operation(other, "div")
    def __pow__(self, other): return self.config_operation(other, "pow")
