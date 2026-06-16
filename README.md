# rawlearn
This is a recent started project about machine learning. The goal of this project is to improve ML knowledge.

I did this for practice and get deeper in the ML fundamentals.

The project contains:

- Scalar tensors: Holds quantitative data and accumulates derivative gradients of a specific operation.
- SGD: Optimizer used to train the linear regression model.
- Linear regression: uses the scalar tensors to execute forward passes and backpropagation.

Usage example:

# Python

from raw_learn.models import LinearRegression

- initialize model with 0.1 step size
model = LinearRegression(lr=0.1)

- training data
inputs = [1,2,3,4]
outputs = [2,4,6,8]

- model training
model.fit(x=inputs, y=outputs, epochs=500)

predict = model.predict(5)

- show prediction for 5 (expected: 10)
print(f"{predict:.2f}")
