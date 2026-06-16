# Usage example

from raw_learn.models import LinearRegression

model = LinearRegression(lr=0.1)

inputs = [1,2,3,4]
outputs = [2,4,6,8]

model.fit(x=inputs, y=outputs, epochs=500)

predict = model.predict(5)

print(f"{predict:.2f}")
