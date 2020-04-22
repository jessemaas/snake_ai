import util
util.use_cupy = False

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

if util.use_cupy:
    import cupy as np
else:
    import numpy as np

total_time = util.start_timer()

model = models.Sequential()
model.add(layers.Conv2D(8, (5, 5), input_shape=(13, 13, 5)))
model.add(layers.PReLU())
model.add(layers.Conv2D(6, (3, 3)))
model.add(layers.PReLU())
model.add(layers.Conv2D(6, (3, 3)))
model.add(layers.PReLU())
model.add(layers.Conv2D(6, (3, 3)))
model.add(layers.Flatten())
model.add(layers.PReLU())
model.add(layers.Dense(32))
model.add(layers.PReLU())
model.add(layers.Dense(1, activation="sigmoid"))

optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

factor = 256

samples = factor

input = np.random.random_sample((samples, 13, 13, 5))
output = np.random.random_sample((samples,))

predictions = model.predict(util.as_tensor(input))
model.fit(util.as_tensor(input), util.as_tensor(output))

predict = util.start_timer()
for epoch in range(1024 * 16 // factor):
    if epoch % 10 == 0:
        print(epoch)
    predictions = model.predict(util.as_tensor(input))
    x = predictions[0]
util.end_timer(predict, 'predict')

train = util.start_timer()
for epoch in range(1024 * 16 // factor):
    if epoch % 10 == 0:
        print(epoch)
    model.fit(util.as_tensor(input), util.as_tensor(output))

util.end_timer(train, 'train')
util.end_timer(total_time, 'total_time')

print(x)

util.print_timers()
print('using cupy:', util.use_cupy)