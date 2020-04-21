import tensorflow as tf

import util

if util.use_cupy:
    import cupy as np
else:
    import numpy as np
    

import time

cp_array = np.zeros((10, 10))

dl_tensor = cp_array.toDlpack()

seq = tf.keras.Sequential()
seq.add(tf.keras.layers.Dense(1, input_shape=(10,)))
seq.compile()

tf_tensor = tf.experimental.dlpack.from_dlpack(dl_tensor)

print(seq.predict(tf_tensor))

print(tf_tensor.device)

print("done")