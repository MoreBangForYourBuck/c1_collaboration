from tensorflow import keras
from helpers.preprocessing import read_all_data



model = keras.models.Sequential([
    keras.layers.LSTM(128, batch_input_shape=(None, 32, 6), return_sequences=True, input_shape=[None, 1]),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(128, return_sequences=False),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(4, activation='softmax')
])
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

data_dict = read_all_data()
imu = data_dict['imu'].to_numpy()
ann = data_dict['ann'].to_numpy().flatten()
del data_dict # Remove to free memory

model.fit(imu, ann, epochs=20, batch_size=64)