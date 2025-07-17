from keras.models import Sequential
from keras.layers import LSTM, Dense

def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape))
    model.add(Dense(1, activation='sigmoid'))  # Change to match your task
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
