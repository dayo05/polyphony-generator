from keras.utils.np_utils import to_categorical
import numpy as np
from keras.layers import LSTM, Dense, Permute, Activation, Input, Embedding, Concatenate, Reshape, Lambda, RepeatVector, Multiply
from keras.models import Model
import tensorflow as tf
import keras.backend as K
from keras.optimizers import Adam
import pandas as pd
import keras

data = pd.read_csv("train_labels.csv")

n_instrument = 76
n_note = 106
n_end_beat = 1002
n_start_beat = 5000

instrument_input = []
note_input = []
end_beat_input = []
start_beat_input = []

instrument_output = []
note_output = []
end_beat_output = []
start_beat_output = []

sequence_length = 32

instruments = data["instrument"].tolist()
notes = data["note"].tolist()
end_beats = data["end_beat"].tolist()
start_beats = data["start_beat"].tolist()

for i in range(0, 320000, 1):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]
    note_input.append(sequence_in)
    note_output.append(sequence_out)

    sequence_in = instruments[i:i + sequence_length]
    sequence_out = instruments[i + sequence_length]
    instrument_input.append(sequence_in)
    instrument_output.append(sequence_out)

    sequence_in = end_beats[i:i + sequence_length]
    sequence_out = end_beats[i + sequence_length]
    end_beat_input.append(sequence_in)
    end_beat_output.append(sequence_out)

    sequence_in = start_beats[i:i + sequence_length]
    sequence_out = start_beats[i + sequence_length]
    start_beat_input.append(sequence_in)
    start_beat_output.append(sequence_out)

note_output = to_categorical(note_output, num_classes=n_note)
instrument_output = to_categorical(instrument_output, num_classes=n_instrument)
end_beat_output = to_categorical(end_beat_output, num_classes=n_end_beat)
start_beat_output = to_categorical(start_beat_output, num_classes=n_start_beat)

embedding_size = 10

note_model_input = Input(shape=(None,))
instrument_model_input = Input(shape=(None,))
start_beat_model_input = Input(shape=(None,))
end_beat_model_input = Input(shape=(None,))

x_note = Embedding(n_note, embedding_size)(note_model_input)
x_instrument = Embedding(n_instrument, embedding_size)(instrument_model_input)
x_start_beat = Embedding(n_start_beat, embedding_size)(start_beat_model_input)
x_end_beat = Embedding(n_end_beat, embedding_size)(end_beat_model_input)

x = Concatenate()([x_note, x_instrument, x_start_beat, x_end_beat])

x = LSTM(128, return_sequences=True)(x)
x = LSTM(128, return_sequences=True)(x)

e = Dense(1, activation="tanh")(x)
e = Reshape([-1])(e)

AF = Activation("softmax")(e)

c = Permute([2,1])( RepeatVector(128)(AF) )
c = Multiply()([x,c])
c = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(128,))(c)

note_model_output = Dense(n_note, activation='softmax', name='pitch')(c)
instrument_model_output = Dense(n_instrument, activation='softmax', name='instrument')(c)
start_beat_model_output = Dense(n_start_beat, activation='softmax', name='start_beat')(c)
end_beat_model_output = Dense(n_end_beat, activation='softmax', name='end_beat')(c)

model = Model([note_model_input, instrument_model_input, start_beat_model_input, end_beat_model_input], [note_model_output, instrument_model_output, start_beat_model_output, end_beat_model_output])
att_model = Model([note_model_input, instrument_model_input, start_beat_model_input, end_beat_model_input], AF)

opt = Adam(lr = 0.005)
model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], optimizer=opt)

print(model.summary())
tf.keras.utils.plot_model(
    model, to_file='polyphony_model_diagram.png', show_shapes=True)

filepath = "model/{epoch:02d}-{loss:.4f}-bigger.hdf5"

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath, monitor='loss',
    verbose=0,
    save_best_only=True,
    mode='min'
)

note_input = np.array(note_input)
note_output = np.array(note_output)
instrument_input = np.array(instrument_input)
instrument_output = np.array(instrument_output)
start_beat_input = np.array(start_beat_input)
start_beat_output = np.array(start_beat_output)
end_beat_input = np.array(end_beat_input)
end_beat_output = np.array(end_beat_output)

model.fit([note_input, instrument_input, start_beat_input, end_beat_input], [note_output, instrument_output, start_beat_output, end_beat_output], epochs=300, batch_size=10000, callbacks=[checkpoint])
