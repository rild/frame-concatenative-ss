from seq2seq.models import Seq2Seq
from seq2seq.models import SimpleSeq2Seq

from sklearn import model_selection

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

from tqdm import tqdm

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import utils
import text_enc as frontend


# Hyper params
batch_size = 16  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.

# Preprocess input data
#     text -> roman texts one-hot vector
#     speech -> cluster label one-hot vector

# Get raw text data
lines, _, _ = utils.load_jsut_spec_data()
# raw_text = ''.join(lines)
# lines = raw_text.split('\n')[:-1]
lines = ''.join(lines).split('\n')[:-1]
# [-1]: to remove ''; ...いた。', '']
#     if not, len(lines) will be 7667

# Change the lines to romas each
# raw_text = utils.raw_to_roma(raw_text)
# input_texts = []
# for i in range(len(lines)):
#     input_texts.append(utils.raw_to_roma(lines[i]))
##### there are many unidentified characters

# Convert raw text to vector
input_text_seqs = list(map(lambda l: frontend.text_to_sequence(l, p=1), lines))
# TOFIX: There is a paper insists that
#     it is good to put randomness on converting text to sequence.


# Load label data
# We can get waveform with kmeans model from label sequence
output_label_seqs, _ = utils.load_label_data()

# Label data contains only class index
# now, the index in range(0, 400), 12/6
# Text sequence is from RAW text, so it contains so many characters.
# That's why input_text_seqs dim is 1
target_label_characters = set()
for label in output_label_seqs:
    for index in label:
        if index not in target_label_characters:
            target_label_characters.add(index)

target_label_characters = sorted(list(target_label_characters))
num_decoder_tokens = len(target_label_characters) 
# >>> 400 # kmeans class num
max_encoder_seq_length = max([len(e) for e in input_text_seqs])
# >>> 135
max_decoder_seq_length = max([len(e) for e in output_label_seqs])
# >>> 1293

target_token_index = dict(
   [(char, i) for i, char in enumerate(target_label_characters)])


num_encoder_tokens = 1
# the reason of dim = 1, is explained before


print("preprocess: make encoder/decoder input, target data array")
encoder_input_data = np.zeros(
    (len(input_text_seqs), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
decoder_input_data = np.zeros(
    (len(input_text_seqs), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
decoder_target_data = np.zeros(
    (len(input_text_seqs), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')

# Make x, y for the mode
print("text sequence")
for i, text_seq in tqdm(enumerate(input_text_seqs)):
    for t, char in enumerate(text_seq):
        encoder_input_data[i, t, 0] = char

## Question: what is the difference between dec-inpu and dec-target?
print("label sequence")
for i, label_seq in tqdm(enumerate(output_label_seqs)):
    for t, char in enumerate(label_seq):
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1


# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)


# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
                    epochs=epochs,
                              validation_split=0.2)
# Save model
model.save('s2s.h5')


