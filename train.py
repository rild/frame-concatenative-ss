from seq2seq.models import Seq2Seq
from seq2seq.models import SimpleSeq2Seq

from sklearn import model_selection

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

from tqdm import tqdm

from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import ModelCheckpoint
import utils
import text_enc as frontend
import test_gen as generator

# Hyper params
batch_size = 32  # Batch size for training.
epochs = 1000  # Number of epochs to train for.
latent_dim = 64  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
modelname = "model_epoch%04d-latentdim%03d.h5", (epochs, latent_dim)

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
## OR load the model
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
check = ModelCheckpoint('model.hdf5')
history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              callbacks=[check])

utils.save_as_pkl(history, 'history,pkl')
# Save model
# model.save(modelname)
model.save('model_120class_s2s.h5')
## loat the pretrained model
# load_model('s2s.h5')

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    # target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

def decode_text_seq(input_seq):
    state_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, num_decoder_tokens)) ## 1, 1, 400
    ## expressing which label the vector is
    ## the output label sequence length
    ## : max_decoder_seq_length >>> 1293

    decoded_label_seq = []
    for i in range(max_decoder_seq_length):
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + state_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        
        ## predict the next label index
        decoded_label_seq.append(int(sampled_char))
        
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        
        states_value = [h, c]
        
    return decoded_label_seq

for seq_index in range(100):
    # Take one sequence (part of the training test)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_text_seq(input_seq)
    print('-')
    print('Input squence:', input_seq)
    print('Decoded sequence:', decoded_sentence)

    filename = 'out%03d.wav' % seq_index
    spectrogram = generator.inv_spec_from_label(decoded_sentence)
    generator.save_waveform_from_spec(spectrogram, filename)

