from __future__ import print_function

from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense
import numpy as np

model = load_model('s2s.h5')
