import utils as k
import audio
from hparams import hparams 
import lws
from os.path import join
import numpy as np

def print_each_var(array, sort=False):
    length = len(array)
    varList = np.zeros([length, 2]) # [var, index]
    for i in range(length):
        varList[i,0] = np.var(array[i])
        varList[i,1] = i
    return varList

def reduce_dim_by_var(spec):
    # spec: ... ,513
    # specT: 513,...
    specT = spec.T
    specTVar = np.zeros(len(specT)) # 513
    for i in range(len(specT)):
        specTVar[i] = np.var(specT[i])

    # sortedSpecTVar = np.sort(specTVar)

    _specT = specT
    index = np.argsort(specTVar)
    for i in range(len(spec)):
       _specT[index < 100, i] = 0
    return _specT.T

def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

def _lws_processor():
    return lws.lws(hparams.fft_size, hparams.hop_size, mode="speech")

def inv_spec_from_label(label_seq):
    filename = "120_kmeans_obj.pkl"
    kmeans = k.load_pkl(filename)
    ## frequency vector, dim = 513
    spec_ = np.empty((513,), np.float32)
    for i in range(len(label_seq)):
        spec_ = np.vstack((spec_, kmeans.cluster_centers_[label_seq[i]]))
    spec_ = np.delete(spec_, 0, 0)

    return spec_

def save_waveform_from_spec(spectrogram, filename):
    waveform = audio.inv_spectrogram(spectrogram)
    audio.save_wav(waveform, filename)

def load_test_data():
    mel_npy_filename = 'jsut-mel-00356.npy'
    spec_npy_filename = 'jsut-spec-00356.npy'
    label_npy_filename = 'jsut-spec-label-00356.npy'

    spec = np.load(join('res/jsut', spec_npy_filename))
    label = np.load(join('res/label_120', label_npy_filename))

    return spec, label

# Generate waveform from kmeans class center vector
def main():
    # Target data
    filename = "120_kmeans_obj.pkl"

    kmeans = k.load_pkl(filename)
    spec, label = load_test_data()

    print("spec", spec.shape)
    print("label", label.shape)

    spec_ = np.empty((513,), np.float32)
    for i in range(len(label)):
        spec_ = np.vstack((spec_, kmeans.cluster_centers_[label[i]]))
    spec_ = np.delete(spec_, 0, 0)

    print("compare data structure ----")
    print("spec: ", spec.shape)
    print("spec_: ", spec_.shape)

    print("spec data:", spec)
    print("spec_ data:", spec_)

    print("min-max spce_ data:", min_max(spec_))

    waveform = audio.inv_spectrogram(spec)
    waveform_ = audio.inv_spectrogram(spec_)
    waveformmm_ = audio.inv_spectrogram(min_max(spec_))

    audio.save_wav(waveform,'ideal_out.wav')
    audio.save_wav(waveform_, 'idela_out_.wav')
    audio.save_wav(waveformmm_, 'idelal_outmm_.wav')

if __name__ == '__main__':
    main()
