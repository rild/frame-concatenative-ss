import kmeans as k
import audio
from hparams import hparams 
import lws
from os.path import join
import numpy as np

def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

def _lws_processor():
    return lws.lws(hparams.fft_size, hparams.hop_size, mode="speech")

# Generate waveform from kmeans class center vector
def main():
    # Target data
    mel_npy_filename = 'jsut-mel-00356.npy'
    spec_npy_filename = 'jsut-spec-00356.npy'
    label_npy_filename = 'jsut-spec-label-00356.npy'
    filename = "400_kmeans_obj.pkl"

    spec = np.load(join('res/jsut', spec_npy_filename))
    label = np.load(join('res/label', label_npy_filename))
    kmeans = k.load_pkl(filename)



    print("spec", spec.shape)
    print("label", label.shape)

    spec_ = np.empty((513,), np.float32)
    for i in range(len(label)):
        spec_ = np.vstack((spec_, kmeans.cluster_centers_[i]))
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

    audio.save_wav(waveform,'out.wav')
    audio.save_wav(waveform, 'out_.wav')
    audio.save_wav(waveformmm_, 'outmm_.wav')

if __name__ == '__main__':
    main()
