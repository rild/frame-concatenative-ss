import kmeans as k

from os.path import join
import numpy as np

def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

# Generate waveform from kmeans class center vector
def main():
    # Target data
    mel_npy_filename = 'jsut-mel-00356.npy'
    label_npy_filename = 'jsut-mel-label-00356.npy'
    filename = "400kmeans_obj.pkl"

    mel = np.load(join('res/jsut', mel_npy_filename))
    label = np.load(join('res/label', label_npy_filename))
    kmeans = k.load_pkl(filename)

    print("mel", mel.shape)
    print("label", label.shape)

    mel_ = np.empty((80,), np.float32)
    for i in range(len(label)):
        mel_ = np.vstack((mel_, kmeans.cluster_centers_[i]))
    mel_ = np.delete(mel_, 0, 0)

    print("compare data structure ----")
    print("mel: ", mel.shape)
    print("mel_: ", mel_.shape)

    print("mel data:", mel)
    print("mel_ data:", mel_)

    print("min-max mel_ data:", min_max(mel_))

if __name__ == '__main__':
    main()
