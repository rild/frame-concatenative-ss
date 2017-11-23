import kmeans as k

from os.path import join
import numpy as np


# Generate waveform from kmeans class center vector
def main():
    # Target data
    mel_npy_filename = 'jsut-mel-00356.npy'
    label_npy_filename = 'jsut-mel-label-00356.npy'
    filename = "400kmeans_obj.pkl"

    mel = np.load(join('res/jsut', mel_npy_filename))
    label = np.load(join('res/label', label_npy_filename))
    kmeans = k.load_pkl(filename)

if __name == '__main__':
    main()
