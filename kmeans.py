import pickle

from tqdm import tqdm
from os.path import dirname, join
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB

import numpy as np

from nnmnkwii.datasets import FileSourceDataset, FileDataSource

def get_path_jsut_mel():
    data_root = "./res/jsut"
    print("target: ", data_root)
    meta = join(data_root, "train.txt")
    with open(meta, "rb") as f:
        lines = f.readlines()
    # col_mel = 1
    col_sec = 0
    lines = list(map(lambda l: l.decode("utf-8").split("|")[col_sec], lines))
    paths = list(map(lambda f: join(data_root, f), lines))
    return lines, paths

def create_target_dataset():
    lines, paths = get_path_jsut_mel()
    # print(paths)
    dataset = np.empty((0, 80), np.float32)
    for i in tqdm(range(len(paths))):
            dataset = np.append(dataset, np.load(paths[i]), axis=0)
    print(dataset.shape)
    return dataset

def save_nparray(array):
    np.save('dataset.npy', array)

def save_as_pkl(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_pkl(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

def save_mel_label(out_dir, label, index):
    np_label = np.array(label)
    label_filename = 'jsut-mel-label-%05d.npy' % (index + 1)
    np.save(join(out_dir, label_filename), np_label)
    return label_filename

def add_text(target_text, text):
    return target_text + text + '\n'

if __name__ == "__main__":
    n_clusters = 400
    filename = str(n_clusters) + "_kmeans_obj.pkl"

    # Sum dataset to an array
    data = create_target_dataset()
    save_nparray(data)

    # Classify the array to n classes
    data = np.load('dataset.npy')
    dataT = data.T

    kmeans = KMeans(n_clusters=n_clusters,
       init='k-means++',
       n_init=10,
       max_iter=300,
       tol=1e-04,
       random_state=0).fit(data) # 転置いる？ 11/22

    save_as_pkl(kmeans, filename)

    # Create label array
    # kmeans = load_pkl(filename)

    # _, paths = get_path_jsut_mel()
    # label_ = kmeans.predict(np.load(paths[0]))


    # out_dir = 'res/label/'
    # label_files_text = ''
    # for i in tqdm(range(len(paths))):
    #         label_ = kmeans.predict(np.load(paths[i]))
    #         label_filename = save_mel_label(out_dir, label_, i)
    #         label_files_text = add_text(label_files_text, label_filename)

    # with open(join(out_dir, 'labels.txt'), 'w') as f:
    #     f.write(label_files_text)
'''

http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
Examples

'''
