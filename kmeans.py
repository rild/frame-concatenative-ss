import pickle

from sklearn.cluster import KMeans
import numpy as np

from nnmnkwii.datasets import FileSourceDataset, FileDataSource

def loader(filename):
    with open(filename, 'rb') as f:
        filename
    return obj

def stft_data_loader(ffile, tfile, Zxxfile):
    f = loader(ffile)
    t = loader(tfile)
    Zxx = loader(Zxxfile)
    return f, t, Zxx

def generate_cluster_center_wav():
    first = True
    for i in labels:
        if first:
            centers = kmeans.cluster_centers_[labels[0]]  # TO FIX
            first = False
        else:
            centers = np.vstack((centers, kmeans.cluster_centers_[i]))
    print(centers.shape)

    ZxxC = centers.T
    print(ZxxC.shape)

    with open(rpath + 'generated' + 'ZxxC_kmpp_30.pickle', 'wb') as file:
        pickle.dump(ZxxC, file)

    # with open('hanekawa_nandemoha01' + '_t_km.pickle', 'wb') as file:
    #     pickle.dump(t_km, file)

def clustering():
    fname = 'hanekawa_nandemoha01'

    rpath = 'res/'
    t_filename = rpath + fname + '_t.pickle'
    f_filename = rpath + fname + '_f.pickle'
    Zxx_filename = rpath + fname + '_Zxx.pickle'


    f, t, Zxx = stft_data_loader(f_filename,
                                 t_filename,
                                 Zxx_filename)

    ZxxT = Zxx.T


    kmeans = KMeans(n_clusters=n_clusters,
        init='k-means++',
        n_init=10,
        max_iter=300,
        tol=1e-04,
        random_state=0).fit(ZxxT)
    #  t_km = km.fit_predict(ZxxT)

    labels = kmeans.labels_
    # print(labels)
    # print(kmeans.cluster_centers_[labels[0]])

class _NPYDataSource(FileDataSource):
    def __init__(self, data_root, col):
        self.data_root = data_root
        self.col = col

    def collect_files(self):
        meta = join(self.data_root, "train.txt")
        with open(meta, "rb") as f:
            lines = f.readlines()
        lines = list(map(lambda l: l.decode("utf-8").split("|")[self.col], lines))
        paths = list(map(lambda f: join(self.data_root, f), lines))
        return paths

    def collect_features(self, path):
        return np.load(path)

class MelSpecDataSource(_NPYDataSource):
    def __init__(self, data_root):
        super(MelSpecDataSource, self).__init__(data_root, 1)


class LinearSpecDataSource(_NPYDataSource):
    def __init__(self, data_root):
        super(LinearSpecDataSource, self).__init__(data_root, 0)

if __name__ == "__main__":
    n_clusters = 400

    data_root = "./res/jsut"
    print("target: ", data_root)


'''

http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
Examples

'''
