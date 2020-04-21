from util import *
import igraph
from sklearn.preprocessing import normalize
import  sklearn.metrics.pairwise as pair
import scipy.sparse as sp
import numpy as np
#from rolemf.predict import *
from rolemf import *
from scipy.sparse.linalg import svds
from numpy.linalg import svd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import scipy.io
import scipy.sparse as sp
from sklearn.utils.validation import check_symmetric
from scipy.sparse.linalg import eigsh
import os
def get_row_similarity(h,rolenum=10):
    if(isinstance(h,np.matrix)):
        h=h.A
    s = h.dot(h.T)
    #sim = s / norm(h, axis=1).reshape(1, -1) / norm(h, axis=1).reshape(-1, 1)
    sim=pair.cosine_similarity(h)
    # for threshhold  in range(70,100,2):
    #     threshhold=threshhold/100.0
    #     print("similarity larger than {} numbers".format(threshhold),np.sum(sim>threshhold))

    np.save("./intermedia/similarity_for_wiki"+str(rolenum),sim)
    #plt.hist(sim.flatten())
    #plt.show()
def get_and_save_recursive_feature():
    g = nx.read_edgelist("/home/xuyou/GER/data/bc_edgelist.txt")
    A = nx.adjacency_matrix(g).toarray()
    gi = igraph.Graph.Adjacency((A > 0).tolist())
    vf = vertex_features(gi)
    #map to natural order
    # node2index=dict(zip(list(g),range(len(g))))
    # reorder = [node2index[str(i)] for i in range(len(g))]
    # np.save("./intermedia/recur_feature_rand",vf.A[reorder])
    np.save("./intermedia/recur_feature_rand", vf.A)
def visual(vf):
    X_embedded = TSNE(n_components=2,metric="cosine").fit_transform(vf)
    plt.scatter(*X_embedded.T)
    plt.show()
def get_and_save_role_similarity(rolenum):
    vf = np.load("./intermedia/recur_feature.npy")
    m, h = get_factorization(vf, rolenum)
    sim=pair.cosine_similarity(m,m)
    np.save("./intermedia/role_similarity"+str(rolenum),sim)


def get_sparse_similarity(h,A, rolenum=10):
    # if (isinstance(h, np.matrix)):
    #     h = h.A
    s = h.dot(h.T)
    # sim = s / norm(h, axis=1).reshape(1, -1) / norm(h, axis=1).reshape(-1, 1)
    sim = pair.cosine_similarity(h)
    # deg = A.sum(0).A[0]
    # set_zero=[i for i in range(A.shape[0]) if deg[i]<1500]
    # sim[set_zero, :] = 0
    # sim[:,set_zero]=0
    # for threshhold  in range(70,100,2):
    #     threshhold=threshhold/100.0
    #     print("similarity larger than {} numbers".format(threshhold),np.sum(sim>threshhold))

    #sp.save_npz("./intermedia/similarity_for_barbell" + str(rolenum), sp.csr_matrix(sim))
    sp.save_npz("./../role_feature/similarity_for_blogcatalog" + str(rolenum), sp.csr_matrix(sim))
def get_igraph_from_sparse(A):
    source,targets=A.nonzero()
    weights=A[source,targets]
    gi=igraph.Graph(list(zip(source,targets)),directed=True,edge_attrs={'weight':weights})
    return gi
if __name__=="__main__":
    #G=nx.read_edgelist("./../data/bc_edgelist.txt")
    #A = nx.adjacency_matrix(G)
    A = load_adjacency_matrix("./../data/blogcatalog.mat",'network')
    #A=scipy.io.loadmat("/home/xuyou/GER/data/flickr.mat")['network']
    gi = get_igraph_from_sparse(A)
    vf=vertex_features(gi)

    #get_and_save_recursive_feature()
    #vf=np.load("./intermedia/recur_feature.npy")
    # vf=vf[np.random.permutation(range(vf.shape[0]))]
    # label = load_label("/home/xuyou/GER/data/blogcatalog.mat",variable_name='group')
    # for num in range(20,40,5):
    #     get_and_save_role_similarity(num)
    for r in range(5,20):
        print("computin for role",r)
        m, h = get_factorization(vf, r)
        #get_row_similarity(m,r)
        get_sparse_similarity(m,A,r)
    #make_sense(Gi, m)
    # for role in range(40,60):
    #     print("for {} role".format(role))
    #     m,h=get_factorization(vf,role)
    #     predict_cv(m.A, label, train_ratio=50 / 100.,
    #                n_splits=10, C=1, random_state=123)
        #get_row_similarity(m)
