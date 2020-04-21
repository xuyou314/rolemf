#!/usr/bin/env python
# encoding: utf-8
import scipy.io
import scipy.sparse as sparse
from scipy.sparse import csgraph
import numpy as np
import argparse
import logging
import theano
from theano import tensor as T
import scipy.sparse as sp
from sklearn.manifold import TSNE
import sys
import networkx as nx
logger = logging.getLogger(__name__)
theano.config.exception_verbosity='high'
sys.path.append("../")

def load_adjacency_matrix(file, variable_name="network"):
     if(file.split('.')[-1]=='mat'):
        data = scipy.io.loadmat(file)
        logger.info("loading mat file %s", file)
        return data[variable_name]
     else:
         g=nx.read_edgelist(file)
         return nx.adjacency_matrix(g)

def deepwalk_filter(evals, window):
    for i in range(len(evals)):
        x = evals[i]
        evals[i] = 1. if x >= 1 else x*(1-x**window) / (1-x) / window
    evals = np.maximum(evals, 0)
    logger.info("After filtering, max eigenvalue=%f, min eigenvalue=%f",
            np.max(evals), np.min(evals))
    return evals
def deepwalk_filter2(evals, window):
    for i in range(len(evals)):
        x = evals[i]
        evals[i] = 1. if x >= 1 else x*(1-x**window) / (1-x) / window
    logger.info("After filtering, max eigenvalue=%f, min eigenvalue=%f",
            np.max(evals), np.min(evals))
    return evals
def approximate_normalized_graph_laplacian(A, rank, which="LA"):
    n = A.shape[0]
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    # X = D^{-1/2} W D^{-1/2}
    X = sparse.identity(n) - L
    logger.info("Eigen decomposition...")
    #evals, evecs = sparse.linalg.eigsh(X, rank,
    #        which=which, tol=1e-3, maxiter=300)
    evals, evecs = sparse.linalg.eigsh(X, rank, which=which)
    logger.info("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(evals), np.min(evals))
    logger.info("Computing D^{-1/2}U..")
    D_rt_inv = sparse.diags(d_rt ** -1)
    D_rt_invU = D_rt_inv.dot(evecs)
    return evals, D_rt_invU

def approximate_deepwalk_matrix(evals, D_rt_invU, window, vol, b):
    evals = deepwalk_filter(evals, window=window)
    X = sparse.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
    m = T.matrix()
    mmT = T.dot(m, m.T) * (vol/b)
    f = theano.function([m], T.log(T.maximum(mmT, 1)))
    Y = f(X.astype(theano.config.floatX))
    logger.info("Computed DeepWalk matrix with %d non-zero elements",
            np.count_nonzero(Y))
    return sparse.csr_matrix(Y)

def svd_deepwalk_matrix(X, dim):
    u, s, v = sparse.linalg.svds(X, dim, return_singular_vectors="u")
    # return U \Sigma^{1/2}
    return sparse.diags(np.sqrt(s)).dot(u.T).T
def tsne_mf(X,dim):
    embed=TSNE(n_components=dim).fit_transform(X.A)
    return embed
def netmf_large(args):
    logger.info("Running NetMF for a large window size...")
    logger.info("Window size is set to be %d", args.window)
    # load adjacency matrix
    A = load_adjacency_matrix(args.input,
            variable_name=args.matfile_variable_name)
    vol = float(A.sum())
    # perform eigen-decomposition of D^{-1/2} A D^{-1/2}
    # keep top #rank eigenpairs
    evals, D_rt_invU = approximate_normalized_graph_laplacian(A, rank=args.rank, which="LA")

    # approximate deepwalk matrix
    deepwalk_matrix = approximate_deepwalk_matrix(evals, D_rt_invU,
            window=args.window,
            vol=vol, b=args.negative)

    # factorize deepwalk matrix with SVD
    deepwalk_embedding = svd_deepwalk_matrix(deepwalk_matrix, dim=args.dim)

    logger.info("Save embedding to %s", args.output)
    np.save(args.output, deepwalk_embedding, allow_pickle=False)


def direct_compute_deepwalk_matrix(A, window, b):
    n = A.shape[0]
    vol = float(A.sum())
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    # X = D^{-1/2} A D^{-1/2}
    X = sparse.identity(n) - L
    S = np.zeros_like(X)
    X_power = sparse.identity(n)
    for i in range(window):
        logger.info("Compute matrix %d-th power", i+1)
        X_power = X_power.dot(X)
        S += X_power
    S *= vol / window / b
    D_rt_inv = sparse.diags(d_rt ** -1)
    M = D_rt_inv.dot(D_rt_inv.dot(S).T)
    m = T.matrix()
    f = theano.function([m], T.log(T.maximum(m, 1)))
    Y = f(M.todense().astype(theano.config.floatX))
    return sparse.csr_matrix(Y)
def compute_deepwalk_matrix_for_softmax(A, window, b):
    n = A.shape[0]
    #vol = float(A.sum())
    #L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    # X = D^{-1/2} A D^{-1/2}
    d_rv=sp.diags(1.0 / A.sum(0).A[0])
    l = d_rv.dot(A)
    S = np.zeros_like(A)
    X_power = sparse.identity(n)
    for i in range(window):
        logger.info("Compute matrix %d-th power", i+1)
        X_power = X_power.dot(l)
        S += X_power
    S =S / window
    sa=S.A

    return sparse.csr_matrix(np.log(sa))
def netmf_small(args):
    logger.info("Running NetMF for a small window size...")
    logger.info("Window size is set to be %d", args.window)
    # load adjacency matrix
    A = load_adjacency_matrix(args.input,
            variable_name=args.matfile_variable_name)
    # directly compute deepwalk matrix
    deepwalk_matrix =compute_deepwalk_matrix_for_softmax(A,
            window=args.window, b=args.negative)

    # factorize deepwalk matrix with SVD
    deepwalk_embedding =svd_deepwalk_matrix(deepwalk_matrix, dim=args.dim)
    logger.info("Save embedding to %s", args.output)
    np.save(args.output, deepwalk_embedding, allow_pickle=False)
def rolemf(args):
    A = load_adjacency_matrix(args.input,
            variable_name=args.matfile_variable_name)
    lens=A.shape[0]
    # g=nx.read_edgelist("/home/xuyou/ger/data/bc_edgelist.txt")
    # set_zero=[int(node) for node in g if g.degree[node] < 1000]
    set_zero = np.where(A.sum(0)<1000)[1]
    #role_sim=np.load("./../role_feature/role_similarity20.npy")
    role_sim = sp.load_npz("./../role_feature/similarity_for_blogcatalog8.npz")
    role_sim[set_zero,:]=0
    role_sim[:,set_zero]=0
    role_sim[role_sim < 0.9] = 0
    role_sim[range(lens),range(lens)]=0
    for lam in [num/100. for num in range(16,200,30)]:
        obj_mat=A+role_sim*lam
        print("computing for lambda equals ",lam)
        #print(obj_mat,obj_mat.shape)
        deepwalk_matrix = compute_deepwalk_matrix_for_softmax(obj_mat,
                                                              window=args.window, b=args.negative)
        deepwalk_embedding = svd_deepwalk_matrix(deepwalk_matrix, dim=args.dim)
        #logger.info("Save embedding to %s", args.output)
        np.save("./../embed/blog_window{}_sm_mf_lamb{}".format(args.window, lam), deepwalk_embedding, allow_pickle=False)


def role_airline_mf(args):
    g=nx.read_edgelist(args.input)
    A = nx.adjacency_matrix(g)
    lens = A.shape[0]
    set_zero=[int(node) for node in g if g.degree[node] < 1000]
    role_sim=np.load("./../intermedia/similarity_for_airline_eu.npy")
    # role_sim[set_zero,:]=0
    # role_sim[:,set_zero]=0
    # role_sim[role_sim < 0.9] = 0
    # role_sim[range(lens),range(lens)]=0
    for lam in [num/100. for num in range(1,400,5)]:
        obj_mat=A+role_sim*lam
        print("computing for lambda equals ",lam)
        print(obj_mat,obj_mat.shape)
        deepwalk_matrix = compute_deepwalk_matrix_for_softmax(obj_mat,
                                                              window=args.window, b=args.negative)
        deepwalk_embedding = svd_deepwalk_matrix(deepwalk_matrix, dim=args.dim)
        #logger.info("Save embedding to %s", args.output)
        np.save("./../output_embed/eu_airline_lamb{}".format(lam), deepwalk_embedding, allow_pickle=False)
    return
def role_airline_mf2(args):
    g=nx.read_edgelist(args.input)
    A = nx.adjacency_matrix(g)
    lens = A.shape[0]
    n2i = dict(zip(list(g), range(len(g))))
    set_zero=[n2i[node] for node in g if g.degree[node] < 1000]
    for r in range(10,11):
        role_sim=np.load("./../intermedia/similarity_for_catablog{}.npy".format(r))
        role_sim[set_zero,:]=0
        role_sim[:,set_zero]=0
        role_sim[role_sim < 0.9] = 0
        role_sim[range(lens),range(lens)]=0
    #for lam in [num/100. for num in range(1,400,5)]:
        for lam in [num / 100. for num in range(1, 400,5)]:
            obj_mat=A+role_sim*lam
            print("computing for lambda equals{} role equals{}".format(lam,r))
            #print(obj_mat,obj_mat.shape)
            deepwalk_matrix = compute_deepwalk_matrix_for_softmax(obj_mat,
                                                                  window=args.window, b=args.negative)
            deepwalk_embedding = svd_deepwalk_matrix(deepwalk_matrix, dim=args.dim)
            #logger.info("Save embedding to %s", args.output)
            np.save("./../output_embed/blog_lamb{}_role{}".format(lam,r), deepwalk_embedding, allow_pickle=False)

def netmf_small_origin(args):
    logger.info("Running NetMF for a small window size...")
    logger.info("Window size is set to be %d", args.window)
    # load adjacency matrix
    A = load_adjacency_matrix(args.input,
            variable_name=args.matfile_variable_name)
    # directly compute deepwalk matrix
    deepwalk_matrix = direct_compute_deepwalk_matrix(A,
            window=args.window, b=args.negative)

    # factorize deepwalk matrix with SVD
    deepwalk_embedding = svd_deepwalk_matrix(deepwalk_matrix, dim=args.dim)
    logger.info("Save embedding to %s", args.output)
    np.save(args.output, deepwalk_embedding, allow_pickle=False)
def role_mf_wiki(args):
    A = load_adjacency_matrix(args.input,
            variable_name=args.matfile_variable_name)
    lens=A.shape[0]
    deg=A.sum(0).A[0]
    set_zero=[i for i in range(lens) if deg[i]<200]
    role_sim=np.load("./../intermedia/similarity_for_wiki5.npy")
    role_sim[set_zero,:]=0
    role_sim[:,set_zero]=0
    role_sim[role_sim < 0.9] = 0
    role_sim[range(lens),range(lens)]=0
    for lam in [num/100. for num in range(100,800,15)]:
        obj_mat=A+role_sim*lam
        print("computing for lambda equals ",lam)
        #print(obj_mat,obj_mat.shape)
        deepwalk_matrix = compute_deepwalk_matrix_for_softmax(obj_mat,
                                                              window=args.window, b=args.negative)
        deepwalk_embedding = svd_deepwalk_matrix(deepwalk_matrix, dim=args.dim)
        #logger.info("Save embedding to %s", args.output)
        np.save("./../output_embed/wiki_rdwmf_degree_role5_lamb{}".format(lam), deepwalk_embedding, allow_pickle=False)
def role_mf_flick(args):
    A = load_adjacency_matrix(args.input,
            variable_name=args.matfile_variable_name)
    role_sim=sp.load_npz("/home/xuyou/GER/intermedia/similarity_for_flicker10.npz")
    for lam in [num/100. for num in range(1,100,5)]:
        obj_mat=A+role_sim*lam
        print("computing for lambda equals ",lam)
        #print(obj_mat,obj_mat.shape)
        deepwalk_matrix = compute_deepwalk_matrix_for_softmax(obj_mat,
                                                              window=args.window, b=args.negative)
        deepwalk_embedding = svd_deepwalk_matrix(deepwalk_matrix, dim=args.dim)
        #logger.info("Save embedding to %s", args.output)
        np.save("./../output_embed/flicker_rdwmf_role10_lamb{}".format(lam), deepwalk_embedding, allow_pickle=False)

def large_rolemf(args):
    g=nx.read_edgelist(args.input)
    A = nx.adjacency_matrix(g)
    lens = A.shape[0]
    #n2i = dict(zip(list(g), range(len(g))))
    i2n=list(g)
    #set_zero=[n2i[node] for node in g if g.degree[node] < 1000]
    set_zero = [i for i in range(len(g)) if g.degree[i2n[i]]<1000]
    for r in range(10,11):
        role_sim=np.load("./../intermedia/similarity_for_catablog{}.npy".format(r))
        role_sim[set_zero,:]=0
        role_sim[:,set_zero]=0
        role_sim[role_sim < 0.9] = 0
        role_sim[range(lens),range(lens)]=0
        for lam in [num / 100. for num in range(1, 400, 5)]:
            obj_mat = A + role_sim * lam
            evals, D_rt_invU = approximate_normalized_graph_laplacian(obj_mat, rank=args.rank, which="BE")
            vol=np.sum(obj_mat)
            # approximate deepwalk matrix
            deepwalk_matrix = approximate_deepwalk_matrix(evals, D_rt_invU,args.window,vol=vol, b=args.negative)
            #deepwalk_matrix=compute_deepwalk_matrix_for_softmax(obj_mat,10,0)
            # factorize deepwalk matrix with SVD
            deepwalk_embedding = svd_deepwalk_matrix(deepwalk_matrix, dim=args.dim)

            logger.info("Save embedding to %s", args.output)
            np.save("./../output_embed/softmax_test_lamb{}".format(lam), deepwalk_embedding, allow_pickle=False)

def approximate_softmax_matrix(evals, D_rt_invU, window,A):
    evals = deepwalk_filter2(evals, window=window)
    X = sparse.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
    Y= X.dot(X.T).dot(np.diag(A.sum(0).A[0]))

    logger.info("Computed DeepWalk matrix with %d non-zero elements",
            np.count_nonzero(Y))
    return sparse.csr_matrix(np.nan_to_num(np.log(Y/window)))
def role_barbell_mf(args):
    g=nx.read_edgelist(args.input)
    A = nx.adjacency_matrix(g)
    #lens = A.shape[0]
    #set_zero=[int(node) for node in g if g.degree[node] < 1000]
    role_sim=sp.load_npz("./../intermedia/similarity_for_barbell6.npz")
    # role_sim[set_zero,:]=0
    # role_sim[:,set_zero]=0
    # role_sim[role_sim < 0.9] = 0
    # role_sim[range(lens),range(lens)]=0
    for lam in [num/100. for num in range(0,20,1)]:
        obj_mat=A+role_sim*lam
        print("computing for lambda equals ",lam)
        print(obj_mat,obj_mat.shape)
        deepwalk_matrix = compute_deepwalk_matrix_for_softmax(obj_mat,
                                                              window=args.window, b=args.negative)
        deepwalk_embedding = svd_deepwalk_matrix(deepwalk_matrix, dim=args.dim)
        #logger.info("Save embedding to %s", args.output)
        np.save("./../output_embed/barbell_lamb_vis{}".format(lam), deepwalk_embedding, allow_pickle=False)
    return
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default="./../data/blogcatalog.mat")
    parser.add_argument('--matfile-variable-name', default='network',
            help='variable name of adjacency matrix inside a .mat file.')
    parser.add_argument("--output", type=str,
    default="/home/xuyou/GER/output_embed/blog_role_large",
            help="embedding output file path")

    parser.add_argument("--rank", default=256, type=int,
            help="#eigenpairs used to approximate normalized graph laplacian.")
    parser.add_argument("--dim", default=128, type=int,
            help="dimension of embedding")
    parser.add_argument("--window", default=8,
            type=int, help="context window size")
    parser.add_argument("--negative", default=1.0, type=float,
            help="negative sampling")

    parser.add_argument('--large', dest="large", action="store_true",
            help="using netmf for large window size")
    parser.add_argument('--small', dest="large", action="store_false",
            help="using netmf for small window size")
    parser.set_defaults(large=False)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(message)s') # include timestamp

    # if args.large:
    #     netmf_large(args)
    # else:
        #role_barbell_mf(args)
    rolemf(args)

