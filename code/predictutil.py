#!/usr/bin/env python
# encoding: utf-8


import os
import pickle as pkl
import numpy as np
import scipy.io
import argparse
import logging
import pandas as pd
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

logger = logging.getLogger(__name__)

def construct_indicator(y_score, y):
    # rank the labels by the scores directly
    num_label = np.sum(y, axis=1, dtype=np.int)
    y_sort = np.fliplr(np.argsort(y_score, axis=1))
    y_pred = np.zeros_like(y, dtype=np.int)
    for i in range(y.shape[0]):
        for j in range(num_label[i]):
            y_pred[i, y_sort[i, j]] = 1
    return y_pred

def load_w2v_feature(file):
    with open(file, "rb") as f:
        nu = 0
        for line in f:
            content = line.strip().split()
            nu += 1
            if nu == 1:
                n, d = int(content[0]), int(content[1])
                feature = [[] for i in range(n)]
                continue
            index = int(content[0])
            for x in content[1:]:
                feature[index].append(float(x))
    for item in feature:
        assert len(item) == d
    return np.array(feature, dtype=np.float32)


def load_label(file, variable_name="group"):
    data = scipy.io.loadmat(file)
    logger.info("loading mat file %s", file)
    label = data[variable_name].todense().astype(np.int)
    label = np.array(label)
    print(label.shape, type(label), label.min(), label.max())
    return label

def predict_cv(X, y, train_ratio=0.2, n_splits=10, random_state=0, C=1.):
    micro, macro = [], []
    shuffle = ShuffleSplit(n_splits=n_splits, test_size=1-train_ratio,
            random_state=random_state)
    for train_index, test_index in shuffle.split(X):
        #print(train_index.shape, test_index.shape)
        assert len(set(train_index) & set(test_index)) == 0
        assert len(train_index) + len(test_index) == X.shape[0]
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = OneVsRestClassifier(
                LogisticRegression(
                    C=C,
                    solver="liblinear",
                    multi_class="ovr"),
                n_jobs=1)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)
        y_pred = construct_indicator(y_score, y_test)
        mi = f1_score(y_test, y_pred, average="micro")
        #acc=np.sum(np.argmax(y_pred,1)==np.argmax(y_test,1))/float(y_pred.shape[0])
        #print("acc",acc)
        ma = f1_score(y_test, y_pred, average="macro")
        #logger.info("micro f1 %f macro f1 %f", mi, ma)
        micro.append(mi)
        macro.append(ma)
    logger.info("%d fold validation, training ratio %f", len(micro), train_ratio)
    logger.info("Average micro %.2f, Average macro %.2f",
            np.mean(micro) * 100,
            np.mean(macro) * 100)
    print ("Average micro %.2f, Average macro %.2f",
            np.mean(micro) * 100,
            np.mean(macro) * 100)
    return [np.mean(micro) * 100,np.mean(macro) * 100]
def predict_svc(X, y, train_ratio=0.2, n_splits=10, random_state=0, C=1.):
    micro, macro,acc = [], [],[]
    shuffle = ShuffleSplit(n_splits=n_splits, test_size=1-train_ratio,
            random_state=random_state)
    for train_index, test_index in shuffle.split(X):
        #print(train_index.shape, test_index.shape)
        assert len(set(train_index) & set(test_index)) == 0
        assert len(train_index) + len(test_index) == X.shape[0]
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = OneVsRestClassifier(
                LogisticRegression(
                    C=C,
                    solver="liblinear",
                    multi_class="ovr"),
                n_jobs=1)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)
        y_pred = construct_indicator(y_score, y_test)
        mi = f1_score(y_test, y_pred, average="micro")
        accu=np.sum(np.argmax(y_pred,1)==np.argmax(y_test,1))/float(y_pred.shape[0])
        acc.append(accu)
        ma = f1_score(y_test, y_pred, average="macro")
        #logger.info("micro f1 %f macro f1 %f", mi, ma)
        micro.append(mi)
        macro.append(ma)
    # logger.info("%d fold validation, training ratio %f", len(micro), train_ratio)
    # logger.info("Average micro %.2f, Average macro %.2f",
    #         np.mean(micro) * 100,
    #         np.mean(macro) * 100)
    print ("Average accuracy%.2f",
            np.mean(acc) * 100)
    return np.mean(acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, default="/home/xuyou/GER/data/blogcatalog.mat",
            help="input file path for labels (.mat)")
    parser.add_argument("--embedding", type=str, default="/home/xuyou/GER/output_embed/rand_emeb.npy",
            help="input file path for embedding (.npy)")
    parser.add_argument("--matfile-variable-name", type=str, default='group',
            help='variable name of adjacency matrix inside a .mat file.')
    parser.add_argument("--seed", type=int, default=123,
            help="seed used for random number generator when randomly split data into training/test set.")
    parser.add_argument("--start-train-ratio", type=int, default=10,
            help="the start value of the train ratio (inclusive).")
    parser.add_argument("--stop-train-ratio", type=int, default=90,
            help="the end value of the train ratio (inclusive).")
    parser.add_argument("--num-train-ratio", type=int, default=9,
            help="the number of train ratio choosed from [train-ratio-start, train-ratio-end].")
    parser.add_argument("--C", type=float, default=1.0,
            help="inverse of regularization strength used in logistic regression.")
    parser.add_argument("--num-split", type=int, default=10,
            help="The number of re-shuffling & splitting for each train ratio.")
    args = parser.parse_args()
    logging.basicConfig(
            #filename="%s.log" % args.embedding, filemode="w", # uncomment this to log to file
            level=logging.INFO,
            format='%(asctime)s %(message)s') # include timestamp
    logger.info("Loading label from %s...", args.label)
    if(args.label.split('.')[1]=="mat"):
        label = load_label(file=args.label, variable_name=args.matfile_variable_name)
    else:
        g = nx.read_edgelist("/home/xuyou/GER/data/brazil-airports.edgelist")
        label = pd.read_csv("/home/xuyou/GER/data/labels-brazil-airports.txt", sep=" ")['label'].to_numpy()
        node2index = dict(zip(list(g), range(len(g))))
        reorder = [node2index[str(i)] for i in range(len(g))]
        label = label[reorder]
        one_hot_label = np.zeros((len(label), 4))
        one_hot_label[range(len(label)), label] = 1
        label=one_hot_label
    logger.info("Label loaded!")

    logger.info("Loading network embedding from %s...", args.embedding)
    ext = os.path.splitext(args.embedding)[1]
    if ext == ".npy":
        embedding = np.load(args.embedding)
    elif ext == ".pkl":
        with open(args.embedding, "rb") as f:
            embedding = pkl.load(f)
    else:
        # Load word2vec format
        embedding = load_w2v_feature(args.embedding)
    logger.info("Network embedding loaded!")

    train_ratios = np.linspace(args.start_train_ratio, args.stop_train_ratio,
            args.num_train_ratio)
    #######tuning lambda
    # for lam in [num / 100. for num in range(1, 100, 5)]:
    #     fname="./../output_embed/blog_final_sm2_mf_lamb"+str(lam)+".npy"
    #     print("evalutating for lambda{}".format(lam))
    #     embed=np.load(fname)
    #     print("features shape",embed.shape,"label shape",label.shape)
    #     for tr in train_ratios:
    #         predict_cv(embed, label, train_ratio=tr/100.,
    #                 n_splits=args.num_split, C=args.C, random_state=args.seed)
    #####tuning role
    for r in range(5,15):
        fname="./../output_embed/blog_lamb0.01_role5.npy"
        print("evalutating for role{}".format(r))
        embed=np.load(fname)
        print("features shape",embed.shape,"label shape",label.shape)
        for tr in train_ratios:
            predict_cv(embed, label, train_ratio=tr/100.,
                    n_splits=args.num_split, C=args.C, random_state=args.seed)