from numpy.core.multiarray import ndarray

from predictutil import *
import networkx as nx
import numpy as np
import pandas as pd
import pickle
import os
def npy_format_eval():
    g = nx.read_edgelist("./data/europe-airports.edgelist")
    #embed=np.load("/home/xuyou/GER/output_embed/airport_br_baseline.npy")
    label=pd.read_csv("/home/xuyou/GER/data/labels-europe-airports.txt",sep=" ")['label'].to_numpy()
    #node2index=dict(zip(list(g),range(len(g))))
    #reorder=[node2index[str(i)] for i in range(len(g))]
    reorder=[int(node) for node in g]
    label=label[reorder]
    one_hot_label=np.zeros((len(label),4))
    one_hot_label[range(len(label)),label]=1
    lambs,accs=[],[]
    for lam in [num / 100. for num in range(1, 400, 5)]:
        fname = "./output_embed/eu_airline_lamb" + str(lam) + ".npy"
        print("evalutating for lambda{}".format(lam))
        embed = np.load(fname)
        for train_ratios in [i/100. for i in range(50,51,1)]:
            print("evalauting for training ratio",train_ratios)
            acc=predict_svc(embed,one_hot_label,train_ratio=train_ratios)
        accs.append(acc)
        lambs.append(lam)
    # d={"lambs":lambs,"accuracys":accs}
    # df=pd.DataFrame(data=d)
    # df=pd.read_csv("/home/xuyou/GER/experiment_visual/lam_parameter")
    # df["acc_eu"]=accs
    # df.to_csv("/home/xuyou/GER/experiment_visual/lam_parameter",index=False)
def npy_format_eval_forrole():
    g = nx.read_edgelist("/home/xuyou/GER/data/brazil-airports.edgelist")
    #embed=np.load("/home/xuyou/GER/output_embed/airport_br_baseline.npy")
    label=pd.read_csv("/home/xuyou/GER/data/labels-brazil-airports.txt",sep=" ")['label'].to_numpy()
    #node2index=dict(zip(list(g),range(len(g))))
    #reorder=[node2index[str(i)] for i in range(len(g))]
    reorder=[int(node) for node in g]
    label=label[reorder]
    one_hot_label=np.zeros((len(label),4))
    one_hot_label[range(len(label)),label]=1
    lambs,accs=[],[]
    for r in range(5,6):
        #fname = "./output_embed/eu_airline_lamb0.5_role{}.npy".format(r)
        fname="./output_embed/graphwave_br.npy"
        print("evalutating for role{}".format(r))
        embed = np.load(fname)
        for train_ratios in [i/100. for i in range(10,91,10)]:
            print("evalauting for training ratio",train_ratios)
            acc=predict_svc(embed,one_hot_label,train_ratio=train_ratios)
        accs.append(acc)
    print (accs)
def w2v_format_eval():
    f = open("./../struc2vec/emb/blog.emb")
    g = nx.read_edgelist("./data/bc_edgelist.txt")
    dictv = {}
    for l in f:
        vec = l.split()
        if (len(vec) > 2):
            dictv[vec[0]] = list(map(float, vec[1:]))
    label=load_label("./data/blogcatalog.mat","group")
    #label=pd.read_csv("/home/xuyou/GER/data/labels-europe-airports.txt",sep=" ")['label'].to_numpy()
    embed=np.array([dictv[str(i)]  for i in range(len(g))])
    #node2index=dict(zip(list(g),range(len(g))))
    #reorder=[node2index[str(i)] for i in range(len(g))]
    #reorder=[int(node) for node in g]
    #label=label[reorder]
    # one_hot_label=np.zeros((len(label),4))
    # one_hot_label[range(len(label)),label]=1
    for train_ratios in [i / 100. for i in range(10, 91, 10)]:
        print("evalauting for training ratio", train_ratios)
        predict_cv(embed,label, train_ratio=train_ratios)
def mat_format_eval():

    #g = nx.read_edgelist("./../data/bc_edgelist.txt")
    # dictv = {}
    # for l in f:
    #     vec = l.split()
    #     if (len(vec) > 2):
    #         dictv[vec[0]] = map(float, vec[1:])
    label=load_label("./../data/blogcatalog.mat","group")  # type: ndarray
    #n2i=dict(zip(list(g),range(len(g))))
    #reorder=[int(node) for node in g]
    #label=label[reorder]
    res_dic=dict( [(i / 100.,[]) for i in range(10, 91, 10)])
    #output = open('f1score.pkl', 'wb')
    #label=pd.read_csv("/home/xuyou/GER/data/labels-europe-airports.txt",sep=" ")['label'].to_numpy()
    for lam in [num / 100. for num in range(1, 100, 5)]:
        #embed=np.load("/home/xuyou/graphwave/graphwave_blog.npy")
        #embed=np.load("./output_embed/softmax_test_lamb{}.npy".format(0.66))
        embed=np.load("./../embed/blog_window8_sm_mf_lamb1.96.npy")
        #node2index=dict(zip(list(g),range(len(g))))
        #reorder=[node2index[str(i)] for i in range(len(g))]
        #reorder=[int(node) for node in g]
        #label=label[reorder]
        # one_hot_label=np.zeros((len(label),4))
        # one_hot_label[range(len(label)),label]=1
        print ("eval lam ",lam)
        for train_ratios in [i / 100. for i in range(10, 91, 10)]:
            print("evalauting for training ratio", train_ratios)
            f1=predict_cv(embed,label, train_ratio=train_ratios)
            res_dic[train_ratios].append(f1)
        break
    #pickle.dump(res_dic,output)

def eval_file(edge_file,label_file,npy_file):
    g = nx.read_edgelist(edge_file)
    #embed=np.load("/home/xuyou/GER/output_embed/airport_br_baseline.npy")
    label=pd.read_csv(label_file,sep=" ")['label'].to_numpy()
    #node2index=dict(zip(list(g),range(len(g))))
    #reorder=[node2index[str(i)] for i in range(len(g))]
    reorder=[int(node) for node in g]
    label=label[reorder]
    one_hot_label=np.zeros((len(label),4))
    one_hot_label[range(len(label)),label]=1
    embed=np.load(npy_file)
    for train_ratios in [i / 100. for i in range(10, 91, 10)]:
        print("evalauting for training ratio", train_ratios)
        predict_svc(embed, one_hot_label, train_ratio=train_ratios)
def wiki_eval():
    label = load_label("/home/xuyou/GER/data/POS.mat", "group")

    #for lam in [num / 100. for num in range(1, 100, 5)]:
    for lam in [num/100. for num in range(100,800,15)]:
        print ("eval lam ", lam)
        embed = np.load("./output_embed/wiki_rdwmf_degree_role5_lamb{}.npy".format(lam))
        for train_ratios in [i / 100. for i in range(10, 91, 10)]:
            print("evalauting for training ratio", train_ratios)
            f1 = predict_cv(embed, label, train_ratio=train_ratios)
def w2v_format_eval_forrole():
    g = nx.read_edgelist("/home/xuyou/GER/data/brazil-airports.edgelist")
    #embed=np.load("/home/xuyou/GER/output_embed/airport_br_baseline.npy")
    label=pd.read_csv("/home/xuyou/GER/data/labels-brazil-airports.txt",sep=" ")['label'].to_numpy()
    #node2index=dict(zip(list(g),range(len(g))))
    #reorder=[node2index[str(i)] for i in range(len(g))]
    #reorder=[int(node) for node in g]
    #label=label[reorder]
    one_hot_label=np.zeros((len(label),4))
    one_hot_label[range(len(label)),label]=1
    lambs,accs=[],[]
    f=open("/home/xuyou/struc2vec/emb/br.emb")
    emb=np.empty(shape=(131,16))
    for l in f:
        if(len(l.split())==2):
            continue
        index=int(l.split()[0])
        values=map(float,l.split()[1:])
        emb[index]=values
    for r in range(5,6):
        #fname = "./output_embed/eu_airline_lamb0.5_role{}.npy".format(r)
        for train_ratios in [i/100. for i in range(10,91,10)]:
            print("evalauting for training ratio",train_ratios)
            acc=predict_svc(emb,one_hot_label,train_ratio=train_ratios)
        accs.append(acc)
    print (accs)
if __name__=="__main__":
    #npy_format_eval()
    # eval_file("/home/xuyou/GER/data/europe-airports.edgelist",
    #           "/home/xuyou/GER/data/labels-europe-airports.txt",
    #           "/home/xuyou/GER/output_embed/eu_br_baseline.npy")
    #w2v_format_eval()
    #w2v_format_eval()
    #wiki_eval()
    mat_format_eval()