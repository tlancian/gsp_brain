import utils
import os
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn import preprocessing
import pygsp
import csv
import random
import pandas as pd


dataset = "age"
class_1 = "adulescents"
class_2 = "children"
filtering = True
sample_reconstruct = True



#SIGNALS
dir1 = "{}/signals/{}".format(dataset, class_1)
files_c1 =  os.listdir(dir1)
c1 = ["{}/{}".format(dir1,elem) for elem in files_c1]

dir2 = "{}/signals/{}".format(dataset, class_2)
files_c2 =  os.listdir(dir2)
c2 = ["{}/{}".format(dir2,elem) for elem in files_c2]


sig_c1 = preprocessing.normalize(np.concatenate(list(map(lambda x : np.transpose(np.loadtxt(x, delimiter = "\t", skiprows = 1)), c1)), axis = 1), norm = "l2")
sig_c2 = preprocessing.normalize(np.concatenate(list(map(lambda x : np.transpose(np.loadtxt(x, delimiter = "\t", skiprows = 1)), c2)), axis = 1), norm = "l2")


kernel = RBF()

sig_c1 = kernel(sig_c1)
sig_c2 = kernel(sig_c2)

np.fill_diagonal(sig_c1, 0)
np.fill_diagonal(sig_c2, 0)

net_c1 = pygsp.graphs.graph.Graph(sig_c1)
net_c1.compute_fourier_basis()
net_c2 = pygsp.graphs.graph.Graph(sig_c2)
net_c2.compute_fourier_basis()

gft_c1 = []
gft_c2 = []

tau = 0.05
def g(x):
    return 1. / (1. + tau * x)

g1 = pygsp.filters.Filter(net_c1, g)
g2 = pygsp.filters.Filter(net_c2, g)


s = random.sample(range(116), int(round(116*0.8, 0)))


for file in c1:
    
    net = np.mean(np.loadtxt(file, delimiter = "\t", skiprows = 1), axis = 0)   
    
    if sample_reconstruct:
        net = np.array(utils.reconstructer(net, s, net_c1.U, 100))
    
    
    if filtering:
        net = g1.filter(net)
        

    gft_c1.append([round(elem,3) for elem in net_c1.gft(net)])


for file in c2:
    
    net = np.mean(np.loadtxt(file, delimiter = "\t", skiprows = 1), axis = 0)
    
    if sample_reconstruct:
        net = np.array(utils.reconstructer(net, s, net_c2.U, 100))
    
    if filtering:
        net = g2.filter(net)
    gft_c2.append([round(elem,3) for elem in net_c2.gft(net)])
    
    
gft_c1 = dict(zip([elem[:-4] for elem in files_c1], gft_c1))
gft_c2 = dict(zip([elem[:-4] for elem in files_c2], gft_c2))




#NETWORKS
dir1 = "{}/networks/{}".format(dataset, class_1)
files_c1 =  os.listdir(dir1)
c1 = ["{}/{}".format(dir1,elem) for elem in files_c1]

dir2 = "{}/networks/{}".format(dataset, class_2)
files_c2 =  os.listdir(dir2)
c2 = ["{}/{}".format(dir2,elem) for elem in files_c2]

netfeat_c1 = dict(zip([elem[:-4] for elem in files_c1], map(lambda x: utils.metrics(x), c1)))
netfeat_c2 = dict(zip([elem[:-4] for elem in files_c2], map(lambda x: utils.metrics(x), c2)))


##############################################################################

metadata = pd.read_csv("raw/metadata.csv")#, usecols = ["SUB_ID", "DX_GROUP", "AGE_AT_SCAN", "SEX", "DSM_IV_TR", "FILE_ID"])
metadata = metadata[metadata["FILE_ID"] != "no_filename"]

dataset = [list(gft_c1[file[:-4]]) + list(netfeat_c1[file[:-4]]) + [float(metadata["AGE_AT_SCAN"][metadata["FILE_ID"] == file[:-4]])] +  [1] for file in files_c1] + [list(gft_c2[file[:-4]]) + list(netfeat_c2[file[:-4]]) + [float(metadata["AGE_AT_SCAN"][metadata["FILE_ID"] == file[:-4]])] + [2] for file in files_c2]

with open('dataset.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["gft_{}".format(num) for num in range(1,117)]+["dg_{}".format(num) for num in range(1,117)]+
                 ["bc_{}".format(num) for num in range(1,117)]+["cc_{}".format(num) for num in range(1,117)] + ["age", "label"])
    writer.writerows(dataset)
    

