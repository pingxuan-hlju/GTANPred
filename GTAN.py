import dgl
import torch
import numpy as np
import torch.nn as nn
import networkx as nx
import dgl.function as fn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from numpy import *
from torch import optim
from sklearn.metrics import auc
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from dgl.nn.pytorch import edge_softmax, GATConv

import torch.utils.data as Data
from numpy import mat,matrix,vstack
from torch.autograd import Variable
from numpy import ndarray, eye, matmul, vstack, hstack, array, newaxis, zeros, genfromtxt, savetxt,exp

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def ReadTxt():
    dis_sim = np.loadtxt('../data/LncRNA/dis_sim_matrix_process.txt')
    lnc_sim = np.loadtxt('../data/LncRNA/lnc_sim.txt')
    lnc_dis = np.loadtxt('../data/LncRNA/lnc_dis_association.txt')
    mi_dis = np.loadtxt('../data/LncRNA/mi_dis.txt')
    lnc_mi = np.loadtxt('../data/LncRNA/yuguoxian_lnc_mi.txt')

    return dis_sim,lnc_sim,lnc_dis,mi_dis,lnc_mi

def Count_Value_1(matrix:ndarray, k:int):
    A = array(nonzero(matrix))
    A = A.T
    random.shuffle(A)
    B = array_split(A, k, 0)

    return B

def Count_Value_0(matrix:ndarray, k:int):
    A = array(np.where(matrix == 0))
    A = A.T
    random.shuffle(A)
    B = []
    for i in range (2687):
        B.append(A[i])
    C = np.array(B)
    D = array_split(C, k, 0)

    return D,A

def Make_Train_Test_Set(train_1:ndarray,train_0:ndarray,all_train_0:ndarray):
    matrix1 = []
    matrix0 = []

    for i in range(len(train_1) - 1):
        for j in range(train_1[i].shape[0]):
            matrix1.append(train_1[i][j])
    for m in range(len(train_0) - 1):
        for n in range(train_0[m].shape[0]):
            matrix1.append(train_0[m][n])

    for p in range(train_1[len(train_1)-1].shape[0]):
        matrix0.append(train_1[len(train_1)-1][p])
    for q in range(len(all_train_0)):
        matrix0.append(all_train_0[q])

    matrix_train = np.array(matrix1)
    matrix_test = np.array(matrix0)

    return matrix_train,matrix_test

def Make_One_Features(dis_sim:ndarray,lnc_sim:ndarray,lnc_dis:ndarray,mi_dis:ndarray,lnc_mi:ndarray,X:int ,Y:int):
    a1 = lnc_sim[X]
    b1 = lnc_dis[X]
    c1 = lnc_mi[X]
    A1 = np.hstack((a1, b1, c1))

    a2 = lnc_dis[:, Y]
    b2 = dis_sim[:, Y]
    c2 = mi_dis[:, Y]

    B1 = np.hstack((a2, b2, c2))
    C1 = np.vstack((A1, B1))

    return C1

def Make_Tow_Graph(lnc_sim:ndarray, dis_sim:ndarray):
    g_LncRNA = dgl.DGLGraph()
    g_LncRNA.add_nodes(240)
    for i in range(lnc_sim.shape[0]):
        for j in range(lnc_sim.shape[1]):
            if lnc_sim[i][j] > 0.5:
                g_LncRNA.add_edges(i, j)

    g_Dise = dgl.DGLGraph()
    g_Dise.add_nodes(405)
    for m in range(dis_sim.shape[0]):
        for n in range(dis_sim.shape[1]):
            if dis_sim[m][n] > 0.5:
                g_Dise.add_edges(m, n)

    print(g_LncRNA.number_of_nodes())
    print(g_LncRNA.number_of_edges())
    print(g_LncRNA.node_attr_schemes())
    print(g_LncRNA.edge_attr_schemes())

    print(g_Dise.number_of_nodes())
    print(g_Dise.number_of_edges())
    print(g_Dise.node_attr_schemes())
    print(g_Dise.edge_attr_schemes())

    return g_LncRNA,g_Dise

def Make_Tow_Graph_Feature(dis_sim:ndarray,lnc_sim:ndarray,lnc_dis:ndarray,mi_dis:ndarray,lnc_mi:ndarray):

    LncRNA_Feature = np.hstack((lnc_sim, lnc_dis, lnc_mi))
    Dise_Feature = np.hstack((lnc_dis.T, dis_sim, mi_dis.T))

    return LncRNA_Feature, Dise_Feature

class My_Dataset(Dataset):

    def __init__(self,dis_sim,lnc_sim,lnc_dis,mi_dis,lnc_mi,matrix):
        self.dis_sim = dis_sim
        self.lnc_sim = lnc_sim
        self.lnc_dis = lnc_dis
        self.mi_dis = mi_dis
        self.lnc_mi = lnc_mi
        self.matrix = matrix

    def __getitem__(self, idx):
        X,Y = self.matrix[idx]
        feature_map = Make_One_Features(self.dis_sim, self.lnc_sim, self.lnc_dis, self.mi_dis, self.lnc_mi, X, Y)
        label = self.lnc_dis[X][Y]

        return X, Y, feature_map, label

    def __len__(self):

        return len(self.matrix)

class My_CNN_Tow(nn.Module):
    def __init__(self):
        super(My_CNN_Tow, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(  # now 1, 2, 1140
                in_channels=16,  # input height
                out_channels=32,  # n_filters
                kernel_size=2,  # filter size
                stride=1,  # filter movement/step
                padding=1,  # padding
            ),  # may 16 * 3 * 1141

            nn.BatchNorm2d(32),
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),  # may 16 * 2 * 1140
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 16, 2, 1, 1),  # may 32 * 3 * 1141
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 1, 0),  # may 32 * 2 * 1140
        )

    def forward(self,x):

        x = self.conv1(x)  # batch_size(50) * 16 * 4 * 492
        x = self.conv2(x)

        # x = x.view(x.size(0), -1)
        # out = self.out(x).float()
        # #out = self.softmax(out)

        return x

class My_CNN(nn.Module):
    def __init__(self):

        super(My_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # now 1, 2, 1140
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=2,  # filter size
                stride=1,  # filter movement/step
                padding=1,  # padding
            ),  # may 32 * 3 * 1141

            nn.BatchNorm2d(16),
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),  # may 32 * 2 * 1140
        )

        # self.out = nn.Sequential(nn.Linear(32 * 2 * 1140, 2), nn.BatchNorm1d(2), nn.Dropout(0.5), nn.Sigmoid())
        # self.softmax = torch.nn.Softmax()

    def forward(self,x):
        x = self.conv1(x)  # batch_size(50) * 16 * 4 * 492

        return x

class Attention_feature_level(nn.Module):
    def __init__(self,input_size:int,output_size:int):
        super(Attention_feature_level,self).__init__()

        self.node = nn.Linear(input_size, output_size,bias=True) #input_size * 2   + zs_size(hidden * 2)
        nn.init.xavier_normal_(self.node.weight)

        self.h_n_parameters = nn.Parameter(torch.randn(output_size,input_size))
        nn.init.xavier_normal_(self.h_n_parameters)

    def forward(self, h_n_states):

        # print(h_n_states)
        # print(h_n_states.shape)

        temp_nodes = self.node(h_n_states) # 4 * (input_size *2) ,4 * output_size
        temp_nodes = torch.tanh(temp_nodes)

        nodes_score = torch.matmul(temp_nodes, self.h_n_parameters)

        alpha = F.softmax(nodes_score,dim=2)

        y_i = alpha * h_n_states

        return y_i

class My_FCN(nn.Module):
    def __init__(self):

        super(My_FCN, self).__init__()
        self.out = nn.Sequential(nn.Linear(16 * 2 * 1140, 2),
                                 nn.Dropout(0.5),
                                 nn.Sigmoid())

    def forward(self,x):

        x = x.view(x.size(0), -1)
        out = self.out(x).float()
        return out

class Attention_model(nn.Module):

    def __init__(self,input_size_lnc,input_size_A,input_size_lncmi,input_size_dis,input_size_midis,
                 output_size1,output_size2,output_size3,output_size4,output_size5,output_size6,batch_size):

        super(Attention_model, self).__init__()

        self.attention_ls = Attention_feature_level(input_size_lnc,output_size1) # ①
        self.attention_A = Attention_feature_level(input_size_A, output_size2) # ②
        self.attention_lm = Attention_feature_level(input_size_lncmi, output_size3) # ③
        self.attention_AT = Attention_feature_level(input_size_lnc, output_size4) # ①
        self.attention_ds = Attention_feature_level(input_size_dis, output_size5) # ④
        self.attention_dm = Attention_feature_level(input_size_midis, output_size6) # ⑤

        self.My_CNN_Tow = My_CNN_Tow()
        self.My_CNN = My_CNN()
        self.My_FCN = My_FCN()

    def forward(self, x):

        # print(x.size()[0]) 50
        # print(x.size()[1]) 1
        # print(x.size()[2]) 2
        # print(x.size()[3]) 1140
        # print(x.size())  # [50, 1, 2, 1140]
        #    240*240      240*405           240 * 495
        #     405*240            405*405          495 * 405

        ls = x[:,:,0,:240]
        A = x[:,:,0,240:645]
        lm = x[:,:,0,645:1140]

        AT = x[:,:,1,:240]  # 240
        ds = x[:,:,1,240:645]  # 405
        dm = x[:,:,1,645:1140]  # 495

        """   
        print(ls.shape) # torch.Size([50, 1, 240])
        print(A.shape) # torch.Size([50, 1, 405])
        print(lm.shape) # torch.Size([50, 1, 495])        
        """

        result_ls = self.attention_ls(ls)
        result_A = self.attention_A(A)
        result_lm = self.attention_lm(lm)

        result_AT = self.attention_AT(AT)
        result_ds = self.attention_ds(ds)
        result_dm = self.attention_dm(dm)

        # print(ls.shape)
        # print(A.shape)
        # print(lm.shape)

        # print(result_ls.size())  # [50, 1, 240]

        lnc_RNA = torch.cat((result_ls, result_A,result_lm), dim=2)
        disease = torch.cat((result_AT, result_ds,result_dm), dim=2)
        reslut = torch.cat((lnc_RNA, disease), dim=1)
        reslut =reslut.unsqueeze(dim=1)

        out1 = self.My_CNN(reslut)

        out2 = self.My_CNN_Tow(out1)

        # print(out1.shape) # [50,32,2,1140]
        # print(out2.shape) # [50,32,2,1140]

        out3 = out1 + out2
        # print(out3.shape) # [50,32,2,1140]

        out = self.My_FCN(out3)
        # print(out.shape) # [50,2]

        return out

class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation ):

        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation

        # input projection (no residual)
        self.gat_layers.append(GATConv(in_feats=in_dim, out_feats=num_hidden,
                                       num_heads=heads[0],activation=self.activation))

        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                in_feats=num_hidden * heads[l-1], out_feats=num_hidden,
                num_heads=heads[l], activation=self.activation))

        # output projection
        self.gat_layers.append(GATConv(
            in_feats=num_hidden * heads[-2], out_feats=num_classes,
            num_heads=heads[-1],activation=self.activation))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits

class My_FCN_GAT(nn.Module):
    def __init__(self):

        super(My_FCN_GAT, self).__init__()
        self.out = nn.Sequential(nn.Linear(2 * 800, 2),
                                 nn.Dropout(0.5),
                                 nn.LeakyReLU())
    def forward(self,x):
        x = x.view(x.size(0), -1)
        # out = self.out(x).float()
        out = self.out(x)
        return out

class My_model(nn.Module):
    def __init__(self,g_LncRNA,g_Dise,num_layers,in_dim,num_hidden,num_classes,heads,activation,
                 LncRNA_Feature, Dise_Feature):
        super(My_model, self).__init__()

        self.GAT_Module_LncRNA = GAT(g_LncRNA,num_layers,in_dim,num_hidden,num_classes,heads,activation)
        self.GAT_Module_Dis = GAT(g_Dise,num_layers,in_dim,num_hidden,num_classes,heads,activation)

        self.LncRNA_Feature = LncRNA_Feature.cuda()
        self.Dise_Feature = Dise_Feature.cuda()

        self.My_FCN_GAT = My_FCN_GAT()

    def forward(self, X,Y ):
        result_Lnc = self.GAT_Module_LncRNA(self.LncRNA_Feature)
        result_Dis = self.GAT_Module_Dis(self.Dise_Feature)

        lnc = result_Lnc[X] # torch.Size([50, 800]) <class 'torch.Tensor'>
        dis = result_Dis[Y] # torch.Size([50, 800]) <class 'torch.Tensor'>
        result_FCN = self.My_FCN_GAT(torch.cat((lnc,dis),dim=1)) #  torch.Size([50, 1600]) <class 'torch.Tensor'>

        return  result_FCN

def Count_valid_data(lnc_dis):

    f = zeros((lnc_dis.shape[0], 1), dtype=float64)
    for i in range(lnc_dis.shape[0]):
        f[i] = sum(lnc_dis[i] > 0)

    return f

def caculate_TPR_FPR(RD, f, B):

    old_id = np.argsort(-RD)
    min_f = int(min(f))
    max_f = int(max(f))

    TP_FN = np.zeros((RD.shape[0], 1), dtype=np.float64)
    FP_TN = np.zeros((RD.shape[0], 1), dtype=np.float64)

    TP = np.zeros((RD.shape[0], max_f), dtype=np.float64)
    TP2 = np.zeros((RD.shape[0], min_f), dtype=np.float64)

    FP = np.zeros((RD.shape[0], max_f), dtype=np.float64)
    FP2 = np.zeros((RD.shape[0], min_f), dtype=np.float64)

    P = np.zeros((RD.shape[0], max_f), dtype=np.float64)
    P2 = np.zeros((RD.shape[0], min_f), dtype=np.float64)

    for i in range(RD.shape[0]):
        TP_FN[i] = sum(B[i] == 1)
        FP_TN[i] = sum(B[i] == 0)

    for i in range(RD.shape[0]):
        kk = f[i] / min_f

        for j in range(int(f[i])):
            if j == 0:
                if B[i][old_id[i][j]] == 1:
                    FP[i][j] = 0
                    TP[i][j] = 1
                    P[i][j] = TP[i][j] / (j + 1)
                else:
                    TP[i][j] = 0
                    FP[i][j] = 1
                    P[i][j] = TP[i][j] / (j + 1)
            else:
                if B[i][old_id[i][j]] == 1:
                    FP[i][j] = FP[i][j - 1]
                    TP[i][j] = TP[i][j - 1] + 1
                    P[i][j] = TP[i][j] / (j + 1)
                else:
                    TP[i][j] = TP[i][j - 1]
                    FP[i][j] = FP[i][j - 1] + 1
                    P[i][j] = TP[i][j] / (j + 1)

    ki = 0
    for i in range(RD.shape[0]):
        if TP_FN[i] == 0:
            TP[i] = 0
            FP[i] = 0
            ki = ki + 1
        else:
            TP[i] = TP[i] / TP_FN[i]
            FP[i] = FP[i] / FP_TN[i]

    for i in range(RD.shape[0]):
        kk = f[i] / min_f
        for j in range(min_f):
            TP2[i][j] = TP[i][int(np.round_(((j + 1) * kk))) - 1]
            FP2[i][j] = FP[i][int(np.round_(((j + 1) * kk))) - 1]
            P2[i][j] = P[i][int(np.round_(((j + 1) * kk))) - 1]

    TPR = TP2.sum(0) / (TP.shape[0] - ki)
    FPR = FP2.sum(0) / (FP.shape[0] - ki)
    P = P2.sum(0) / (P.shape[0] - ki)

    return TPR, FPR, P

def curve(FPR,TPR,P):

    plt.figure()

    plt.subplot(121)
    plt.xlim(0.0,1.0)
    plt.ylim(0.0,1.0)

    plt.title("ROC curve  (AUC = %.4f)" % (auc(FPR, TPR)))
    plt.xlabel('FPR')
    plt.ylabel('TPR')

    plt.plot(FPR, TPR)
    plt.subplot(122)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)

    plt.title("PR curve  (AUC = %.4f)" % (auc(TPR,P) + TPR[0] * P[0]))

    plt.xlabel('TPR')
    plt.ylabel('FPR')

    plt.plot(TPR,P)
    plt.show()

if __name__ == '__main__':

    dis_sim, lnc_sim, lnc_dis, mi_dis, lnc_mi = ReadTxt()
    Score_lnc_dis = lnc_dis / 1

    Score_left = lnc_dis / 1
    Score_right = lnc_dis / 1

    positive_sample = Count_Value_1(lnc_dis, 5)
    negative_sample,all_negative_sample = Count_Value_0(lnc_dis, 5)

    print(type(positive_sample),len(positive_sample)) # <class 'list'>
    print(type(negative_sample),len(negative_sample)) # <class 'list'>
    print(type(all_negative_sample),all_negative_sample.shape) # <class 'numpy.ndarray'>

    Coordinate_Matrix_Train, Coordinate_Matrix_Test = Make_Train_Test_Set(positive_sample, negative_sample, all_negative_sample)

    print(Coordinate_Matrix_Train.shape,type(Coordinate_Matrix_Train))
    print(Coordinate_Matrix_Test.shape,type(Coordinate_Matrix_Test))

    # np.savetxt("../Result_3/Coordinate_Matrix_Train" + ".txt", Coordinate_Matrix_Train, fmt="%d", delimiter=",")
    # np.savetxt("../Result_3/Coordinate_Matrix_Test" + ".txt", Coordinate_Matrix_Test, fmt="%d", delimiter=",")

    LncRNA_Feature, Dise_Feature = Make_Tow_Graph_Feature(dis_sim, lnc_sim, lnc_dis, mi_dis, lnc_mi)
    LncRNA_Feature = torch.from_numpy(LncRNA_Feature).float()
    Dise_Feature = torch.from_numpy(Dise_Feature).float()

    # type: # <class 'torch.Tensor'> shape: torch.Size([240, 1140])
    #  # type: <class 'torch.Tensor'> shape:torch.Size([240, 1140])
    #  LncRNA_Feature = torch.from_numpy(LncRNA_Feature)
    #  Dise_Feature = torch.from_numpy(Dise_Feature)
    #  print('{}'.format(LncRNA_Feature)) # dtype=torch.float64

    g_LncRNA, g_Dise = Make_Tow_Graph(lnc_sim, dis_sim)

    learning_rate = 1e-2
    batch_size = 50
    num_epoches_right = 260
    num_epoches_left = 260
    r = 0.4

    input_size_lnc = 240
    input_size_A = 405
    input_size_lncmi = 495
    input_size_dis = 405
    input_size_midis = 495

    output_size1 = 240
    output_size2 = 405
    output_size3 = 495
    output_size4 = 240
    output_size5 = 405
    output_size6 = 495

    num_layers = 3
    in_dim = 1140
    num_hidden = 1000
    num_classes = 800
    heads = [2, 2, 2]
    activation = F.leaky_relu

    train_set = My_Dataset(dis_sim, lnc_sim, lnc_dis, mi_dis, lnc_mi, Coordinate_Matrix_Train)
    test_set = My_Dataset(dis_sim, lnc_sim, lnc_dis, mi_dis, lnc_mi, Coordinate_Matrix_Test)

    train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    if torch.cuda.is_available():
        model_right = Attention_model(input_size_lnc, input_size_A, input_size_lncmi, input_size_dis, input_size_midis,
                                output_size1, output_size2, output_size3, output_size4, output_size5, output_size6,
                                batch_size).cuda()
    else:
        model_right = Attention_model(input_size_lnc, input_size_A, input_size_lncmi, input_size_dis, input_size_midis,
                                output_size1, output_size2, output_size3, output_size4, output_size5, output_size6,
                                batch_size)

    criterion_right = nn.CrossEntropyLoss()
    optimizer_right = optim.SGD(model_right.parameters(), lr=learning_rate)

    model_right = model_right.train()

    for epoch in range(num_epoches_right):
        # print('{}'.format(epoch + 1))
        for step, (_, _, x, y) in enumerate(train_data):

            # print(type(x), x.shape) # <class 'torch.Tensor'> torch.Size([50, 2, 1140])
            # print(type(y), y.shape) # <class 'torch.Tensor'> torch.Size([50])

            # if (step + 1) % 50 == 0:
            #     print('{}，loss:{:.4f}'.format(step + 1,loss_right.item()))

            if torch.cuda.is_available():
                feature = Variable(x).float().cuda()
                feature = feature.unsqueeze(1)
                label = Variable(y).long().cuda()
            else:
                feature = Variable(x).float()
                feature = feature.unsqueeze(1)
                label = Variable(y).long()

            # print("进入模型前的",type(feature),feature.shape) #  <class 'torch.Tensor'> torch.Size([50, 1, 2, 1140])——float（）
            # print("进入模型前的",type(y),y.shape)<class 'torch.Tensor'> torch.Size([50])  ————long（）
            output_right = model_right(feature)

            loss_right = criterion_right(output_right, label)
            optimizer_right.zero_grad()
            loss_right.backward()
            optimizer_right.step()

    model_right.eval()
    label_list_right = []
    output_list_right = []

    for step, (_, _, x, y,) in enumerate(test_data):

        label_list_right.append(y.numpy())

        if torch.cuda.is_available():
            feature = Variable(x).float().cuda()
            feature = feature.unsqueeze(1)
            label = Variable(y).long().cuda()
        else:
            feature = Variable(x).float()
            feature = feature.unsqueeze(1)
            label = Variable(y).long()

        output_right = model_right(feature)
        output_right = F.softmax(output_right, dim=1)

        output_list_right.append(output_right.cpu().detach().numpy())

        loss_right = criterion_right(output_right, label)
        # if (step + 1) % 50 == 0:
        #     print('{}，loss:{:.4f}'.format(step + 1, loss_right.item()))

    outputs_right = vstack(output_list_right)
    labels_right = vstack(label_list_right)
    scores_right=list(zip(outputs_right[:,1],labels_right))

    if torch.cuda.is_available():
        model_left = My_model(g_LncRNA, g_Dise, num_layers, in_dim, num_hidden, num_classes, heads, activation,
                         LncRNA_Feature, Dise_Feature).cuda()
    else:
        model_left = My_model(g_LncRNA, g_Dise, num_layers, in_dim, num_hidden, num_classes, heads, activation,
                         LncRNA_Feature, Dise_Feature)

    criterion_left = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    optimizer_left = optim.SGD(model_left.parameters(), lr=learning_rate)

    model_left = model_left.train()
    for epoch in range(num_epoches_left):

        # print('{}'.format(epoch + 1))
        for step, (X, Y, _, label) in enumerate(train_data):
            # if (step + 1) % 50 == 0:
            #     print('{}，loss:{:.4f}'.format(step + 1, loss_left.item()))

            if torch.cuda.is_available():
                label = label.long().cuda()
            else:
                label = label.long()

            output_left = model_left(X, Y)  # <class 'torch.Tensor'> torch.Size([50, 2]) device='cuda:0', grad_fn=<LeakyReluBackward0>
            loss_left = criterion_left(output_left, label)
            optimizer_left.zero_grad()
            loss_left.backward()
            optimizer_left.step()

    model_left.eval()
    label_list_left = []
    output_list_left = []

    for step, (X, Y, _, label) in enumerate(test_data):

        label_list_left.append(label.numpy())

        if torch.cuda.is_available():
            label = label.long().cuda()
        else:
            label = label.long()

        with torch.no_grad():
            output_left = model_left(X, Y)
        output_left = F.softmax(output_left, dim=1)

        loss_left = criterion_left(output_left, label)

        # if (step + 1) % 50 == 0:
        #     print('{}，loss:{:.4f}'.format(step + 1, loss_left.item()))

        output_list_left.append(output_left.cpu().detach().numpy())

    outputs_left = vstack(output_list_left)
    labels_left = vstack(label_list_left)

    print( outputs_left.shape)
    scores_left = list(zip(outputs_left[:, 1], labels_left))

    for p in range(Coordinate_Matrix_Test.shape[0]):
        Score_left[Coordinate_Matrix_Test[p][0]][Coordinate_Matrix_Test[p][1]] = outputs_left[p][1]

    for q in range(Coordinate_Matrix_Test.shape[0]):
        Score_right[Coordinate_Matrix_Test[q][0]][Coordinate_Matrix_Test[q][1]] = outputs_right[q][1]

    for j in range(Coordinate_Matrix_Test.shape[0]):
        lnc_dis[Coordinate_Matrix_Test[j][0]][Coordinate_Matrix_Test[j][1]] = r * outputs_right[j][1] + (1-r) * outputs_left[j][1]

    for i in range(Coordinate_Matrix_Train.shape[0]):
        lnc_dis[Coordinate_Matrix_Train[i][0]][Coordinate_Matrix_Train[i][1]] = -1
        Score_lnc_dis[Coordinate_Matrix_Train[i][0]][Coordinate_Matrix_Train[i][1]] = -1

    f = Count_valid_data(lnc_dis)
    print(f)

    TPR,FPR,P = caculate_TPR_FPR(lnc_dis,f,Score_lnc_dis)
    curve(FPR, TPR, P)
    

