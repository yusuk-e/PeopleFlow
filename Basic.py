# -*- coding:utf-8 -*-

import pdb
from time import time
import datetime as dt
import numpy as np
from scipy.special import gammaln
import resource
import codecs
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#variable---------------------------------------------------------------------------------
Hall_dic = []
Point_dic = []
D = defaultdict(lambda: defaultdict(int))
#各場所における入出時刻 D[hall][point]
max_time = -10 ** 14
min_time = 10 ** 14
#-----------------------------------------------------------------------------------------


#Input------------------------------------------------------------------------------------
t0 = time()
filename = "Data/5min/hall_ap_id_st_ed_5minT.csv"

fin = open(filename)
for row in fin:
    temp = row.rstrip("\r\n").split(",")
    for j in range(len(temp)):
        if j == 0:
            Hall = int(temp[j])
            if Hall not in Hall_dic:
                Hall_dic.append(Hall)
        elif j == 1:
            Point = int(temp[j])
            if Point not in Point_dic:
                Point_dic.append(Point)

    s_time = int((dt.datetime.strptime(temp[3],'%Y/%m/%d %H:%M:%S') - dt.datetime(1899,12,31)).seconds)
    e_time = int((dt.datetime.strptime(temp[4],'%Y/%m/%d %H:%M:%S') - dt.datetime(1899,12,31)).seconds)
    #経過時間（秒）に変換

    if min_time > s_time:
        min_time = s_time
    if max_time < e_time:
        max_time = e_time

    if np.size(D[Hall][Point]) == 1:
        d = np.array([s_time, e_time, e_time - s_time])
        D[Hall][Point] = d

    else:
        d = np.array([s_time, e_time, e_time - s_time])
        D[Hall][Point] = np.vstack([D[Hall][Point],d])

fin.close()
print "Input time:%f" % (time()-t0)
#-----------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------
NX = max_time - min_time + 1
X = np.arange(0, NX)
X = X * 0.0001
x = np.concatenate( (X,X[::-1]) )
Y = np.zeros(np.size(X))

SubD = D[2][1]
I = np.shape(SubD)[0]
Y2 = np.zeros(np.size(X))
for i in range(I):
    T = SubD[i][0] - min_time
    Y2[T] = 0.4
y = np.concatenate( (Y,Y2[::-1]) )

fig = plt.figure()
plt.fill(x,y, facecolor = 'k')#, edgecolor = 'none')







'''
Group_dic_inv = dict([(v,k) for k,v in Group_dic.items()])
Item_dic_inv = dict([(v,k) for k,v in Item_dic.items()])

MAX_TIME = max(D[:,4])
#-----------------------------------------------------------------------------------------
View_Time_th = 30
D_View_Trend = D[np.where(D[:,4] > (MAX_TIME - View_Time_th))[0],:]
P_View = defaultdict(float)
for i in range(np.shape(D_View_Trend)[0]):
    Item_id = D_View_Trend[i,2]
    P_View[int(Item_id)] += 1

Recommend_Item = np.zeros([100,2])
counter = 0
counter2 = 0
A = sorted(P_View.items(), key=lambda x:x[1], reverse=True)
for i in range(1000000):
    Item_id = A[i][0]
    Item_value = A[i][1]
    if int(Item_dic_inv[Item_id]) not in RECOMMEND_Possi:
        counter2 += 1
    elif int(Item_dic_inv[Item_id]) in RECOMMEND_Possi:
        Recommend_Item[counter,0] = Item_id
        Recommend_Item[counter,1] = Item_value
        counter += 1
    if counter == 100:
        break
Recommend_Item[:,1] = Recommend_Item[:,1] / np.sum(Recommend_Item[:,1])
pdb.set_trace()

Output = np.zeros([1,3])
for Group_id in Group_dic.itervalues():
    G = np.zeros([100,1])
    G[:] = Group_id
    temp = np.hstack((G,Recommend_Item))
    Output = np.vstack((Output,temp))
Output = np.delete(Output,0,0)

D_Buy = D[np.where(D[:,3] == 1)[0],:]
Buy_Time_th = 90
D_Buy_Trend = D[np.where(D[:,4] > (MAX_TIME - Buy_Time_th))[0],:]
P_Buy = defaultdict(lambda: defaultdict(float))
N_Buy = defaultdict(float)
for i in range(np.shape(D_Buy_Trend)[0]):
    Group_id = D_Buy_Trend[i,0]
    Item_id = D_Buy_Trend[i,2]
    if Item_id in Recommend_Item[:,0]:
        P_Buy[int(Group_id)][int(Item_id)] += 1
        N_Buy[int(Group_id)] += 1

counter = 0
NewOutput = np.zeros([np.shape(Output)[0],4])
for Group_id in Group_dic.itervalues():
    for Item_id in Item_dic.itervalues():
        if Item_id in Recommend_Item[:,0]:
            NewOutput[counter,0] = int(Group_dic_inv[Output[counter,0]])
            NewOutput[counter,1] = int(Item_dic_inv[Output[counter,1]])
            NewOutput[counter,2] = Output[counter,2]
            NewOutput[counter,3] = P_Buy[Group_id][Item_id] / N_Buy[Group_id]
            counter += 1

index = np.unique(NewOutput[:,0])
N2Output = np.zeros([1,4])
for i in range(len(index)):
    N2Output = np.vstack((N2Output,NewOutput[np.where(NewOutput[:,0] == index[i])[0],:]))
N2Output = np.delete(N2Output,0,0)

filename = "Output.csv"
fout = open(filename,'w')
for i in range(np.shape(N2Output)[0]):
    fout.write(str(int(N2Output[i,0])))
    fout.write(",")
    fout.write(str(int(N2Output[i,1])))
    fout.write(",")
    fout.write(str(N2Output[i,2]))
    fout.write(",")
    fout.write(str(N2Output[i,3]))
    fout.write("\n")
fout.close()
'''                   


pdb.set_trace()
