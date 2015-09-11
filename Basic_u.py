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
from lifelines.plotting import plot_lifetimes
from numpy.random import uniform, exponential

#variable---------------------------------------------------------------------------------
User_dic = []
D = defaultdict(int)
N = defaultdict(int)
#各場所における入出時刻 D[hall][point]
max_time = -10 ** 14
min_time = 10 ** 14
#-----------------------------------------------------------------------------------------


#Input------------------------------------------------------------------------------------
def input():
    global max_time, min_time
    t0 = time()
    filename = "Data/5min/hall_ap_id_st_ed_5minT.csv"

    fin = open(filename)
    for row in fin:
        temp = row.rstrip("\r\n").split(",")
        for j in range(len(temp)):
            if j == 2:
                User = int(temp[j])
                if User not in User_dic:
                    User_dic.append(User)

        s_time = int((dt.datetime.strptime(temp[3],'%Y/%m/%d %H:%M:%S') - dt.datetime(1899,12,31)).seconds)
        e_time = int((dt.datetime.strptime(temp[4],'%Y/%m/%d %H:%M:%S') - dt.datetime(1899,12,31)).seconds)
        #経過時間（秒）に変換

        if min_time > s_time:
            min_time = s_time
        if max_time < e_time:
            max_time = e_time

        delta = e_time - s_time
        if np.size(D[User]) == 1:
#            if delta != 0:
            d = np.array([s_time, e_time, delta])
            D[User] = d

        else:
#            if delta != 0:
            d = np.array([s_time, e_time, delta])
            D[User] = np.vstack([D[User],d])

    fin.close()
    print "Input time:%f" % (time()-t0)
#-----------------------------------------------------------------------------------------

#birth_series-----------------------------------------------------------------------------
def birth_series():
    NX = max_time - min_time + 1
    X = np.arange(0, NX)
    X = X * 0.0001
    x = np.concatenate( (X,X[::-1]) )
    Y = np.zeros(np.size(X))

    counter = 0
    counter2 = 0
    for h in range(len(Hall_dic)):
        Hall = Hall_dic[h]
        SubPoint = Point_dic[Hall]
        for p in range(len(SubPoint)):
            Point = SubPoint[p]
            SubD = D[Hall][Point]

            I = np.shape(SubD)[0]
            N[Hall][Point] = I

            Y2 = np.zeros(np.size(X))
            for i in range(I):
                T = SubD[i][0] - min_time
                Y2[T] = 0.4
            y = np.concatenate( (Y,Y2[::-1]) )

            if counter == 0:
                fig = plt.figure(figsize=(16,9))
                plt.subplots_adjust(hspace=1.0)
                ax1 = fig.add_subplot(611)
                ax2 = fig.add_subplot(612)
                ax3 = fig.add_subplot(613)
                ax4 = fig.add_subplot(614)
                ax5 = fig.add_subplot(615)
                ax6 = fig.add_subplot(616)

                p = ax1.fill(x,y, facecolor = 'k')#, edgecolor = 'none')
                ax1.set_title('Hall' + str(Hall) + '  Point' + str(Point))
                ax1.axis('off')
                counter += 1
            elif counter == 1:
                p = ax2.fill(x,y, facecolor = 'k')#, edgecolor = 'none')
                ax2.set_title('Hall' + str(Hall) + '  Point' + str(Point))
                ax2.axis('off')
                counter += 1
            elif counter == 2:
                p = ax3.fill(x,y, facecolor = 'k')#, edgecolor = 'none')
                ax3.set_title('Hall' + str(Hall) + '  Point' + str(Point))
                ax3.axis('off')
                counter += 1
            elif counter == 3:
                p = ax4.fill(x,y, facecolor = 'k')#, edgecolor = 'none')
                ax4.set_title('Hall' + str(Hall) + '  Point' + str(Point))
                ax4.axis('off')
                counter += 1
            elif counter == 4:
                p = ax5.fill(x,y, facecolor = 'k')#, edgecolor = 'none')
                ax5.set_title('Hall' + str(Hall) + '  Point' + str(Point))
                ax5.axis('off')
                counter += 1
            elif counter == 5:
                p = ax6.fill(x,y, facecolor = 'k')#, edgecolor = 'none')
                ax6.set_title('Hall' + str(Hall) + '  Point' + str(Point))
                ax6.axis('off')
                counter = 0
                plt.savefig('birth_series/birth_series' + str(counter2) + '.png')
                counter2 += 1
                plt.close()
#-----------------------------------------------------------------------------------------

#death_series-----------------------------------------------------------------------------
def death_series():
    NX = max_time - min_time + 1
    X = np.arange(0, NX)
    X = X * 0.0001
    x = np.concatenate( (X,X[::-1]) )
    Y = np.zeros(np.size(X))

    counter = 0
    counter2 = 0
    for h in range(len(Hall_dic)):
        Hall = Hall_dic[h]
        SubPoint = Point_dic[Hall]
        for p in range(len(SubPoint)):
            Point = SubPoint[p]
            SubD = D[Hall][Point]

            I = np.shape(SubD)[0]
            N[Hall][Point] = I

            Y2 = np.zeros(np.size(X))
            for i in range(I):
                T = SubD[i][1] - min_time
                Y2[T] = 0.4
            y = np.concatenate( (Y,Y2[::-1]) )

            if counter == 0:
                fig = plt.figure(figsize=(16,9))
                plt.subplots_adjust(hspace=1.0)
                ax1 = fig.add_subplot(611)
                ax2 = fig.add_subplot(612)
                ax3 = fig.add_subplot(613)
                ax4 = fig.add_subplot(614)
                ax5 = fig.add_subplot(615)
                ax6 = fig.add_subplot(616)

                p = ax1.fill(x,y, facecolor = 'k')#, edgecolor = 'none')
                ax1.set_title('Hall' + str(Hall) + '  Point' + str(Point))
                ax1.axis('off')
                counter += 1
            elif counter == 1:
                p = ax2.fill(x,y, facecolor = 'k')#, edgecolor = 'none')
                ax2.set_title('Hall' + str(Hall) + '  Point' + str(Point))
                ax2.axis('off')
                counter += 1
            elif counter == 2:
                p = ax3.fill(x,y, facecolor = 'k')#, edgecolor = 'none')
                ax3.set_title('Hall' + str(Hall) + '  Point' + str(Point))
                ax3.axis('off')
                counter += 1
            elif counter == 3:
                p = ax4.fill(x,y, facecolor = 'k')#, edgecolor = 'none')
                ax4.set_title('Hall' + str(Hall) + '  Point' + str(Point))
                ax4.axis('off')
                counter += 1
            elif counter == 4:
                p = ax5.fill(x,y, facecolor = 'k')#, edgecolor = 'none')
                ax5.set_title('Hall' + str(Hall) + '  Point' + str(Point))
                ax5.axis('off')
                counter += 1
            elif counter == 5:
                p = ax6.fill(x,y, facecolor = 'k')#, edgecolor = 'none')
                ax6.set_title('Hall' + str(Hall) + '  Point' + str(Point))
                ax6.axis('off')
                counter = 0
                plt.savefig('death_series/death_series' + str(counter2) + '.png')
                counter2 += 1
                plt.close()
#-----------------------------------------------------------------------------------------

#survival---------------------------------------------------------------------------------
def survival():
    for u in range(len(User_dic)):
        User = User_dic[u]
        SubD = D[User]
        
        flag = len(np.shape(SubD))
        if flag == 1:
            N[User] = 1
            T = SubD[0] - min_time
            S = SubD[2]
        elif flag == 2:
            I = np.shape(SubD)[0]
            N[User] = I
            T = np.zeros(np.shape(SubD)[0])
            S = SubD[:,2]
            for i in range(I):
                T[i] = SubD[i,0] - min_time
        else:
            print 'err'

        fig = plt.figure(figsize=(16,9))
        plt.title('User' + str(User))
        plt.xlabel('time')
        plot_lifetimes(S, birthtimes=T)
        fig.savefig('survival_u/survival_User' + str(User) + '.png')

input()
#birth_series()
#death_series()
#survival()

pdb.set_trace()
