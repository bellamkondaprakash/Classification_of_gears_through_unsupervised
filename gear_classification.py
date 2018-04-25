import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(5)

td = pd.read_csv("147.csv",header=None)
col = ["columnsID", "Speed","RPM", "EngineLoad"]
td.columns = col


data       = td[["Speed","RPM","EngineLoad"]]
speed      = np.array(td['Speed'])
RPM        = np.array(td['RPM'])
EngineLoad = np.array(td['EngineLoad'])
data       = np.array(data)

nbrs = NearestNeighbors(n_neighbors=6).fit(data)
estimators = [('k_means_iris_6', KMeans(n_clusters=6))]

fignum = 1
titles = ['6 clusters']
for name, est in estimators:
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=60, azim=150)
    est.fit(data)
    labels = est.labels_

    ax.scatter(speed,RPM,EngineLoad,c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Speed')
    ax.set_ylabel('RPM')
    ax.set_zlabel('EngineLoad')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fignum = fignum + 1
    plt.savefig('sample.png')

import seaborn as sns
sns.set(style="ticks", color_codes=True)
label = est.labels_
lb    = pd.DataFrame(label,columns=["lable"])

hold=[]
for i in range(len(speed)):
    hold.append([speed[i],RPM[i],EngineLoad[i],label[i]])

colx = ["Speed","RPM", "EngineLoad","lable"]
new_dataframe = pd.DataFrame(hold,columns=colx)
dix = {5:'lb-5',4:'lb-4',1:"lb-1",3:'lb-3',0:'lb-0',2:'lb-2'}
new_dataframe['lable']=new_dataframe['lable'].map(dix)
sns.pairplot(new_dataframe,hue='lable')
plt.savefig('comparasim_speed_rpm_engineload.png')

gear = {'lb-0':'gear-3','lb-1':'gear-1','lb-2':'reverse','lb-3':'gear-2',
       'lb-4':'gear-5','lb-5':'gear-4'}

new_dataframe['lable'] = new_dataframe['lable'].map(gear)
new_dataframe.to_csv('gear_classification.csv')