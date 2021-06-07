import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot
import seaborn as sns
import time
np.set_printoptions(suppress=True)
data=pd.read_excel(r"C:\Users\kevin\Desktop\pwc\geocode_result.xlsx")
gps=data[data.province=='gps']
circle=data[data.province!='gps']
city="成都市"
##
circlecity=circle[circle.city==city]
gpscity=gps[gps.city==city]
gps_ll=gpscity.values[:,-3:]
circle_ll=circlecity.values[:,-2:]
n=2*int(pow(circle_ll.shape[0],0.5))

clf = KMeans(n_clusters=n)
gpslabel=clf.fit_predict(gps_ll[:,1:3])
gpslabel=gpslabel.reshape(-1,1)
gps_ll=np.concatenate((gps_ll,gpslabel),axis=1)
gpscluster=clf.cluster_centers_

circlelabel=clf.fit_predict(circle_ll)
circlelabel=circlelabel.reshape(-1,1)
circle_ll=np.concatenate((circle_ll,circlelabel),axis=1)
circlecluster=clf.cluster_centers_
GMVmd=np.empty([n])
GMVsum=np.empty([n])
for i in range(n):
    mask=np.in1d(gps_ll[:,3],[i])
    gpsGMV=gps_ll[mask]
    GMVmd[i]=int(np.median(gpsGMV[:,0]))
    GMVsum[i]=int(np.sum(gpsGMV[:,0]))
GMVmd=GMVmd.reshape(-1,1)
GMVsum=GMVsum.reshape(-1,1)
gpscluster=np.concatenate((gpscluster,GMVmd),axis=1)
gpscluster=np.concatenate((gpscluster,GMVsum),axis=1)
print(gpscluster)
print('\n')
print(circlecluster)
