# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 16:39:18 2021

@author: xukai
"""
import copy as cp
import numpy as np
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import OPTICS
from rfp_fun import * 

######################################################
##track the RFPs
#######

gmf_l=r'/gufm_coef_mat'                  #fold of gufm gauss coef mat files

flist=os.listdir(os.getcwd()+gmf_l)
Lme=3                                     #max degree used to identify the magnetic equator
Lf=14                                     #max degree used to calculate the field
r=3480                                     # radius of CMB
theta=np.arange(0.45,180,0.45)
phi=np.arange(0.,360,0.45)                    # grids of spherical surface
lat_w=np.around(theta[1]-theta[0],3)
lon_w=np.around(phi[1]-phi[0],3)
[phim,thetam]=np.meshgrid(phi,theta)



gmodel=loadmat(os.getcwd()+gmf_l+'/'+flist[0])    #load gauss coefficients
g=gmodel['g']
h=gmodel['h']

Br_me=get_Br(Lme, phim, thetam, g, h,r)
Br_f=get_Br(Lf, phim, thetam, g, h,r)

[mequt_lat,mequt_lon]=idf_mequt(Br_me,phi,theta)    
[rfp_lon,rfp_lat,e_lon,e_lat]=idf_rfp(mequt_lon,mequt_lat,Br_f,phi,theta)

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
map=Basemap(projection='hammer',lon_0=0,resolution='c')
map.drawcoastlines(linewidth=0.3)
map.drawparallels(np.arange(-60,90,30),labels=[0,0,0,0],linewidth=0.3)
map.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,0],linewidth=0.3)
map.scatter(rfp_lon,90-rfp_lat,color='r',s=1,latlon=True)
map.scatter(mequt_lon,90-mequt_lat,s=1,color='b',latlon=True)
ax.set_title('l_eq_max=3') 



######################## initial the track of rfps
k=0
u_lon_c=np.zeros(len(phi))
u_lon_o=np.zeros(len(phi))
for i in np.arange(len(phi)):
    u_lon_c[k]=np.sum(np.around(rfp_lon,5)==np.around(phi[i],5))
    u_lon_o[k]=phi[i]
    k=k+1
    
sc=np.argsort(u_lon_c)
stl=u_lon_o[sc][0]
n_rfp_lon=rfp_lon-stl
[inrf]=np.where(n_rfp_lon<0)
n_rfp_lon[inrf]=n_rfp_lon[inrf]+360  #transfer the original point of longitude of the lon-lat plane coordinate
  
X=np.zeros((len(rfp_lon),2))
X[:,0]=n_rfp_lon
X[:,1]=90-rfp_lat                 # transferred reversed patches points for clustering

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(X[:,0],X[:,1],s=1,color='r')


coloM=['xkcd:blue','xkcd:orange','xkcd:green','xkcd:red','xkcd:pink','xkcd:brown','xkcd:purple','xkcd:teal','xkcd:light blue','xkcd:light green','xkcd:magenta','xkcd:yellow','xkcd:grey']

clust = OPTICS(min_samples=50, xi=.05, min_cluster_size=5)
clust.fit(X)                       #  cluster the reversed points with optics
cl_label=clust.labels_
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
tlst_lon=[]
tlst_lat=[]
for klass, cr in zip(range(0, max(cl_label)+1), coloM):
    Xk = X[clust.labels_ == klass]
    tlst_lon=np.append(tlst_lon,np.mean(Xk[:,0]))
    tlst_lat=np.append(tlst_lat,np.mean(Xk[:,1]))
    ax.scatter(Xk[:, 0], Xk[:, 1],color=cr,s=1)

tlst_lon=tlst_lon+stl
tlst_lon[tlst_lon>360]=tlst_lon[tlst_lon>360]-360
tlst_lab=np.arange(0,len(tlst_lon))


###################  examples of tracking the rfps
pcla=[]            # latitude of the centers of the patches
pclo=[]            #longitude of the center of the patches
labc=[]            # label of the centers
yrfp=[]            # time point of the centers

for i in np.arange(30):
    gmodel=loadmat(os.getcwd()+gmf_l+'/'+flist[i])
    g=gmodel['g']
    h=gmodel['h']
    
    Br_me=get_Br(Lme, phim, thetam, g, h,r)
    Br_f=get_Br(Lf, phim, thetam, g, h,r)
    
    [mequt_lat,mequt_lon]=idf_mequt(Br_me,phi,theta)    
    [rfp_lon,rfp_lat,e_lon,e_lat]=idf_rfp(mequt_lon,mequt_lat,Br_f,phi,theta)
    
    k=0
    u_lon_c=np.zeros(len(phi))
    u_lon_o=np.zeros(len(phi))
    for j in np.arange(len(phi)):
        u_lon_c[k]=np.sum(np.around(rfp_lon,5)==np.around(phi[j],5))
        u_lon_o[k]=phi[j]
        k=k+1
    sc=np.argsort(u_lon_c)
    
    stl=u_lon_o[sc][0]
    
    n_rfp_lon=rfp_lon-stl
    [inrf]=np.where(n_rfp_lon<0)
    n_rfp_lon[inrf]=n_rfp_lon[inrf]+360
      
    X=np.zeros((len(rfp_lon),2))
    X[:,0]=n_rfp_lon
    X[:,1]=90-rfp_lat
    
    ##### optics
    coloM=['xkcd:blue','xkcd:orange','xkcd:green','xkcd:red','xkcd:pink','xkcd:brown','xkcd:purple','xkcd:teal','xkcd:light blue','xkcd:light green','xkcd:magenta','xkcd:yellow','xkcd:grey']
    clust = OPTICS(min_samples=50, xi=.05, min_cluster_size=5)
    clust.fit(X)
    cl_label=clust.labels_
    # fig=plt.figure()
    # ax=fig.add_subplot(1,1,1)
    tmp_lat=[]
    tmp_lon=[]
    tmp_lab=[]
    for klass, cr in zip(range(0, max(cl_label)+1), coloM):
        Xk = X[clust.labels_ == klass]
        
        tf_lon=np.mean(Xk[:,0])+stl
        if tf_lon>360:
            tf_lon=tf_lon-360
            
        d_lon=abs(tlst_lon-tf_lon)
        d_lon[d_lon>180]=d_lon[d_lon>180]-360
        dist=d_lon**2+(tlst_lat-np.mean(Xk[:,1]))**2
        
        tmp_lat=np.append(tmp_lat,np.mean(Xk[:,1]))
        tmp_lon=np.append(tmp_lon,tf_lon)
        tmp_lab=np.append(tmp_lab,tlst_lab[np.argmin(dist)])
        labc=np.append(labc,tlst_lab[np.argmin(dist)])
        yrfp=np.append(yrfp,float(flist[i][0:-4]))
#        ax.scatter(tf_lon, np.mean(Xk[:, 1]),color=coloM[np.argmin(dist)],s=20)
#    ax.plot(X[clust.labels_ == -1,0], X[clust.labels_ == -1,1], 'k+')
#    ax.set_xlim(-10,370)
#    ax.set_ylim(-55,30)


    tlst_lon=cp.deepcopy(tmp_lon)
    tlst_lat=cp.deepcopy(tmp_lat)
    tlst_lab=cp.deepcopy(tmp_lab)
    pclo=np.append(pclo,tmp_lon)
    pcla=np.append(pcla,tmp_lat)
    print(i)

#############################
# plot time evolution of the rfp centers    
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
for uk, cr in zip(range(10), coloM):
    pcx = pclo[labc == uk]
    pyr = yrfp[labc == uk]
    ax.plot(pyr,pcx,color=cr)
ax.set_title('lon')
ax.set_xlim(1600,1650)
    
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
for uk, cr in zip(range(10), coloM):
    pyr = yrfp[labc == uk]
    pcy = pcla[labc == uk]
    ax.plot(pyr,pcy,color=cr)
ax.set_title('lat')
ax.set_xlim(1600,1650)

######################
# plot the track of the rfp centers in map    
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
map=Basemap(projection='hammer',lon_0=0,resolution='c')
map.drawcoastlines(linewidth=0.3)
map.drawparallels(np.arange(-60,90,30),labels=[0,0,0,0],linewidth=0.3)
map.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,0],linewidth=0.3)
for uk, cr in zip(np.unique(labc), coloM):
    pcx, pcy = map(pclo[labc == uk][0:50], pcla[labc == uk][0:50])
    map.plot(pcx,pcy,color=cr)