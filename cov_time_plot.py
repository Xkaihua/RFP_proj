# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 13:49:54 2021

@author: xukai
"""
import numpy as np
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl

from rfp_fun import * 

############
# identify and plot the RFPs of cov-obs.x1 model

pwd=os.getcwd()

#fd=r'\cov_obsx1_coefs_mat'
#fd=r'\cov_obsx1_average_coefs_mat'
fd=r'\cov_obsx1_coefs_100_mat'     # fold that stores the gauss coef of cov-obs.x1 model
fL=os.listdir(pwd+fd)

theta=np.arange(0,180,0.45)
phi=np.arange(0,360,0.45)            # grid of spherical surface
[phim,thetam]=np.meshgrid(phi,theta)

lat_w=np.around(theta[1]-theta[0],3)
lon_w=np.around(phi[1]-phi[0],3)      

lat=90-theta
lon=phi
[lonm,latm]=np.meshgrid(lon,lat)

# a=6371
r=3480

L1=1
L2=4
L3=3                       # max degree for finding magnetic equator
L_f=14                    # max degree of the field 

Br_me_L1=np.zeros((len(theta),len(phi)))
Br_me_L2=np.zeros((len(theta),len(phi)))
Br_me_L3=np.zeros((len(theta),len(phi)))
Br_me_lf=np.zeros((len(theta),len(phi)))  #sum of Br of 100 ensembles
for i in np.arange(len(fL)):
    mL=os.listdir(pwd+fd+r'\\'+fL[i])
    gmodel=loadmat(pwd+fd+r'\\'+fL[i]+r'\\'+mL[60]) #choose epoch 1900 
    g=gmodel['g']
    h=gmodel['h']
    Brm_L1=get_Br(L1, phim, thetam, g, h,r)
    Br_me_L1=Br_me_L1+Brm_L1
    
    Brm_L2=get_Br(L2, phim, thetam, g, h,r)
    Br_me_L2=Br_me_L2+Brm_L2
    
    Brm_L3=get_Br(L3, phim, thetam, g, h,r)
    Br_me_L3=Br_me_L3+Brm_L3
    
    Brm_lf=get_Br(L_f, phim, thetam, g, h,r)
    Br_me_lf=Br_me_lf+Brm_lf

# gmodel=loadmat(pwd+fd+r'\\'+fL[121])
# Br_me_L1=get_Br(L1, phim, thetam, g, h,r)
# Br_me_L2=get_Br(L2, phim, thetam, g, h,r)
# Br_me_L3=get_Br(L3, phim, thetam, g, h,r)
# Br_me_lf=get_Br(L_f, phim, thetam, g, h,r)

tname=mL[60][0:-4]


fig=plt.figure()
ax=fig.add_subplot(1,1,1)
map=Basemap(projection='hammer',lon_0=0,resolution='c')
map.drawcoastlines(linewidth=0.3)
map.drawparallels(np.arange(-60,90,30),labels=[0,0,0,0],linewidth=0.3)
map.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],linewidth=0.3)
nr=mpl.colors.Normalize(vmin=-1e6, vmax=1e6)
p=map.pcolormesh(lonm,latm,Br_me_lf/len(fL),norm=nr, cmap='bwr',latlon='True')
cbar=map.colorbar(p,'bottom')
cbar.set_label('nT')
ax.set_title(tname)  

[mequt_lat,mequt_lon]=idf_mequt(Br_me_L3/len(fL),phi,theta)
[rfp_lon,rfp_lat,e_lon,e_lat]=idf_rfp(mequt_lon,mequt_lat,Br_me_lf/len(fL),phi,theta)
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
map=Basemap(projection='hammer',lon_0=0,resolution='c')
map.drawcoastlines(linewidth=0.3)
map.drawparallels(np.arange(-60,90,30),labels=[0,0,0,0],linewidth=0.3)
map.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],linewidth=0.3)
map.scatter(rfp_lon,90-rfp_lat,color='r',s=1,latlon=True)
map.scatter(mequt_lon,90-mequt_lat,s=1,color='b',latlon=True)
ax.set_title('l_eq_max=3')  

[mequt_lat,mequt_lon]=idf_mequt(Br_me_L1/len(fL),phi,theta)
[rfp_lon,rfp_lat,e_lon,e_lat]=idf_rfp(mequt_lon,mequt_lat,Br_me_lf/len(fL),phi,theta)
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
map=Basemap(projection='hammer',lon_0=0,resolution='c')
map.drawcoastlines(linewidth=0.3)
map.drawparallels(np.arange(-60,90,30),labels=[0,0,0,0],linewidth=0.3)
map.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],linewidth=0.3)
map.scatter(rfp_lon,90-rfp_lat,color='r',s=1,latlon=True)
map.scatter(mequt_lon,90-mequt_lat,s=1,color='b',latlon=True)
ax.set_title('l_eq_max=1') 

[mequt_lat,mequt_lon]=idf_mequt(Br_me_L2/len(fL),phi,theta)
[rfp_lon,rfp_lat,e_lon,e_lat]=idf_rfp(mequt_lon,mequt_lat,Br_me_lf/len(fL),phi,theta)
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
map=Basemap(projection='hammer',lon_0=0,resolution='c')
map.drawcoastlines(linewidth=0.3)
map.drawparallels(np.arange(-60,90,30),labels=[0,0,0,0],linewidth=0.3)
map.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],linewidth=0.3)
map.scatter(rfp_lon,90-rfp_lat,color='r',s=0.5,latlon=True)
map.scatter(mequt_lon,90-mequt_lat,s=0.1,color='b',latlon=True)
ax.set_title('l_eq_max=4') 

[mequt_lat,mequt_lon]=idf_mequt(Br_me_lf/len(fL),phi,theta)
[rfp_lon,rfp_lat,e_lon,e_lat]=idf_rfp(mequt_lon,mequt_lat,Br_me_lf/len(fL),phi,theta)
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
map=Basemap(projection='hammer',lon_0=0,resolution='c')
map.drawcoastlines(linewidth=0.3)
map.drawparallels(np.arange(-60,90,30),labels=[0,0,0,0],linewidth=0.3)
map.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],linewidth=0.3)
map.scatter(rfp_lon,90-rfp_lat,color='r',s=0.5,latlon=True)
map.scatter(mequt_lon,90-mequt_lat,s=0.1,color='b',latlon=True)
ax.set_title('l_eq_max=14') 

mequt_lon=phi
mequt_lat=90*np.ones(len(phi))
[rfp_lon,rfp_lat,e_lon,e_lat]=idf_rfp(mequt_lon,mequt_lat,Br_me_lf/len(fL),phi,theta)
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
map=Basemap(projection='hammer',lon_0=0,resolution='c')
map.drawcoastlines(linewidth=0.3)
map.drawparallels(np.arange(-60,90,30),labels=[0,0,0,0],linewidth=0.3)
map.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],linewidth=0.3)
map.scatter(rfp_lon,90-rfp_lat,color='r',s=0.5,latlon=True)
map.scatter(mequt_lon,90-mequt_lat,s=0.1,color='b',latlon=True)
ax.set_title('geographic equator') 

