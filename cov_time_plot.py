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

pwd=os.getcwd()
gmf=r'\gufm_coef_mat\1930'

##################### 
# test the function
#####################
gmpath=pwd+gmf  
# gmpath=r'E:\CAS\dipole_simulate\igrf_coe\2015'
gmodel=loadmat(gmpath)
g=gmodel['g']
h=gmodel['h']

theta=np.arange(0,180,0.45)
phi=np.arange(0,360,0.45)
lat_w=np.around(theta[1]-theta[0],3)
lon_w=np.around(phi[1]-phi[0],3)

# a=6371
r=3480

L=3                       # max degree for finding equator
L_f=14                    # max degree of the field 
[phim,thetam]=np.meshgrid(phi,theta)

Brm=get_Br(L, phim, thetam, g, h,r)
lat=90-theta
lon=phi
[lonm,latm]=np.meshgrid(lon,lat)
fig=plt.figure
map=Basemap(projection='hammer',lon_0=0,resolution='c')
map.drawcoastlines()
p=map.pcolormesh(lonm,latm,Brm,cmap='bwr',latlon='True')
cbar=map.colorbar(p,'right')
cbar.set_label('nT')

[mequt_lat,mequt_lon]=idf_mequt(Brm,phi,theta)
fig=plt.figure()
map=Basemap(projection='hammer',lon_0=0,resolution='c')
map.drawcoastlines()
p=map.pcolormesh(lonm,latm,Brm,cmap='bwr',latlon='True')
cbar=map.colorbar(p,'right')
cbar.set_label('nT')
map.scatter(mequt_lon,90-mequt_lat,s=1,latlon='True')

Brm_l10=get_Br(L_f, phim, thetam, g, h,r)
[rfp_lon,rfp_lat]=idf_rfp(mequt_lon,mequt_lat,Brm_l10,phi,theta)
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
map=Basemap(projection='hammer',lon_0=0,resolution='c')
map.drawcoastlines()
map.scatter(rfp_lon,90-rfp_lat,color='r',s=1,latlon=True)
map.scatter(mequt_lon,90-mequt_lat,s=1,color='b',latlon=True)
ax.set_title(gmf[-4:])  

###########################
# im=np.argsort(mequt_lon)
# mequt_lon=mequt_lon[im]
# mequt_lat=mequt_lat[im]
# rfp_lat_st=[]
# rfp_lat_nt=[]
# rfp_lon_st=[]
# rfp_lon_nt=[]
# for i in np.arange(len(mequt_lon)):
# #for i in np.arange(180,181):

#     [i_lat]=np.where(np.around(theta,5)==np.around(mequt_lat[i],5))
#     [i_lon]=np.where(np.around(phi,5)==np.around(mequt_lon[i],5))

#     if i_lon.size==0:
#         print(mequt_lon[i])

#     t_lat=i_lat[0]
#     t_lon=i_lon[0]
#     Br_north=Brm_l10[0:t_lat,t_lon]
#     Br_south=Brm_l10[t_lat+1:,t_lon]
#     [i_n]=np.where(Br_north>0)
#     [i_s]=np.where(Br_south<0)
#     rfp_lat_nt=np.append(rfp_lat_nt,theta[0:t_lat][i_n])
#     rfp_lat_st=np.append(rfp_lat_st,theta[t_lat+1:][i_s])
#     rfp_lon_nt=np.append(rfp_lon_nt,mequt_lon[i]*np.ones(len(i_n)))
#     rfp_lon_st=np.append(rfp_lon_st,mequt_lon[i]*np.ones(len(i_s)))


#[rfp_lon_nt,rfp_lat_nt,rfp_lon_st,rfp_lat_st]=idf_rfp_test(mequt_lon,mequt_lat,Brm_l10,phi,theta)
# fig=plt.figure()
# ax=fig.add_subplot(1,1,1)
# map=Basemap(projection='hammer',lon_0=0,resolution='c')
# map.drawcoastlines()
# map.scatter(rfp_lon_nt,90-rfp_lat_nt,s=1,color='r',latlon=True)
# map.scatter(rfp_lon_st,90-rfp_lat_st,s=1,color='r',latlon=True)
# map.scatter(mequt_lon,90-mequt_lat,s=1,color='k',latlon=True)
# ax.set_title(gmf[-4:])  

# fig=plt.figure()
# ax=fig.add_subplot(1,1,1)
# ax.scatter(rfp_lon,90-rfp_lat,color='b',s=1)
# ax.scatter(rfp_lon_nt,90-rfp_lat_nt,s=1,color='r')
# ax.scatter(rfp_lon_st,90-rfp_lat_st,s=1,color='r')
# ax.set_title(gmf[-4:])  

# fig=plt.figure()
# ax=fig.add_subplot(1,1,1)
# ax.scatter(rfp_lon_nt,90-rfp_lat_nt,s=1,color='r')
# ax.scatter(rfp_lon_st,90-rfp_lat_st,s=1,color='r')
# ax.set_title(gmf[-4:])  

# print(len(rfp_lon_st)+len(rfp_lon_nt))

# area_temp=get_area2rfp(rfp_lat,lat_w,lon_w)
# s_rate=(area_temp)/(4*np.pi*3480*3480)
# print(s_rate)

#####################
pwd=os.getcwd()
#gmf=r'\cals3k.4_coef_mat\1980'
gmf=r'\gufm_coef_mat\1840'
#fd=r'\cov_obsx1_coefs_mat'
#fd=r'\cov_obsx1_average_coefs_mat'
fd=r'\cov_obsx1_coefs_100_mat'
fL=os.listdir(pwd+fd)

theta=np.arange(0,180,0.45)
phi=np.arange(0,360,0.45)
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
L3=3                       # max degree for finding equator
L_f=14                    # max degree of the field 

Br_me_L1=np.zeros((len(theta),len(phi)))
Br_me_L2=np.zeros((len(theta),len(phi)))
Br_me_L3=np.zeros((len(theta),len(phi)))
Br_me_lf=np.zeros((len(theta),len(phi)))
for i in np.arange(len(fL)):
    mL=os.listdir(pwd+fd+r'\\'+fL[i])
    gmodel=loadmat(pwd+fd+r'\\'+fL[i]+r'\\'+mL[60])
    g=gmodel['g']
    h=gmodel['h']
    Brm_L1=get_Br(L1, phim, thetam, g, h,r)
    Br_me_L1=Br_me_L1+Brm_L1
    
    Brm_L2=get_Br(L2, phim, thetam, g, h,r)
    Br_me_L2=Br_me_L2+Brm_L2
    
    Brm_L3=get_Br(L, phim, thetam, g, h,r)
    Br_me_L3=Br_me_L3+Brm_L3
    
    Brm_lf=get_Br(L_f, phim, thetam, g, h,r)
    Br_me_lf=Br_me_lf+Brm_lf

# gmodel=loadmat(pwd+fd+r'\\'+fL[121])
# Br_me_L1=get_Br(L1, phim, thetam, g, h,r)
# Br_me_L2=get_Br(L2, phim, thetam, g, h,r)
# Br_me_L3=get_Br(L3, phim, thetam, g, h,r)
# Br_me_lf=get_Br(L_f, phim, thetam, g, h,r)

tname=mL[60][0:-4]

# fig=plt.figure()
# ax=fig.add_subplot(1,1,1)
# map=Basemap(projection='hammer',lon_0=0,resolution='c')
# map.drawcoastlines()
# p=map.pcolormesh(lonm,latm,Br_me/len(fL),cmap='bwr',latlon='True')
# cbar=map.colorbar(p,'right')
# cbar.set_label('nT')
# ax.set_title(mL[121][0:-4])  

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
[rfp_lon,rfp_lat]=idf_rfp(mequt_lon,mequt_lat,Br_me_lf/len(fL),phi,theta)
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
[rfp_lon,rfp_lat]=idf_rfp(mequt_lon,mequt_lat,Br_me_lf/len(fL),phi,theta)
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
[rfp_lon,rfp_lat]=idf_rfp(mequt_lon,mequt_lat,Br_me_lf/len(fL),phi,theta)
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
[rfp_lon,rfp_lat]=idf_rfp(mequt_lon,mequt_lat,Br_me_lf/len(fL),phi,theta)
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
[rfp_lon,rfp_lat]=idf_rfp(mequt_lon,mequt_lat,Br_me_lf/len(fL),phi,theta)
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
map=Basemap(projection='hammer',lon_0=0,resolution='c')
map.drawcoastlines(linewidth=0.3)
map.drawparallels(np.arange(-60,90,30),labels=[0,0,0,0],linewidth=0.3)
map.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],linewidth=0.3)
map.scatter(rfp_lon,90-rfp_lat,color='r',s=0.5,latlon=True)
map.scatter(mequt_lon,90-mequt_lat,s=0.1,color='b',latlon=True)
ax.set_title('geographic equator') 

##################
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
m = Basemap(width=12000000,height=9000000,projection='lcc',
            resolution='c',lat_1=40.,lat_2=55,lat_0=50,lon_0=-100.)
# draw a boundary around the map, fill the background.
# this background will end up being the ocean color, since
# the continents will be drawn on top.
#m.drawmapboundary(fill_color='aqua')
# fill continents, set lake color same as ocean color.
#m.fillcontinents(color='coral',lake_color='aqua')
# draw parallels and meridians.
# label parallels on right and top
# meridians on bottom and left
tx_lat=[45,45,35,35,45]
tx_lon=[-120,-100,-100,-120,-120]
parallels = np.arange(0.,81,10)
# labels = [left,right,top,bottom]
m.drawparallels(parallels,labels=[0,0,0,0])
meridians = np.arange(10.,351.,20.)
m.drawmeridians(meridians,labels=[0,0,0,0])
# plot blue dot on Boulder, colorado and label it as such.
lon, lat = -110, 40 # Location of Boulder
# convert to map projection coords.
# Note that lon,lat can be scalars, lists or numpy arrays.
xpt,ypt = m(lon,lat)
x_tx,y_tx=m(tx_lon,tx_lat)
# convert back to lat/lon
lonpt, latpt = m(xpt,ypt,inverse=True)
m.plot(xpt,ypt,'bo')  # plot a blue dot there
m.plot(x_tx,y_tx,'r')
# put some text next to the dot, offset a little bit
# (the offset is in map projection coordinates)
plt.text(xpt+1200000,ypt,'(lat,lon)' )
#plt.show()