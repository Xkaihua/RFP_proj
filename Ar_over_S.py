# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:43:56 2021

@author: xukai
"""
import numpy as np
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl

from rfp_fun import * 

######################################################
##calculate the ratio of flux area over spherical surface
#######

gmf_l=r'\gufm_coef_mat'                  #fold of gufm gauss coef mat files

flist=os.listdir(os.getcwd()+gmf_l)
Lme=3                                     #max degree used to identify the magnetic equator
Lf=14                                     #max degree used to calculate the field
r=3480                                     # radius of CMB
theta=np.arange(0,180,1)
phi=np.arange(0,360,1)                    # grids of spherical surface
lat_w=np.around(theta[1]-theta[0],3)
lon_w=np.around(phi[1]-phi[0],3)
[phim,thetam]=np.meshgrid(phi,theta)

f_area_r=[]                             # reversed flux patches area
f_area_e=[]                             # reversed flux patches edge point area
f_year=[]                               # year 
ck=0                                    #  count of the loop
for fp in flist:
    gmodel=loadmat(os.getcwd()+gmf_l+'\\'+fp)
    g=gmodel['g']
    h=gmodel['h']
    Br_me=get_Br(Lme, phim, thetam, g, h,r)
    Br_f=get_Br(Lf, phim, thetam, g, h,r)
    
    [mequt_lat,mequt_lon]=idf_mequt(Br_me,phi,theta)    
    [rfp_lon,rfp_lat,e_lon,e_lat]=idf_rfp(mequt_lon,mequt_lat,Br_f,phi,theta)

    
    area_r=get_area2rfp(rfp_lat,lat_w,lon_w)
    area_e=get_area2rfp(e_lat,lat_w,lon_w)        
    f_area_r=np.append(f_area_r,area_r)
    f_area_e=np.append(f_area_e,area_e)
    f_year=np.append(f_year,float(fp[0:-4]))
    
    ck=ck+1
    print(ck)

s_rate=(f_area_r-f_area_e)/(4*np.pi*r*r)
s_rate_with_edge=(f_area_r)/(4*np.pi*r*r)

ify=np.argsort(f_year)
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
plot1=ax.plot(f_year[ify],s_rate[ify],color='b')
ax.set_xlim(1840,2015)
#ax.set_ylim(0.08,0.24)
ax.set_ylim(0.08,0.24)    
# plt.xticks([])
# plt.yticks([])
ax.set_title('mel_max=3')            
ax.set_xlabel('year')
ax.set_ylabel('Area/s') 
plt.show()