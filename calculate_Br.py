# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 15:53:48 2021

@author: xukai
"""
######  script to creat Br datafile

import numpy as np
import os
from scipy.io import loadmat
from pathlib import Path
import rfp_functions as rf

gmf_l=Path('cov_obsx1_aver_coefs_1yr_mat/')  #folder of gauss coefficients
pwd=os.getcwd()

flist=os.listdir(gmf_l)
L=3                                  #max degree used to identify the magnetic equator
r=3480                               # radius of CMB
grid_space=0.45                        # grid space           
theta=np.arange(grid_space,180,grid_space)
phi=np.arange(0.,360,grid_space)                    # grids of spherical surface
lat_w=np.around(theta[1]-theta[0],3)
lon_w=np.around(phi[1]-phi[0],3)
[phim,thetam]=np.meshgrid(phi,theta)

spath=Path('Br/cov_obsx1/grid={gs}/L={l}/'.format(gs=grid_space,l=L))   # path for storing the Br
spath.mkdir(parents=True)

for fp in flist:
    gmodel=loadmat(gmf_l / fp)
    g=gmodel['g']
    h=gmodel['h']
    Br=rf.get_Br(L, phim, thetam, g, h,r)
    fname=fp[0:-4]+'.npy'
    np.save(spath / fname,Br)
    