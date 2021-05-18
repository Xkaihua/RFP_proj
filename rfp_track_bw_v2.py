# -*- coding: utf-8 -*-
"""
Created on Sat May  1 16:14:11 2021

@author: xukai
"""
## label and track patches in binary image

import copy as cp
import numpy as np
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
# from sklearn.cluster import KMeans
# from sklearn.cluster import DBSCAN
# from sklearn.cluster import SpectralClustering
# from sklearn.cluster import OPTICS
from rfp_fun import * 
from skimage import measure



#########################
# def pek_intensity(Br_f,label_bw):
#     bn=np.size(Br_f,axis=0)
#     bm=np.size(Br_f,axis=1)
#     Bp=np.zeros((bn+2,bm+2))
#     Bp[1:-1,0]=Br_f[:,-1]
#     Bp[1:-1,1:-1]=Br_f
#     Bp[1:-1,-1]=Br_f[:,0]
#     Bp[0,:]=Bp[1,:]
#     Bp[-1,:]=Bp[-2,:]
#     pekh=[]
#     pekhx=[]
#     pekhy=[]
#     pekl=[]
#     peklx=[]
#     pekly=[]
    
#     for ilab in range(1,max(label_bw.flatten())+1):
#         [ir, ic] = np.where(label_bw == ilab)
#         for r,c in zip(ir,ic):
#             pt=np.zeros(8)
#             p=Bp[r+1,c+1]
#             pt[0]=Bp[r,c]
#             pt[1]=Bp[r,c+1]
#             pt[2]=Bp[r,c+2]
#             pt[3]=Bp[r+1,c]
#             pt[4]=Bp[r+1,c+2]
#             pt[5]=Bp[r+2,c]
#             pt[6]=Bp[r+2,c+1]
#             pt[7]=Bp[r+2,c+2]
#             if all((p-pt) > 0):
#                 pekh=np.append(pekh,p)
#                 pekhx=np.append(pekhx,theta[r])
#                 pekhy=np.append(pekhy,phi[c])
#             elif all((p-pt) < 0):
#                 pekl=np.append(pekl,p)
#                 peklx=np.append(peklx,theta[r])
#                 pekly=np.append(pekly,phi[c])
                
#     pek=np.append(pekl,pekh)
#     pekx=np.append(peklx,pekhx)
#     peky=np.append(pekly,pekhy)
    
#     return(pek,pekx,peky)

#########

def merge_edge_label(label_bw):
# merge the label near 0 latitude
    for i in range(0, len(label_bw)):
        if label_bw[i,0] != 0 and label_bw[i,-1] != 0:
            st_lab=np.sort([label_bw[i,0],label_bw[i,-1]])
            label_bw[label_bw==st_lab[1]]=st_lab[0]
            if st_lab[0]!=st_lab[1]:
                label_bw[label_bw>st_lab[1]]=label_bw[label_bw>st_lab[1]]-1
                
    return(label_bw)


##
def edge_from_label(label_bw,edge_label):
    [m,n]=np.shape(label_bw)
    for i in range(1,m-1):
        for j in range(1,n-1):
            if ((label_bw[i,j] == label_bw[i-1,j]) & (label_bw[i,j] == label_bw[i+1,j]) & (label_bw[i,j] == label_bw[i,j-1]) & (label_bw[i,j] == label_bw[i,j+1])).all():
                edge_label[i,j] = 0
    for k in range(1,m-1):
        if ((label_bw[k,0] == label_bw[k-1,0]) & (label_bw[k,0] == label_bw[k+1,0]) & (label_bw[k,0] == label_bw[k,n-1]) & (label_bw[k,0] == label_bw[k,1])).all():
            edge_label[k,0] = 0
        if ((label_bw[k,n-1] == label_bw[k-1,n-1]) & (label_bw[k,n-1] == label_bw[k+1,n-1]) & (label_bw[k,n-1] == label_bw[k,n-2]) & (label_bw[k,n-1] == label_bw[k,0])).all():
            edge_label[k,n-1] = 0
                  
    return(edge_label)
#####
#############

gmf_l=r'/gufm_coef_mat'                  #fold of gufm gauss coef mat files

flist=os.listdir(os.getcwd()+gmf_l)
Lme=3                                  #max degree used to identify the magnetic equator
Lf=14                                    #max degree used to calculate the field
r=3480                                     # radius of CMB
theta=np.arange(0.45,180,0.45)
phi=np.arange(0.,360,0.45)                    # grids of spherical surface
lat_w=np.around(theta[1]-theta[0],3)
lon_w=np.around(phi[1]-phi[0],3)
[phim,thetam]=np.meshgrid(phi,theta)

# start point of the track
sp_i = 0
# end point of the track
edp_i = 2

### initial an example

gmodel=loadmat(os.getcwd()+gmf_l+'/'+flist[sp_i])    #load gauss coefficients
g = gmodel['g']
h = gmodel['h']
Br_me = get_Br(Lme, phim, thetam, g, h,r)
Br_f = get_Br(Lf, phim, thetam, g, h,r)
[mequt_lat,mequt_lon] = idf_mequt(Br_me,phi,theta)    
[rfp_lon,rfp_lat,e_lon,e_lat] = idf_rfp(mequt_lon,mequt_lat,Br_f,phi,theta)
bw = bw_rfp(mequt_lon,mequt_lat,Br_f,phi,theta)
label_bw = measure.label(bw,connectivity=1)
label_bw = merge_edge_label(label_bw)    

fa_label=np.arange(1,max(label_bw.flatten())+1,dtype='int32')
new_pn=max(fa_label)
#coloM=['grey','brown','r','orange','gold','olive','yellow','greenyellow','green','cyan','skyblue','blue','blueviolet','violet','purple','deeppink','pink']
colo_sed=np.array(['pink','crimson','palevioletred','deeppink','orchid','fuchsia','purple','slateblue','blue','navy','steelblue','skyblue','cadetblue','cyan','teal','turquoise','springgreen','seagreen','limegreen','green','greenyellow','olivedrab','yellow','olive','gold','goldenrod','orange','darkorange','peru','chocolate','sienna','coral','orangered','red','firebrick','maroon'])
coloM=np.append(colo_sed,colo_sed)
coloM=np.append(coloM,colo_sed)
coloM=np.append(coloM,colo_sed)

#### test 

#test bw 

# fig=plt.figure()
# ax=fig.add_subplot(1,1,1)
# map=Basemap(projection='hammer',lon_0=0,resolution='c')
# map.drawcoastlines(linewidth=0.3)
# map.drawparallels(np.arange(-60,90,30),labels=[0,0,0,0],linewidth=0.3)
# map.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,0],linewidth=0.3)
# map.scatter(rfp_lon,90-rfp_lat,color='r',s=0.01,latlon=True)
# map.scatter(mequt_lon,90-mequt_lat,s=0.01,color='b',latlon=True)
# ax.set_title('l_eq_max=3') 

# fig=plt.figure()
# ax=fig.add_subplot(1,1,1)
# map=Basemap(projection='hammer',lon_0=0,resolution='c')
# map.drawcoastlines(linewidth=0.3)
# map.drawparallels(np.arange(-60,90,30),labels=[0,0,0,0],linewidth=0.3)
# map.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,0],linewidth=0.3)
# map.pcolor(phim,90-thetam,bw,cmap='binary',latlon=True)
# map.scatter(mequt_lon,90-mequt_lat,s=0.01,color='b',latlon=True)
# ax.set_title('l_eq_max=3') 

# ### test bwlabel and merge of the patches divided by 0 latitude 

# fig=plt.figure()
# ax=fig.add_subplot(1,1,1)
# map=Basemap(projection='hammer',lon_0=0,resolution='c')
# map.drawcoastlines(linewidth=0.3)
# map.drawparallels(np.arange(-60,90,30),labels=[0,0,0,0],linewidth=0.3)
# map.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,0],linewidth=0.3)
# map.scatter(mequt_lon,90-mequt_lat,s=0.01,color='b',latlon=True)
# for klass, cr in zip(range(1, max(label_bw.flatten())+1), coloM):
#     phit = phim[label_bw == klass]
#     thetat = thetam[label_bw == klass]
#     map.scatter(phit.T, 90-thetat.T,color=cr,s=1,latlon=True)
    
# fig=plt.figure()
# ax=fig.add_subplot(1,1,1)
# for klass, cr in zip(range(1, max(label_bw.flatten())+1), coloM):
#     phit = phim[label_bw == klass]
#     thetat = thetam[label_bw == klass]
#     ax.scatter(phit.T, 90-thetat.T,color=cr,s=30)
# ax.set_ylim(-40,40)
# ax.set_xlim(58,62)
# ax.set_xlabel('longitude')
# ax.set_ylabel('latitude')
    
##test colorM
# ck=10
# fig=plt.figure()
# ax=fig.add_subplot(1,1,1)
# for im in range(10):
#     cm=coloM[im]
#     ax.scatter(ck,10,color=cm,s=50,marker='o')
#     ck=ck+10

#########  main , track the rfps at every time point

phi_mass_yr=[]
theta_mass_yr=[]
mass_yr=[]
mass_label=np.array([],dtype='int32')

phi_pek_yr=[]
theta_pek_yr=[]
pek_yr=[]
pek_label=np.array([],dtype='int32')

phi_mm_yr=[]
theta_mm_yr=[]
mm_yr=[]
mm_label=np.array([],dtype='int32')

phi_edge=[]
theta_edge=[]

for ig in range(sp_i,edp_i):
    yr=float(flist[ig][0:-4])
    gmodel=loadmat(os.getcwd()+gmf_l+'/'+flist[ig])
    g=gmodel['g']
    h=gmodel['h']
    Br_me=get_Br(Lme, phim, thetam, g, h,r)
    Br_f=get_Br(Lf, phim, thetam, g, h,r)
    [mequt_lat,mequt_lon]=idf_mequt(Br_me,phi,theta)    
    bw_tmp=bw_rfp(mequt_lon,mequt_lat,Br_f,phi,theta)
    bw_label_tmp=measure.label(bw_tmp,connectivity=1)
    bw_label_tmp=merge_edge_label(bw_label_tmp)
   
    ########## determin the father label of every patch
    crit_labelm=np.zeros((max(label_bw.flatten()),max(bw_label_tmp.flatten())))
    fa_label_tmp = np.arange(1,max(bw_label_tmp.flatten())+1,dtype='int32')
    
    for ilb in range(1,max(label_bw.flatten())+1):
        phi_blt=phim[label_bw == ilb]
        theta_blt=thetam[label_bw == ilb]
        point_blt=np.array([phi_blt,theta_blt])
        point_blt=np.around(point_blt.T,5)
        point_blt_lst=point_blt.tolist()
        for jbw in range(1,max(bw_label_tmp.flatten())+1):
            phi_tmp=phim[bw_label_tmp == jbw]
            theta_tmp=thetam[bw_label_tmp == jbw]
            point_tmp=np.array([phi_tmp,theta_tmp])
            point_tmp=np.around(point_tmp.T,5)
            point_tmp_lst=point_tmp.tolist()
            spt=[x for x in point_tmp_lst if x in point_blt_lst]
            crit_labelm[ilb-1,jbw-1]=len(spt)/(len(point_tmp_lst)+len(point_blt_lst))*100
    crit_labelm[crit_labelm < 5] = 0        
    for il in range(np.size(crit_labelm,axis = 1)):
        [kn] = np.where(crit_labelm[:,il] != 0)
        if len(kn) == 1:
            fa_label_tmp[il] = fa_label[kn]
        elif len(kn) >1:
            sot_cl = np.argmax(crit_labelm[kn,il])
            fa_label_tmp[il] = fa_label[kn[sot_cl]]
        else :
            fa_label_tmp[il] = new_pn  + 1
            new_pn = new_pn +1
            
    for jl in range(np.size(crit_labelm, axis = 0)):
        [km] = np.where(crit_labelm[jl,:] != 0)
        if len(km) != 1:
            sot_rl = np.argsort(-crit_labelm[jl,km])     #sort descending
            for js in range(1,len(sot_rl)):
                fa_label_tmp[km[sot_rl[js]]] = new_pn +1
                new_pn = new_pn +1
    
    #fa_label_tmp = np.arange(1,max(bw_label_tmp.flatten())+1,dtype='int32')
    
    # phi_mm_tmp = np.zeros(max(bw_label_tmp.flatten()))
    # theta_mm_tmp = np.zeros(max(bw_label_tmp.flatten()))
    # fa_label_tmp = np.zeros(max(bw_label_tmp.flatten()))
    # for ilb in range(1, max(bw_label_tmp.flatten())+1):
    #     phi_patch = np.around(phim[bw_label_tmp == ilb],5)
    #     theta_patch = np.around(thetam[bw_label_tmp == ilb],5)
    #     Brt = Br_f[bw_label_tmp == ilb]
    #     phi_mm_tmp[ilb-1] = phi_patch[np.argmax(abs(Brt))]
    #     theta_mm_tmp[ilb-1] = theta_patch[np.argmax(abs(Brt))]
    #     fa_label_tmp[ilb-1] = ilb
        
    
                
    ## pek intensity
    [pek,pek_theta,pek_phi,pek_bwlabel] = pek_intensity(Br_f,bw_label_tmp,theta,phi) 
    ## fliter the weak patch
    a_pek = abs(pek)
    crit_pek = max(a_pek)/2
    phi_pek_yr = np.append(phi_pek_yr,pek_phi)
    theta_pek_yr = np.append(theta_pek_yr,pek_theta)
    pek_yr = np.append(pek_yr, yr*np.ones(len(pek_phi)))
    fa_label_pek=fa_label_tmp[pek_bwlabel -1]
    fa_label_pek[a_pek <= crit_pek] = 0
    pek_label = np.append(pek_label,fa_label_pek)
    
    ######### plot patches and find the mass centers     
    edge_label = cp.deepcopy(bw_label_tmp)
    edge_label = edge_from_label(bw_label_tmp, edge_label)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    map=Basemap(projection='hammer',lon_0=0,resolution='c')
    map.drawcoastlines(linewidth=0.3)
    map.drawparallels(np.arange(-60,90,30),labels=[0,0,0,0],linewidth=0.3)
    map.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,0],linewidth=0.3)
    map.scatter(mequt_lon,90-mequt_lat,s=0.01,color='b',latlon=True)
    for klass in range(1, max(bw_label_tmp.flatten())+1):
        phit = np.around(phim[bw_label_tmp == klass],5)
        thetat = np.around(thetam[bw_label_tmp == klass],5)
        Bt = np.around(Br_f[bw_label_tmp == klass],5)
        w=abs(Bt)
        [Xt,Yt,Zt]=rtp2xyz(r, thetat, phit)
        x_mass = np.sum(Xt*w)/np.sum(w)
        y_mass = np.sum(Yt*w)/np.sum(w)
        z_mass = np.sum(Zt*w)/np.sum(w)
        [theta_mass,phi_mass] = xyz2rtp(x_mass, y_mass, z_mass)
        phi_mass_yr = np.append(phi_mass_yr,phi_mass)
        theta_mass_yr = np.append(theta_mass_yr,theta_mass)
        mass_yr = np.append(mass_yr,yr)
        mass_label = np.append(mass_label, fa_label_tmp[klass-1])
    
        theta_mm=thetat[np.argmax(w)]
        phi_mm=phit[np.argmax(w)]
        theta_mm_yr = np.append(theta_mm_yr,theta_mm)
        phi_mm_yr = np.append(phi_mm_yr,phi_mm)
        mm_yr = np.append(mm_yr,yr)
        mm_label = np.append(mm_label, fa_label_tmp[klass-1])
        
        phi_edge_tmp = phim[edge_label == klass]
        phi_edge.append(phi_edge_tmp)
        theta_edge_tmp = thetam[edge_label == klass]
        theta_edge.append(theta_edge_tmp)        
            
        cr=coloM[fa_label_tmp[klass-1]-1]
        map.scatter(phit, 90-thetat,color=cr,s=1,latlon=True)  
        # map.scatter(phi_mass,90-theta_mass,color='k',s=30,marker='*',latlon=True)
        # map.scatter(phi_mm,90-theta_mm,color='k',s=30,marker='o',latlon=True)
        xpt,ypt = map(phi_mass,90-theta_mass)
        plt.text(xpt,ypt,str(fa_label_tmp[klass-1]))
        
    # phi_p = pek_phi[fa_label_pek >0 ]
    # theta_p = pek_theta[fa_label_pek >0 ]
    # map.scatter(phi_p,90-theta_p,color='k',s=10,marker='o',latlon=True)
    ax.set_title(str(yr)) 
        
    label_bw = cp.deepcopy(bw_label_tmp)
    fa_label = cp.deepcopy(fa_label_tmp)
    Br_f_last = cp.deepcopy(Br_f)
    
##############

## diagnose codes
# klass=2
# fig=plt.figure()
# ax=fig.add_subplot(1,1,1)
# map=Basemap(projection='hammer',lon_0=0,resolution='c')
# map.drawcoastlines(linewidth=0.3)
# map.drawparallels(np.arange(-60,90,30),labels=[0,0,0,0],linewidth=0.3)
# map.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,0],linewidth=0.3)
# map.scatter(mequt_lon,90-mequt_lat,s=0.01,color='b',latlon=True)
# phit = phim[label_bw == klass]
# thetat = thetam[label_bw == klass]
# [Xt,Yt,Zt]=rtp2xyz(r, thetat, phit)
# [theta_t,phi_t] = xyz2rtp(Xt, Yt, Zt)
# Bt = np.around(Br_f[label_bw == klass],5)
# w=abs(Bt)
# phi_mass_dirc=np.sum(phit*w)/np.sum(w)
# theta_mass_dirc=np.sum(thetat*w)/np.sum(w)

# phi_mass = phi_mass_yr[mass_label == klass]
# theta_mass = theta_mass_yr[mass_label == klass]
# phi_mm = phi_mm_yr[mm_label == klass]
# theta_mm =theta_mm_yr[mm_label == klass]
# cr=coloM[fa_label_tmp[klass-1]-1]
# map.scatter(phit, 90-thetat,color=cr,s=1,latlon=True)
# map.scatter(phi_t, 90-theta_t,color='pink',s=1,latlon=True)
# map.scatter(phi_mass_dirc, 90-theta_mass_dirc,color='b',s=30,marker='+',latlon=True)
# map.scatter(phi_mass,90-theta_mass,color='k',s=30,marker='*',latlon=True)
# map.scatter(phi_mm,90-theta_mm,color='k',s=30,marker='o',latlon=True)


# Br_plot_area=cp.deepcopy(Br_f)
# [lx,ly]=np.where(label_bw != klass)
# Br_plot_area[lx,ly] = 0
# fig=plt.figure()
# ax=fig.add_subplot(1,1,1)
# map=Basemap(projection='hammer',lon_0=0,resolution='c')
# map.drawcoastlines(linewidth=0.3)
# map.drawparallels(np.arange(-60,90,30),labels=[0,0,0,0],linewidth=0.3)
# map.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,0],linewidth=0.3)
# map.scatter(mequt_lon,90-mequt_lat,s=0.01,color='b',latlon=True)
# map.pcolor(phim,90-thetam,Br_plot_area,cmap='bwr',latlon=True)

###########################

# # plot mass centre track
# fig=plt.figure()
# ax=fig.add_subplot(1,1,1)
# for uk in range(1,max(mass_label)+1):
#     pc_phi = phi_mass_yr[mass_label == uk]
#     if len(pc_phi) < 5:
#         continue
#     pc_phi[pc_phi > 180] = pc_phi[pc_phi >180] -360
#     pyr = mass_yr[mass_label == uk]
#     cr = coloM[uk-1]
#     ax.scatter(pyr,pc_phi,color=cr,s=1)
# ax.set_ylim(-180,180)
# ax.set_xlim(1500,2000)
# maloc=plt.MultipleLocator(60)
# ax.yaxis.set_major_locator(maloc)
# ax.set_title('phi')
# ax.grid(which='major',axis='y') 

# fig=plt.figure()
# ax=fig.add_subplot(1,1,1)
# for uk in range(1,max(mass_label)+1):
#     pc_theta = theta_mass_yr[mass_label == uk]
#     if len(pc_theta) < 5:
#         continue
#     pyr = mass_yr[mass_label == uk]
#     cr = coloM[uk-1]
#     ax.scatter(pyr,pc_theta,color=cr,s=1)
# ax.set_ylim(0,180)
# ax.set_xlim(1500,2000)
# ax.invert_yaxis()
# maloc=plt.MultipleLocator(60)
# ax.yaxis.set_major_locator(maloc)
# ax.set_title('theta')    
# ax.grid(which='major',axis='y')           

# ######
# ## plot max(min) centre track
# fig=plt.figure()
# ax=fig.add_subplot(1,1,1)
# for um in range(1,max(mm_label)+1):
#     pc_phi_mm = phi_mm_yr[mm_label == um]
#     if len(pc_phi_mm) < 5:
#         continue
    
#     pc_phi_mm[pc_phi_mm > 180] = pc_phi_mm[pc_phi_mm >180] -360
#     pyr_mm = mm_yr[mm_label == um]
#     cr_mm = coloM[um-1]
#     ax.scatter(pyr_mm,pc_phi_mm,color=cr_mm,s=1)
# ax.set_ylim(-180,180)
# ax.set_xlim(1500,2000)
# maloc=plt.MultipleLocator(60)
# ax.yaxis.set_major_locator(maloc)
# ax.set_title('phi')
# ax.grid(which='major',axis='y') 

# fig=plt.figure()
# ax=fig.add_subplot(1,1,1)
# for uk in range(1,max(mm_label)+1):
#     pc_theta_mm = theta_mm_yr[mm_label == uk]
#     if len(pc_theta_mm) < 5:
#         continue
#     pyr_mm = mm_yr[mm_label == uk]
#     cr_mm = coloM[uk-1]
#     ax.scatter(pyr_mm,pc_theta_mm,color=cr_mm,s=1)
# ax.set_ylim(0,180)
# ax.set_xlim(1500,2000)
# ax.invert_yaxis()
# maloc=plt.MultipleLocator(60)
# ax.yaxis.set_major_locator(maloc)
# ax.set_title('theta')    
# ax.grid(which='major',axis='y')    

# #####
# # plot peak track
# fig=plt.figure()
# ax=fig.add_subplot(1,1,1)
# for up in range(1,max(pek_label)+1):
#     pc_phi_pek = phi_pek_yr[pek_label == up]
#     pc_phi_pek[pc_phi_pek > 180] = pc_phi_pek[pc_phi_pek >180] -360
#     pyr_pek = pek_yr[pek_label == up]
#     cr = coloM[up-1]
#     ax.scatter(pyr_pek,pc_phi_pek,color=cr,s=1)
# ax.set_ylim(-180,180)
# ax.set_xlim(1500,2000)
# maloc=plt.MultipleLocator(60)
# ax.yaxis.set_major_locator(maloc)
# ax.set_title('phi')
# ax.grid(which='major',axis='y')    

# fig=plt.figure()
# ax=fig.add_subplot(1,1,1)
# for up in range(1,max(pek_label)+1):
#     pc_theta_pek = theta_pek_yr[pek_label == up]
#     pyr_pek = pek_yr[pek_label == up]
#     cr = coloM[up-1]
#     ax.scatter(pyr_pek,pc_theta_pek,color=cr,s=1)
# ax.set_ylim(0,180)
# ax.set_xlim(1500,2000)
# maloc=plt.MultipleLocator(60)
# ax.yaxis.set_major_locator(maloc)
# ax.invert_yaxis()
# ax.set_title('theta')    
# ax.grid(which='major',axis='y')


# ##########
