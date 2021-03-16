# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 16:03:30 2021

@author: xukai
"""

import numpy as np
import os
from scipy.io import loadmat
from scipy.io import savemat
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


def get_gaucoefs(tfl):
#get the gauss coefficients data from txt file 
#input
#     tfl: path of the txt file
#output
#     g,h: gauss coefficients g and h   
    tgau=np.loadtxt(tfl,skiprows=1)
    n=tgau[:,0]
    m=tgau[:,1]
    gl=tgau[:,2]
    hl=tgau[:,3]
    L=np.max(n.astype(int))
    g=np.zeros((L,L+1))
    h=np.zeros((L,L+1))
    i=1
    while i<L+1:
        idx=np.argwhere(m==i)
        g[i-1:L,i]=gl[idx[0:L+1-i]].flatten()
        h[i-1:L,i]=hl[idx[0:L+1-i]].flatten()
        i=i+1
    id0=np.argwhere(m==0)
    g[:,0]=gl[id0].flatten()
    h[:,0]=hl[id0].flatten()
    return(g,h)
#[g,h]=get_gaucoefs(tfile)


def legendre(l,theta):
# calculate the Schmidt seminormalized associated Legendre functions
# input 
#       l: the max degree of  Legendre functions
#       theta: grid points where you want to calculate the Legendre functions
# output
#       Pnm:  values of Legendre functions with max degree l at theta
    rad_theta=theta*np.pi/180
    lt=len(theta)
    p00=np.ones(lt)
    p10=np.cos(rad_theta)
    p11=np.sin(rad_theta)
    Pm=np.zeros((l+1,lt))
    Pm[0,:]=p00
    Pm[1,:]=p11
    k=2
    while k<l+1:
        m=k
        w1=(2*m-1)/(2*m)
        Pm[k,:]=np.sqrt(w1)*np.sin(rad_theta)*Pm[k-1,:]
        k=k+1
    Pnm=np.zeros((l+1,l+1,lt))
    Pnm[0,0,:]=Pm[0,:]
    Pnm[1,0,:]=p10
    Pnm[1,1,:]=Pm[1,:]
    for i in np.arange(2,l+1):
        Pnm[i,i,:]=Pm[i,:]
        for j in np.arange(i):
            pn_1m=Pnm[i-1,j,:]
            pn_2m=Pnm[i-2,j,:]
            n=i
            Rnm=np.sqrt(np.square(n)-np.square(j))
            Rn_1m=np.sqrt(np.square(n-1)-np.square(j))
            Pnm[i,j,:]=((2*n-1)*np.cos(rad_theta)*pn_1m-Rn_1m*pn_2m)/Rnm
           
            
    return(Pnm)

#pnm=legendre(5,theta)  
#ledr=pnm[5,0,:]
#plt.plot(theta,ledr)

pwd=os.getcwd()
#gmf=r'\cals3k.4_coef_mat\1980'
gmf=r'\gufm_coef_mat\1962.5'
# tfile=r'E:\CAS\cals_model\CALS10k2\coefsdata\0.txt'
# [g,h]=get_gaucoefs(tfile)

#####################
gmpath=pwd+gmf  
# gmpath=r'E:\CAS\dipole_simulate\igrf_coe\2015'
gmodel=loadmat(gmpath)
g=gmodel['g']
h=gmodel['h']

#theta=np.arange(0,181)
#phi=np.arange(0,360)
theta=np.arange(0,180,1)
phi=np.arange(0,360,1)
lat_w=np.around(theta[1]-theta[0],3)
lon_w=np.around(phi[1]-phi[0],3)

# a=6371
# r=3471

L=3                       # max degree for finding equator
L_f=14
[phim,thetam]=np.meshgrid(phi,theta)

def get_Br(L,phim,thetam,g,h):
# calculate Br on CMB
# input
#      L: max degree to calculate Br
#      phim: colatitude(unit:degree) of the grid points
#      thetam: longitude(unit:degree) of the grid points
#      g,h: gauss coefficients g and h
# output
#       Brm: radial magnetic field Br
    a=6371
    r=3480
    rad_theta=thetam.flatten('F')*np.pi/180
    rad_phi=phim.flatten('F')*np.pi/180
    pnm=legendre(L,thetam.flatten('F'))
    zm=np.zeros((L,len(rad_theta)))
    for i in np.arange(L):
        zn=np.zeros((i+2,len(rad_theta)))
        for j in np.arange(i+2): 
            n=i+1
            Rnm=np.sqrt(np.square(n)-np.square(j))
            kn=np.power(a/r,n+2)*(n+1)
            zn[j,:]=kn*((g[i,j]*np.cos(j*rad_phi)+h[i,j]*np.sin(j*rad_phi))*pnm[n,j,:])
        zm[i,:]=np.sum(zn,axis=0)
    Z=np.sum(zm,axis=0)*(-1)
    Sz=phim.shape
    mZ=Z.reshape(Sz[:],order='F')
    Brm=-mZ
    return(Brm)

Brm=get_Br(L, phim, thetam, g, h)
lat=90-theta
lon=phi
[lonm,latm]=np.meshgrid(lon,lat)
fig=plt.figure
map=Basemap(projection='hammer',lon_0=0,resolution='c')
map.drawcoastlines()
p=map.pcolormesh(lonm,latm,Brm,cmap='bwr',latlon='True')
cbar=map.colorbar(p,'right')
cbar.set_label('nT')
# plt.show()
#plt.pcolormesh(lonm,latm,mZ,cmap='bwr')

idg=0
def idf_mequt(Brm,phi,theta):
# identify the magnetic equator from Br matix
    signal_br=Brm[0:-1,:]*Brm[1:,:]
    [ix,iy]=np.where(signal_br<=0)
    ilat=theta[ix]
    ilon=phi[iy]
    sy=np.argsort(ilon)
    ilon=ilon[sy]
    ilat=ilat[sy]
    k=0
    u_lon_c=np.zeros(len(phi))
    u_lon_o=np.zeros(len(phi))
    for i in np.arange(len(phi)):
        u_lon_c[k]=np.sum(np.around(ilon,5)==np.around(phi[i],5))
        u_lon_o[k]=phi[i]
        k=k+1
    sc=np.argsort(u_lon_c)
    u_lon_c_sort=u_lon_c[sc]
    u_lon_o_sort=u_lon_o[sc]
    
    sta_lon=u_lon_o_sort[0]
    mequt_lat=np.zeros(1)
    mequt_lon=np.zeros(1)
    [il]=np.where(np.around(ilon,5)==np.around(sta_lon,5))
    diff_lat2eqt=abs(ilat[il[:]]-90)
    il2=np.argmin(diff_lat2eqt)
    mequt_lat[0]=ilat[il[il2]]
    mequt_lon[0]=sta_lon
    
    temp_y=np.delete(ilon, il[il2])
    temp_x=np.delete(ilat, il[il2])
    lon_step=phi[1]-phi[0]
    while len(temp_y)>0 :
        dis_lon=abs(temp_y-mequt_lon[-1])
        dis_lon_i=np.where(dis_lon>180)
        dis_lon[dis_lon_i[0][:]]=dis_lon[dis_lon_i[0][:]]-360
        temp_dist=(dis_lon)**2+(temp_x-mequt_lat[-1])**2
        nxt_p=np.argmin(temp_dist)
        
        # temp_y=np.around(temp_y,5)
        # temp_x=np.around(temp_x,5)
        # nb_lon1=np.around(mequt_lon[-1]+lon_step,5)
        # nb_lon2=np.around(mequt_lon[-1]-lon_step,5)   #set precision to the two decimal places
        # if nb_lon1 < phi[0]:
        #     nb_lon1=nb_lon1+360
        # elif nb_lon1 > phi[-1]:
        #     nb_lon1=nb_lon1-360
        
        # if nb_lon2 < phi[0]:
        #     nb_lon2=nb_lon2+360
        # elif nb_lon2 > phi[-1]:
        #     nb_lon2=nb_lon2-360
            
        # [i_nb_lon1]=np.where(temp_y==nb_lon1)
        # [i_nb_lon2]=np.where(temp_y==nb_lon2)
        # Dm1=abs(temp_x[i_nb_lon1]-mequt_lat[-1])
        # Dm2=abs(temp_x[i_nb_lon2]-mequt_lat[-1])
        # nxt_p1=np.argsort(Dm1)
        # nxt_p2=np.argsort(Dm2)
    
        # if nxt_p1.size!=0 and nxt_p2.size==0:
        #     nxt_p=i_nb_lon1[nxt_p1[0]]
        # elif nxt_p1.size==0 and nxt_p2.size!=0:
        #     nxt_p=i_nb_lon2[nxt_p2[0]]
        # elif nxt_p1.size==0 and nxt_p2.size==0:
        #     break
        #     # dis_lon=abs(temp_y-mequt_lon[-1])
        #     # dis_lon_i=np.where(dis_lon>180)
        #     # dis_lon[dis_lon_i[0][:]]=dis_lon[dis_lon_i[0][:]]-360
        #     # temp_dist=(dis_lon)**2+(temp_x-mequt_lat[-1])**2
        #     # nxt_p=np.argmin(temp_dist)
        
        # elif nxt_p1.size!=0 and nxt_p2.size!=0:
        #     if Dm1[nxt_p1][0]<=Dm2[nxt_p2][0]:
        #         nxt_p=i_nb_lon1[nxt_p1[0]]
        #     elif Dm1[nxt_p1][0]>Dm2[nxt_p2][0]:
        #         nxt_p=i_nb_lon2[nxt_p2[0]]
                
        mequt_lat=np.append(mequt_lat,temp_x[nxt_p])
        mequt_lon=np.append(mequt_lon,temp_y[nxt_p])
        #mequt_lon[j+1]=temp_y[nxt_p]
        temp_y=np.delete(temp_y,nxt_p)
        temp_x=np.delete(temp_x,nxt_p)
          
    return(mequt_lat,mequt_lon)

[mequt_lat,mequt_lon]=idf_mequt(Brm,phi,theta)
    
fig=plt.figure()
map=Basemap(projection='hammer',lon_0=0,resolution='c')
map.drawcoastlines()
p=map.pcolormesh(lonm,latm,Brm,cmap='bwr',latlon='True')
cbar=map.colorbar(p,'right')
cbar.set_label('nT')
map.scatter(mequt_lon,90-mequt_lat,s=1,latlon='True')


# fig=plt.figure()
# ax=fig.add_subplot(1,1,1)
# plot1=ax.scatter(iy,90-ix,lw=0.5) 
# ax.set_title('edge line')            
# ax.set_xlabel('lon')
# ax.set_ylabel('lat') 
# plt.show()



dif_lon=abs(np.around(mequt_lon[1:]-mequt_lon[0:-1],3))
[dp_i_lon]=np.where((lon_w < dif_lon)& (dif_lon < np.around(phi[-1]-phi[0],3))) 
if dp_i_lon.size==0:
    it_lon=len(mequt_lon)-1
else:
    [iil]=np.where(dp_i_lon>=0.99*len(phi))
    it_lon=dp_i_lon[iil[0]]
    
dif_lat=np.around(mequt_lat[1:]-mequt_lat[0:-1],3)
[dp_i_lat]=np.where(abs(dif_lat)>60)
if dp_i_lat.size==0:
    it_lat=len(mequt_lat)-1
else:
    it_lat=dp_i_lat[0]


it_dp_i=np.min([it_lon,it_lat])
#it_dp_i=it_lon

mequt_lon_p=mequt_lon[0:it_dp_i+1]
mequt_lat_p=mequt_lat[0:it_dp_i+1]


fig=plt.figure()
ax=fig.add_subplot(1,1,1)
plot1=ax.plot(mequt_lon,90-mequt_lat,lw=0.5) 
ax.set_title('ordered edge line')            
ax.set_xlabel('lon')
ax.set_ylabel('lat') 
plt.show()

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
plot1=ax.scatter(np.arange(len(mequt_lon)),mequt_lon,color='b',s=1,label='edge line')
plot2=ax.scatter(np.arange(len(mequt_lon[0:it_dp_i+1])),mequt_lon[0:it_dp_i+1],color='r',s=1,label='magnetic equator')
ax.legend(loc='upper left') 
ax.set_title('ordered lon')            
ax.set_xlabel('number')
ax.set_ylabel('lon') 
plt.show


fig=plt.figure()
ax=fig.add_subplot(1,1,1)
plot1=ax.scatter(np.arange(len(dif_lon)),dif_lon,s=1)
ax.set_title('diff lon')            
ax.set_xlabel('number')
ax.set_ylabel('lon') 
plt.show()

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
plot1=ax.scatter(np.arange(len(mequt_lat)),mequt_lat,color='b',s=1,label='edge line')
plot2=ax.scatter(np.arange(len(mequt_lat[0:it_dp_i+1])),mequt_lat[0:it_dp_i+1],color='r',s=1,label='magnetic equator')
ax.legend(loc='upper left') 
ax.set_title('ordered lat')            
ax.set_xlabel('number')
ax.set_ylabel('lon') 
plt.show

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
plot1=ax.scatter(np.arange(len(dif_lat)),dif_lat,s=1)
ax.set_title('diff lat')            
ax.set_xlabel('number')
ax.set_ylabel('lon') 
plt.show()

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
plot1=ax.plot(mequt_lon[0:it_dp_i+1],90-mequt_lat[0:it_dp_i+1],color='r',label='magnetic equator')
plot2=ax.plot(mequt_lon[it_dp_i+1:],90-mequt_lat[it_dp_i+1:],color='b',label='RFP')
ax.legend(loc='lower right') 
ax.set_title('magnetic equator and RFP')            
ax.set_xlabel('lon')
ax.set_ylabel('lat') 
plt.show()


fig=plt.figure()
map=Basemap(projection='hammer',lon_0=0,resolution='c')
map.drawcoastlines()
p=map.pcolormesh(lonm,latm,Brm,cmap='bwr',latlon='True')
cbar=map.colorbar(p,'right')
cbar.set_label('nT')
x,y=map(mequt_lon_p,90-mequt_lat_p)
im=np.argsort(x)
px=x[im]
py=y[im]
map.scatter(px,py,s=1,color='b')

##################################
Brm_l10=get_Br(L_f, phim, thetam, g, h)

fig=plt.figure()
map=Basemap(projection='hammer',lon_0=0,resolution='c')
map.drawcoastlines()
p=map.pcolormesh(lonm,latm,Brm_l10,cmap='bwr',latlon=True)
cbar=map.colorbar(p,'right')
cbar.set_label('nT')
map.scatter(px,py,s=1,color='k')

def idf_rfp(mequt_lon_p,mequt_lat_p,Brm_l10,phi,theta):
    # rfp_lat_st=[]
    # rfp_lat_nt=[]
    # rfp_lon_st=[]
    # rfp_lon_nt=[]
    rfp_lat=[]
    rfp_lon=[]
    e_lat=[]
    e_lon=[]
    # phi=np.around(phi,5)
    # theta=np.around(theta,5)
    for i in np.arange(len(phi)):
        #tag_lon=np.around(phi[i],5)
        [i_lon]=np.where(np.around(mequt_lon_p,5)==np.around(phi[i],5))
        tag_lat=mequt_lat_p[i_lon[:]]
        r1=0
        p=2
        if len(tag_lat)==0:
            continue
        else:
            for j in np.arange(len(tag_lat)):
                tag_lat=np.sort(tag_lat)
                #tag_lat=tag_lat[sl]
                [i_lat]=np.where(np.around(theta,5)==np.around(tag_lat[j],5))
                r2=i_lat[0]
                if (p & 1)==0:
                    [ir]=np.where(Brm_l10[r1:r2,i]>0)
                else:
                    [ir]=np.where(Brm_l10[r1:r2,i]<0)   
                
                rfp_lat=np.append(rfp_lat,theta[r1:r2][ir[:]])
                rfp_lon=np.append(rfp_lon,phi[i]*np.ones(len(ir)))
                
                if ir.size>0:
                    e_lat=np.append(e_lat,theta[r1:r2][ir[0]])
                    e_lat=np.append(e_lat,theta[r1:r2][ir[-1]])
                    e_lon=np.append(e_lon,phi[i]*np.ones(2))
                # elif ir.size==1:
                #     e_lat=np.append(e_lat,theta[r1:r2][ir[-1]])
                #     e_lon=np.append(e_lon,phi[i]*np.ones(1))
                    
                p=p+1
                r1=r2+1
                
            [ir]=np.where(Brm_l10[r2+1:,i]<0) 
            rfp_lat=np.append(rfp_lat,theta[r2+1:][ir[:]])
            rfp_lon=np.append(rfp_lon,phi[i]*np.ones(len(ir)))
            
            if ir.size>0:
                e_lat=np.append(e_lat,theta[r2+1:][ir[0]])
                e_lat=np.append(e_lat,theta[r2+1:][ir[-1]])
                e_lon=np.append(e_lon,phi[i]*np.ones(2))
            # elif ir.size==1:
            #     e_lat=np.append(e_lat,theta[r2+1:][ir[-1]])
            #     e_lon=np.append(e_lon,phi[i]*np.ones(1))
        
    return(rfp_lon,rfp_lat,e_lon,e_lat)         
            
    # t_lat=np.around(mequt_lat_p[i],3)
    # [i_lon]=np.where(phi==t_lon)
    # [i_lat]=np.where(theta==t_lat)
    # Br_north=Brm_l10[0:i_lat[0],i_lon[0]]
    # Br_south=Brm_l10[i_lat[0]+2:,i_lon[0]]
    # [i_n]=np.where(Br_north>0)
    # [i_s]=np.where(Br_south<0)
    # rfp_lat_nt=np.append(rfp_lat_nt,theta[0:i_lat[0]][i_n])
    # rfp_lat_st=np.append(rfp_lat_st,theta[i_lat[0]+2:][i_s])
    # rfp_lon_nt=np.append(rfp_lon_nt,t_lon*np.ones(len(i_n)))
    # rfp_lon_st=np.append(rfp_lon_st,t_lon*np.ones(len(i_s)))
    
#    return(rfp_lon_nt,rfp_lat_nt,rfp_lon_st,rfp_lat_st)

#[rfp_lon_nt,rfp_lat_nt,rfp_lon_st,rfp_lat_st]=idf_rfp(mequt_lon_p,mequt_lat_p,Brm_l10,phi,theta)
    
[rfp_lon,rfp_lat,e_lon,e_lat]=idf_rfp(mequt_lon_p,mequt_lat_p,Brm_l10,phi,theta)
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
map=Basemap(projection='hammer',lon_0=0,resolution='c')
map.drawcoastlines()
map.scatter(rfp_lon,90-rfp_lat,color='r',s=1,latlon=True)
#map.scatter(rfp_lon_st,90-rfp_lat_st,color='r',latlon=True)
map.scatter(px,py,s=1,color='k')
ax.set_title(gmf[15:])  
# fig=plt.figure()
# ax=fig.add_subplot(1,1,1)
# ax.scatter(mequt_lon_p,90-mequt_lat_p,s=10)
# ax.scatter(rfp_lon,90-rfp_lat,color='r',s=10)
# ax.set_xlim(100,110)
# ax.set_ylim(-25,50)

def get_area2rfp(rfp_lat,lat_w,lon_w):
    r=3480
    S_rfp=[]
    for i_rfp in np.arange(len(rfp_lat)):
        l_p1=np.sin((rfp_lat[i_rfp]-lat_w/2)*np.pi/180)*r*lon_w*np.pi/180
        l_p2=np.sin((rfp_lat[i_rfp]+lat_w/2)*np.pi/180)*r*lon_w*np.pi/180
        h=lat_w*np.pi*r/180
        s_t=(l_p1+l_p2)*h/2
        S_rfp=np.append(S_rfp,s_t)
    Sp=np.sum(S_rfp)
    return(Sp)

# spn=get_area2rfp(rfp_lat_nt,lat_w,lon_w)
# sps=get_area2rfp(rfp_lat_st,lat_w,lon_w)
# print((spn+sps)/(4*np.pi*3480*3480))

print(len(e_lat))

area_s=get_area2rfp(rfp_lat,lat_w,lon_w)
s_rate=(area_s)/(4*np.pi*3480*3480)
print(s_rate)

area_e=get_area2rfp(e_lat,lat_w,lon_w)
se_rate=(area_s-area_e)/(4*np.pi*3480*3480)
print(se_rate)

######################################################################
#gmf_l=r'\cals3k.4_coef_mat'
gmf_l=r'\gufm_coef_mat'
flist=os.listdir(os.getcwd()+gmf_l)
Lme=3
Lf=14
theta=np.arange(0,180,1)
phi=np.arange(0,360,1)
lat_w=np.around(theta[1]-theta[0],3)
lon_w=np.around(phi[1]-phi[0],3)
[phim,thetam]=np.meshgrid(phi,theta)
# f_area_n=[]
# f_area_s=[]
f_area_r=[]
f_area_e=[]
f_year=[]
ck=0
for fp in flist:
    gmodel=loadmat(os.getcwd()+gmf_l+'\\'+fp)
    g=gmodel['g']
    h=gmodel['h']
    Br_me=get_Br(Lme, phim, thetam, g, h)
    Br_f=get_Br(Lf, phim, thetam, g, h)
    [mequt_lat,mequt_lon]=idf_mequt(Br_me,phi,theta)
    
    dif_lon=abs(np.around(mequt_lon[1:]-mequt_lon[0:-1],3))
    [dp_i_lon]=np.where((lon_w < dif_lon)& (dif_lon < np.around(phi[-1]-phi[0],3))) 
    if dp_i_lon.size==0:
        it_lon=len(mequt_lon)-1
    else:
        [iil]=np.where(dp_i_lon>=0.99*len(phi))
        it_lon=dp_i_lon[iil[0]]
        
    dif_lat=np.around(mequt_lat[1:]-mequt_lat[0:-1],3)
    [dp_i_lat]=np.where(abs(dif_lat)>60)
    if dp_i_lat.size==0:
        it_lat=len(mequt_lat)-1
    else:
        it_lat=dp_i_lat[0]
    
#    it_dp_i=it_lon
    it_dp_i=np.min([it_lon,it_lat])
    
    mequt_lon_p=mequt_lon[0:it_dp_i+1]
    mequt_lat_p=mequt_lat[0:it_dp_i+1]
    
    [rfp_lon,rfp_lat,e_lon,e_lat]=idf_rfp(mequt_lon_p,mequt_lat_p,Br_f,phi,theta)
    # Narea_temp=get_area2rfp(rfp_lat_nt,lat_w,lon_w)
    # Sarea_temp=get_area2rfp(rfp_lat_st,lat_w,lon_w)
    # #Narea_temp=len(rfp_lon_nt)
    # #Sarea_temp=len(rfp_lon_st)

    # f_area_n=np.append(f_area_n,Narea_temp)
    # f_area_s=np.append(f_area_s,Sarea_temp)
    
    area_r=get_area2rfp(rfp_lat,lat_w,lon_w)
    area_e=get_area2rfp(e_lat,lat_w,lon_w)
    #area_temp=len(rfp_lat)
    f_area_r=np.append(f_area_r,area_r)
    f_area_e=np.append(f_area_e,area_e)
    f_year=np.append(f_year,float(fp[0:-4]))
    
    ck=ck+1
    print(ck)

s_rate=(f_area_r-f_area_e)/(4*np.pi*3480*3480)
s_rate_with_edge=(f_area_r)/(4*np.pi*3480*3480)
#s_rate=(f_area)/(len(phi)*len(theta))
ify=np.argsort(f_year)
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
plot1=ax.plot(f_year[ify],s_rate_with_edge[ify],color='b')
ax.set_xlim(1840,2015)
#ax.set_ylim(0.08,0.24)
ax.set_ylim(0.08,0.24)    
# plt.xticks([])
# plt.yticks([])
ax.set_title('mel_max=3')            
ax.set_xlabel('year')
ax.set_ylabel('Area/s') 
plt.show()

#savemat(pwd+r'\\'+'sr_1d', {'sr':s_rate,'sr_we':s_rate_with_edge,'fyr':f_year})

    




