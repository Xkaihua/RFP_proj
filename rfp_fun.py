# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 13:47:34 2021

@author: xukai
"""
import numpy as np

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


def get_Br(L,phim,thetam,g,h,r):
# calculate Br on CMB
# input
#      L: max degree to calculate Br
#      phim: colatitude(unit:degree) of the grid points
#      thetam: longitude(unit:degree) of the grid points
#      g,h: gauss coefficients g and h
#      r: radius used to calculate the field, default is 3480
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


def idf_mequt(Brm,phi,theta):
# identify the magnetic equator from Br matix
#input 
#     Brm: Br matrix
#     phi: longitude grid array
#     theta: colatitude grid array
#output
#     mequt_lat_p: colatitude of the points in magnetic equator 
#     mequt_lon_p: longitude of the points in magnetic equator
    lon_w=np.around(phi[1]-phi[0],3)
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
        u_lon_c[k]=np.sum(ilon==phi[i])
        u_lon_o[k]=phi[i]
        k=k+1
    sc=np.argsort(u_lon_c)
    u_lon_c_sort=u_lon_c[sc]
    u_lon_o_sort=u_lon_o[sc]
    
    sta_lon=u_lon_o_sort[0]
    mequt_lat=np.zeros(1)
    mequt_lon=np.zeros(1)
    [il]=np.where(ilon==sta_lon)
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
        mequt_lat=np.append(mequt_lat,temp_x[nxt_p])
        mequt_lon=np.append(mequt_lon,temp_y[nxt_p])
        #mequt_lon[j+1]=temp_y[nxt_p]
        temp_y=np.delete(temp_y,nxt_p)
        temp_x=np.delete(temp_x,nxt_p)
        
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
    
    
    mequt_lon_p=mequt_lon[0:it_dp_i+1]
    mequt_lat_p=mequt_lat[0:it_dp_i+1]
          
    return(mequt_lat_p,mequt_lon_p)


def idf_rfp(mequt_lon_p,mequt_lat_p,Brm_l10,phi,theta):
# identify the reversed flux patches from Br matix
#input 
#     mequt_lat_p: colatitude of the points in magnetic equator 
#     mequt_lon_p: longitude of the points in magnetic equator   
#     Brm_l10: Br matrix
#     phi: longitude grid array
#     theta: colatitude grid array
#output
#     rfp_lat: colatitude of the points in reversed flux patches
#     rfp_lon: longitude of the points in reversed flux patches
    rfp_lat=[]
    rfp_lon=[]
    e_lat=[]
    e_lon=[]
    phi=np.around(phi,5)
    theta=np.around(theta,5)
    for i in np.arange(len(phi)):
#        tag_lon=np.around(phi[i],5)
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
                [i_lat]=np.where(theta==np.around(tag_lat[j],5))
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
                elif ir.size==1:
                    e_lat=np.append(e_lat,theta[r1:r2][ir[-1]])
                    e_lon=np.append(e_lon,phi[i]*np.ones(1))
                
                p=p+1
                r1=r2+1
                
            [ir]=np.where(Brm_l10[r2+1:,i]<0) 
            rfp_lat=np.append(rfp_lat,theta[r2+1:][ir[:]])
            rfp_lon=np.append(rfp_lon,phi[i]*np.ones(len(ir)))
            
            if ir.size>0:
                e_lat=np.append(e_lat,theta[r2+1:][ir[0]])
                e_lat=np.append(e_lat,theta[r2+1:][ir[-1]])
                e_lon=np.append(e_lon,phi[i]*np.ones(2))
            elif ir.size==1:
                e_lat=np.append(e_lat,theta[r2+1:][ir[-1]])
                e_lon=np.append(e_lon,phi[i]*np.ones(1))
        
    return(rfp_lon,rfp_lat,e_lon,e_lat)    


################
### test code of identifying the rfp

# def idf_rfp_test(mequt_lon_p,mequt_lat_p,Brm_l10,phi,theta):
#     rfp_lat_st=[]
#     rfp_lat_nt=[]
#     rfp_lon_st=[]
#     rfp_lon_nt=[]
#     for i in np.arange(len(mequt_lon_p)):

#         [i_lat]=np.where(theta==np.around(mequt_lat_p[i],5))
#         [i_lon]=np.where(phi==np.around(mequt_lon_p[i],5))

#         if i_lon.size==0:
#             print(mequt_lon_p[i])
#             return
#         t_lat=i_lat[0]
#         t_lon=i_lon[0]
#         Br_north=Brm_l10[0:t_lat,t_lon]
#         Br_south=Brm_l10[t_lat+1:,t_lon]
#         [i_n]=np.where(Br_north>0)
#         [i_s]=np.where(Br_south<0)
#         rfp_lat_nt=np.append(rfp_lat_nt,theta[0:t_lat][i_n])
#         rfp_lat_st=np.append(rfp_lat_st,theta[t_lat+1:][i_s])
#         rfp_lon_nt=np.append(rfp_lon_nt,t_lon*np.ones(len(i_n)))
#         rfp_lon_st=np.append(rfp_lon_st,t_lon*np.ones(len(i_s)))
#     return(rfp_lon_nt,rfp_lat_nt,rfp_lon_st,rfp_lat_st)

def get_area2rfp(rfp_lat,lat_w,lon_w):
## calculate the area of the reversed flux patches
#input 
#     rfp_lat: colatitude of the points in reversed flux patches
#       lat_w: latitude grid space 
#       lon_w: longitude grid space
#output 
#         Sp: area of the reversed flux patches grid points
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