from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator, interp1d, RegularGridInterpolator
import struct

# Note in this code there are 2 frames of reference:
#   XYZ = coordinate system defined in the PALM simulation of Boston
#   RST = coordinate system in the frame of vehicle (r aft, s starboard, t up)

# setting global variables from helios inputs.py file
rinf = 1.224
gamma = 1.4                        # ratio of specific heats
rgas = 287.05                   # Gas constant,      J/kg/K
rinf = 1.1397633                       # density,           kg/m^3
pinf = 93658.360000000000582                # pressure,          N/m^2
tinf = pinf/(rgas*rinf)                  # Temperature,       K
viscosity = 0.0000178029                # dynamic viscosity, N.s/m^2
ainf = np.sqrt(gamma*pinf/rinf)             # Speed of sound
mach = 0.2

# CFD simulation inputs
rpm = 4000
cfdgridunit = 0.001
dpsi = 90.00
dtcfd = dpsi/(rpm*60)
xyzlo = [-16,16,16] # minimum pt in domain
Lxdom = 32 # lengths of cfd domain
Lydom = 32 # lengths of cfd domain
Lzdom = 32 # lengths of cfd domain
nr = 128  # number of probes in the aft dir
ns = 128  # number of probes in starboard dir
nt = 128  # number of probes in vertical dir

# Vehicle trajectory information
Ltime = 20 # length of trajectory
vh_s0 = [175,25,45] # this is the vertiport location (meters)

def main():
    run = 'Beach4pm'
    global clim
    match run:
        case 'Beach2am':
            file_id = Dataset("/projectnb/turbomac/Emmanuel_PALM_simulation_output/vertiport_40_beach_street_child3_2am_24th_July_time_5mins.nc")
            clim = [-1,1]    
        case 'Beach9am':   
            file_id = Dataset("/projectnb/turbomac/Emmanuel_PALM_simulation_output/vertiport_40_beach_street_child3_9am_24th_July_time_5mins.nc")
            clim = [-1.5,1.5]
        case 'Beach4pm':
            file_id = Dataset("/projectnb/turbomac/Emmanuel_PALM_simulation_output/vertiport_40_beach_street_child2_4pm_24th_July")
            clim = [-10,10]

    # Read PALM coordinate data
    x = np.asarray(file_id.variables['x'])
    y = np.asarray(file_id.variables['yv'])
    z = np.asarray(file_id.variables['zu_3d'])
    time = np.asarray(file_id.variables['time'])
    
    # DEBUG: OUTPUT PALM INFLOW
#    print("Variables: ")
#    print(file_id.variables)
#    print('Time = ',time)
#    print('nsteps = ',time.shape)
#    print(f"x limits: {x[0]} to {x[-1]}")
#    print(f"y limits: {y[0]} to {y[-1]}")
#    print(f"z limits: {z[0]} to {z[-1]}")


    # start at different timesteps to get different disturbances
    for t_start in [1]: #, 500, 1000, 1500, 2000, 2500, 3000]:
        # DEBUG THIS XXX not sure if right
        print("time start index: ", t_start)
        t_end = np.argsort(abs(time-(time[t_start]+Ltime)))[0]
        ntime = t_end-t_start+1
        ftime = time[t_start:t_end+1]-time[t_start]
        tcfd = np.arange(0,ftime[-1],dtcfd) 
        nstepcfd = len(tcfd)

        #==================================================
        # COMPUTE THE FLIGHT TRAJECTORY
        #==================================================
        vh_s,vh_v,vh_a = computeTrajectory(vh_s0,45,ftime)

        #==================================================
        # GENERATE THE HELIOS ABL INITIAL CONDITION FILE
        #==================================================
        # get vehicle direction vectors at time 0 
        r_hat,s_hat,t_hat = getRSTVectors(ftime,vh_s,0,1)   

        # define grid around vehicle in RST coord sys
        vecr = np.linspace(-Lxdom/2,Lxdom/2,nr+1)
        vecs = np.linspace(-Lydom/2,Lydom/2,ns+1)
        vect = np.linspace(-Lzdom/2,Lzdom/2,nt+1)
        rgrids,sgrids,tgrids = np.meshgrid(vecr,vecs,vect,indexing='ij') 
        xyz = (vh_s[0,:]
          + rgrids[..., np.newaxis] * r_hat
          + sgrids[..., np.newaxis] * s_hat
          + tgrids[..., np.newaxis] * t_hat)

        # interpolate velocities from PALM data
        q = np.zeros((nr+1,ns+1,nt+1,5))
        U,V,W = extractPALM(file_id,xyz,0,x,y,z)

        # convert velocities to RST
        vec = np.stack([U, V, W], axis=-1) + vh_v[0,:]
        q[:,:,:,0] = 1.0
        q[:,:,:,1] = (vec @ r_hat) / ainf
        q[:,:,:,2] = (vec @ s_hat) / ainf
        q[:,:,:,3] = (vec @ t_hat) / ainf
        vmag = np.linalg.norm(q[:,:,:,1:4]/q[:,:,:,0:1], axis=-1)  
        q[:,:,:,4] = 1/(gamma*(gamma-1)) + 0.5*vmag**2

        print('writing IC file. q = ',q.shape)
        writeP3D(rgrids/cfdgridunit,sgrids/cfdgridunit,tgrids/cfdgridunit,q,'initfield')

        #==================================================
        # GENERATE HELIOS ABL TIME SERIES FILE       
        #==================================================
        print('generating time series file')
        q = np.zeros((ns+1,nt+1,ntime,5))  
        for k in range(ntime):
            print("Plane " + str(k+1) + " out of " + str(ntime) + "...")
    
            # For each step in the trajectory, create a plane of probes to save and
            # calculate the location in x,y,z, closest each probe location
            # at each timestep
            if(k==0):
                k1 = k
                k2 = k+1
            elif(k==ntime-1):
                k1 = k-1
                k2 = k
            else:
                k1 = k-1
                k2 = k+1    
            r_hat,s_hat,t_hat = getRSTVectors(ftime,vh_s,k1,k2)

            # compute the XYZ coordinates of the upstream plane
            vecr = np.linspace(-Lxdom/2,Lxdom/2,nr+1)
            vecs = np.linspace(-Lydom/2,Lydom/2,ns+1)
            vect = np.linspace(-Lzdom/2,Lzdom/2,nt+1)
            rgrids,sgrids,tgrids = np.meshgrid(vecr,vecs,vect,indexing='ij') 
            xyz = (vh_s[k,:]
              + rgrids[..., np.newaxis] * r_hat
              + sgrids[..., np.newaxis] * s_hat
              + tgrids[..., np.newaxis] * t_hat)

            U,V,W = extractPALM(file_id,xyz[0:1,:,:,:],k,x,y,z) 

            # convert velocities to RST
            vec = (np.stack([U, V, W], axis=-1) + vh_v[min(k,ntime-2),:])[0] 

            q[:,:,k,0] = 1.0
            q[:,:,k,1] = (vec @ r_hat) / ainf
            q[:,:,k,2] = (vec @ s_hat) / ainf
            q[:,:,k,3] = (vec @ t_hat) / ainf
            vmag = np.linalg.norm(q[:,:,k,1:4]/q[:,:,k,0:1], axis=-1)  
            q[:,:,k,4] = 1/(gamma*(gamma-1)) + 0.5*vmag**2

        # Write out data to plot3D 
        # Helios ABL routine reads time series in the inlet plane ref frame
        # x = starboard, y = vertical up, z = time
        q = np.asarray(q[:,:,:,[0,2,3,1,4]],dtype=np.float64)
        
        # check if any nans at the end
        print("Indices of nans in solution: ")
        print(np.where(np.isnan(q))[0])

        # Here we use Pchip interpolation to resample time series data 
        # to match Helios CFD's timestep. Helps avoid interpolation
        # issues with Helios ABL's linear interpolation
        # 
        # q = size(nstarboard,nvertical,ntime,5)
        # qi = size(nstarboard,nvertical,ntimeCFD,5)
        qi = np.zeros((ns+1,nt+1,nstepcfd,5))
        print('resampling to ',qi.shape)
        for i in range(ns+1): 
            for j in range(nt+1):
                for v in range(5):
                    pchip = PchipInterpolator(ftime,q[i,j,:,v])
                    qi[i,j,:,v] = pchip(tcfd)
        
        # Write to PLOT3D
        fname = run + '_Disturbance_tStart' + str(t_start).zfill(5) 
        print('writing' + fname)
        vecr = np.linspace(0,nstepcfd*Lxdom/ns,nstepcfd)
        vecs = Lydom*np.linspace(0,1,ns+1)
        vect = Lzdom*np.linspace(0,1,nt+1)
        sgrids,tgrids,rgrids = np.meshgrid(vecs,vect,vecr,indexing='ij') 
        writeP3D(sgrids/cfdgridunit,tgrids/cfdgridunit,rgrids/cfdgridunit,qi,fname)
               
def plotTrajectoryField(fn,file_id,tstart,t_hat,time,x,y,z,vh_s):
    ind = [0,0,0]
    for j, coord in enumerate((x, y, z)):
        # cut planes through the vertiport
        ind[j] = int(np.argmin(abs(coord - vh_s[-1,j])))
    nx = len(x)
    ny = len(y)
    nz = len(z)
    for s_hat in range(3):     
        match s_hat:
            case 0:
                xvals = ind[0]
                yvals = range(ny)
                zvals = range(nz)
                px = y
                py = z
                vx = vh_s[:,1]
                vy = vh_s[:,2]
                xl = "Y (m)"
                yl = "Z (m)"
                fname = fn + "_XPlane_t" + str(t_hat).zfill(5) + ".png"
            case 1:
                xvals = range(nx)
                yvals = ind[1]
                zvals = range(nz)
                px = x
                py = z
                vx = vh_s[:,0]
                vy = vh_s[:,2]
                xl = "X (m)"
                yl = "Z (m)"
                fname = fn + "_YPlane_t" + str(t_hat).zfill(5) + ".png"
            case 2:
                xvals = range(nx)
                yvals = range(ny)
                zvals = ind[2]
                px = x
                py = y
                vx = vh_s[:,0]
                vy = vh_s[:,1]
                xl = "X (m)"
                yl = "Y (m)"
                fname = fn + "_ZPlane_t" + str(t_hat).zfill(5) + ".png"
        
        # slice data
        u = np.asarray(file_id.variables['u'][t_hat+tstart,zvals,yvals,xvals])
        v = np.asarray(file_id.variables['v'][t_hat+tstart,zvals,yvals,xvals])
        w = np.asarray(file_id.variables['w'][t_hat+tstart,zvals,yvals,xvals])
        speed = np.sqrt(u*u + v*v + w*w)

        # plot data
        X, Y = np.meshgrid(px,py)
        fig, ax = plt.subplots()
        contour_levels = np.linspace(clim[0],clim[1],25)
#        cfig = plt.contourf(X,Y,speed,extend='both')
        cfig = plt.contourf(X,Y,speed,levels=contour_levels,extend='both')
        plt.plot(vx,vy,color='k',marker='x',linestyle='None',linewidth=4.5)
        plt.plot(vx[t_hat],vy[t_hat],color='red',marker='o',linewidth=2.5,markersize=15)
        plt.plot(vx[-1],vy[-1],color='red',marker='x',linewidth=2.5,markersize=15)
        plt.xlabel(xl)
        plt.ylabel(yl)
        plt.title("Time = " + str(time[t_hat]) + " (sec)")
        ax.set_aspect('equal')
        fig.colorbar(cfig,ax=ax,label='Speed')     
        plt.savefig(fname)
        plt.close('all')

def getRSTVectors(ftime,vh_s,k1,k2):
    dt = ftime[k2]-ftime[k1]
    dx = (vh_s[k2,0]-vh_s[k1,0])/dt
    dy = (vh_s[k2,1]-vh_s[k1,1])/dt
    dz = (vh_s[k2,2]-vh_s[k1,2])/dt

    d_hat = np.array([dx,dy,dz])
    d_hat = -d_hat/np.linalg.norm(d_hat) # direction unit vector

    t_hat = np.array([0,0,1]) # vertical unit vector fixed
    s_hat = np.cross(t_hat,d_hat) # lateral unit vector
    s_hat = s_hat/np.linalg.norm(s_hat)
    r_hat = np.cross(s_hat,t_hat) # aft unit vector
    r_hat = r_hat/np.linalg.norm(r_hat)

    return r_hat,s_hat,t_hat

def computeTrajectory(vh_s0,azi,ftime):
    # vertiport approach acceleration components
    # |vh_a| = 0.1*9.81         0.1g from Jeremy Bain
    # sqrt(vx**2+vy**2)=sqrt(2*vx**2)=8*vz    8:1 horizontal:vertical descent ratio from FAA vertiport approach guidelines
    vh_a = np.zeros(3)
    vh_a[2] = np.sqrt((0.1*9.81)**2/8**2) # vertical acceleration
    vh_a[1] = 8*vh_a[2]*np.sin(np.pi*azi/180) 
    vh_a[0] = 8*vh_a[2]*np.cos(np.pi*azi/180)

    # vehicle displacement array
    vh_s = np.zeros((len(ftime),3))

    # setting vertiport location as t0 and calculating departure as it's_hat simpler
    vh_s[0]=vh_s0

    # need the 0:1 slice here to keep from flattening to 1D automatically
    #                  |
    #                  v                 newaxis allows shape [ntime,3]
    vh_s[1:,:] = vh_s[0:1,:] + 0.5*vh_a*ftime[1:,np.newaxis]**2

    # flip in time to make it approach, not departure
    vh_s = np.flip(vh_s, axis=0)
    #print("Initial vehicle location: ", vh_s0,vh_s[0])

    # plot trajectory over initial flowfield
#    fname = 'InitialField_tStart' + str(t_start) + '_Azi' + str(azi)
#    plotTrajectoryField(fname,file_id,t_start+i,time,x,y,z,vh_s)

    # calculate vehicle velocity from displacement
    vh_v = np.diff(vh_s,axis=0) / np.diff(ftime,axis=0)[:,np.newaxis]
    return vh_s, vh_v, vh_a

def extractPALM(file_id,xyz,t,x,y,z):
    n0 = xyz.shape[0]
    n1 = xyz.shape[1]
    n2 = xyz.shape[2]

    # get the outerbounds of the required data
    xmin,xmax = xyz[:,:,:,0].min(), xyz[:,:,:,0].max()
    ymin,ymax = xyz[:,:,:,1].min(), xyz[:,:,:,1].max()
    zmin,zmax = xyz[:,:,:,2].min(), xyz[:,:,:,2].max()
    ix0 = max(np.searchsorted(x,xmin)-2, 0)
    ix1 = min(np.searchsorted(x,xmax)+2, len(x)-1)
    iy0 = max(np.searchsorted(y,ymin)-2, 0)
    iy1 = min(np.searchsorted(y,ymax)+2, len(y)-1)
    iz0 = max(np.searchsorted(z,zmin)-2, 0)
    iz1 = min(np.searchsorted(z,zmax)+2, len(z)-1)

    # get the PALM flowfield
    u_sub = np.asarray(file_id.variables['u'][t, iz0:iz1, iy0:iy1, ix0:ix1])
    v_sub = np.asarray(file_id.variables['v'][t, iz0:iz1, iy0:iy1, ix0:ix1])
    w_sub = np.asarray(file_id.variables['w'][t, iz0:iz1, iy0:iy1, ix0:ix1])
    x_sub, y_sub, z_sub = x[ix0:ix1], y[iy0:iy1], z[iz0:iz1]

    # mask PALM fill values
    u_sub[u_sub < -9000] = 0.0
    v_sub[v_sub < -9000] = 0.0
    w_sub[w_sub < -9000] = 0.0

    # generate UVW interpolators    
    Ufunc = RegularGridInterpolator((x_sub, y_sub, z_sub), u_sub.T, method='linear',bounds_error=False,fill_value=0.0)
    Vfunc = RegularGridInterpolator((x_sub, y_sub, z_sub), v_sub.T, method='linear',bounds_error=False,fill_value=0.0)
    Wfunc = RegularGridInterpolator((x_sub, y_sub, z_sub), w_sub.T, method='linear',bounds_error=False,fill_value=0.0)

    pts = xyz.reshape(-1,3)  
    U = Ufunc(pts).reshape(n0,n1,n2)
    V = Vfunc(pts).reshape(n0,n1,n2)
    W = Wfunc(pts).reshape(n0,n1,n2)

    return U,V,W

def writeP3D(X,Y,Z,q,fname):
    imax, jmax, kmax = X.shape

    with open(fname + ".xyz", "wb") as f:
        f.write(struct.pack(">i", 1))
        f.write(struct.pack(">iii", imax, jmax, kmax))
        f.write(X.flatten(order='F').astype('>f8').tobytes())
        f.write(Y.flatten(order='F').astype('>f8').tobytes()) 
        f.write(Z.flatten(order='F').astype('>f8').tobytes())

    with open(fname + ".q", "wb") as f:
        f.write(struct.pack(">i", 1))
        f.write(struct.pack(">iii", imax, jmax, kmax))
        f.write(struct.pack('>4d', mach, 0.0, 0.0, 0.0))
        for comp in range(5):
            f.write(
                q[:, :, :, comp]
                .flatten(order='F')
                .astype('>f8')
                .tobytes()
            )

def write_record(f, data_bytes):
    """Write a big-endian Fortran unformatted record."""
    nbytes = len(data_bytes)
    f.write(struct.pack(">i", nbytes))
    f.write(data_bytes)
    f.write(struct.pack(">i", nbytes))

if __name__=="__main__":
    main()

