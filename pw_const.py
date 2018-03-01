from scipy.integrate import odeint

import numpy as np
import math
import matplotlib.pylab as mp
import sys
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', serif=['Computer Modern Roman'])
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from xppcall import xpprun



c = 16./np.sqrt(2)
tarr = np.loadtxt('oct_x1.dat')[:,0]
x1 = np.loadtxt('oct_x1.dat')[:,1]
y1 = np.loadtxt('oct_y1.dat')[:,1]


#npa,vn = xpprun('limit_cycle_pw_const_coupled.ode')

def M(region):
    """
    region is the jump matrix number not the region number.
    """
    if region == 1:
        return np.array([[c/16.,c/16.],[-1.+c/16.,1.+c/16.]])
    if region == 2:
        return np.array([[16./c,1.],[0.,1.]])
    if region == 3:
        return np.array([[1.+c/16.,1.-c/16.],[-c/16.,c/16.]])
    if region == 4:
        return np.array([[1.,0.],[-1.,16./c]])
    if region == 5:
        return np.array([[c/16.,c/16.],[-1.+c/16.,1.+c/16.]])
    if region == 6:
        return np.array([[16./c,1.],[0.,1.]])
    if region == 7:
        return np.array([[1.+c/16.,1.-c/16.],[-c/16.,c/16.]])
    if region == 8:
        return np.array([[1.,0.],[-1.,16./c]])

def oct_phase_reset_analytic(phi,dt=.001):

    # get all jump values
    someval = 1-np.sqrt(2)
    prcvalues = np.zeros((8,2))
    prcvalues[0,:] = np.array([1.,1./someval])/16.

    for R in range(1,8):
        prcvalues[R,:] = np.dot(M(R+1),prcvalues[R-1,:])
        
    sol = np.zeros((len(phi),2))

    iter_per_region = len(sol)/8

    Rtot = 8

    pfinal = phi[-1]

    j = 0

    for i in range(len(phi)):
        j = i+1
        p = phi[i]
        if (p >= 0) and (p < pfinal/8.):
            sol[i,:] = prcvalues[0]
        elif (p >= pfinal/8.) and (p < 2.*pfinal/8.):
            sol[i,:] = prcvalues[1]

        elif (p >= 2.*pfinal/8.) and (p < 3.*pfinal/8.):
            sol[i,:] = prcvalues[2]
        elif (p >= 3.*pfinal/8.) and (p < 4.*pfinal/8.):
            sol[i,:] = prcvalues[3]

        elif (p >= 4.*pfinal/8.) and (p < 5.*pfinal/8.):
            sol[i,:] = prcvalues[4]
        elif (p >= 5.*pfinal/8.) and (p < 6.*pfinal/8.):
            sol[i,:] = prcvalues[5]

        elif (p >= 6.*pfinal/8.) and (p < 7.*pfinal/8.):
            sol[i,:] = prcvalues[6]
        elif (p >= 7.*pfinal/8.) and (p <= 8.*pfinal/8.):
            sol[i,:] = prcvalues[7]

    return sol

def oct_phase_reset(phi, dx=0., dy=0., steps_per_cycle = 200000,
                    num_cycles = 3, return_intermediates=False,
                    ode_ver='fast'):

    if ode_ver == 'fast':
        fn = 'limit_cycle_pw_const_coupled.ode'
        T = 1.
    elif ode_ver == 'slow':
        fn = 'limit_cycle_pw_const_coupled_slower.ode'
        T = 16.
    else:
        raise ValueError('invalid choice =', ode_ver)


    print phi
    # total period

    dt = float(T)/steps_per_cycle

    steps_before = int(phi/(2*math.pi) * steps_per_cycle) + 1

    # run up to perturbation
    npa,vn = xpprun(fn,
                    inits={'x1':1,'y1':2.41421},
                    parameters={'meth':'euler',
                                'dt':dt,
                                'eps':0.,
                                'total':steps_before*dt},
                    clean_after=True)
    t1 = npa[:,0]
    vals1 = npa[:,1:3]
    
    # run after perturbation
    steps_after = steps_per_cycle * num_cycles - steps_before
    npa,vn = xpprun(fn,
                    inits={'x1':vals1[-1,0]+dx,'y1':vals1[-1,1]+dy},
                    parameters={'meth':'euler',
                                'dt':dt,
                                'eps':0.,
                                'total':steps_after*dt},
                    clean_after=True)
    t2 = npa[:,0]+T*phi/(2*math.pi)
    vals2 = npa[:,1:3]

    x_section = 1.

    crossings = ((vals2[:-1,0] <= x_section) * (vals2[1:,0] > x_section)
                 * (vals2[1:,1] > 0))

    if len(crossings) == 0:
        raise RuntimeError("No complete cycles after the perturbation")

    crossing_fs = ((vals2[1:,0][crossings] - x_section)
            / (vals2[1:,0][crossings]-vals2[:-1,0][crossings]) )
    crossing_times = (crossing_fs * t2[:-1][crossings] 
            + (1-crossing_fs) * t2[1:][crossings])

    #crossing_times = crossing_times - crossing_times[0]

    crossing_phases = np.fmod(crossing_times, T)# * 2 * math.pi

    print crossing_phases

    #crossing_phases[crossing_phases > math.pi] -= 2*math.pi
    crossing_phases[crossing_phases > T/2.] -= T

    crossing_phases /= T


        
    if return_intermediates:
        return dict(t1=t1, vals1=vals1, t2=t2, vals2=vals2,
                    crossings=crossings,
                    crossing_times=crossing_times,
                    crossing_phases=crossing_phases)
    else:
        return -crossing_phases[-1]


def oct_lc(steps_per_cycle=5000,ode_ver='fast'):
    """
    return the limit cycle.
    """
    T = 1.
    dt = float(T)/steps_per_cycle
    total=steps_per_cycle*dt

    if ode_ver == 'fast':
        fn = 'limit_cycle_pw_const_coupled.ode'
        T = 1.
    elif ode_ver == 'slow':
        fn = 'limit_cycle_pw_const_coupled_slower.ode'
        T = 16.
    else:
        raise ValueError('invalid choice =',ode_ver)

    npa,vn = xpprun(fn,
                    inits={'x1':-1.,'y1':2.41421},
                    parameters={'meth':'euler',
                                'dt':dt,
                                'eps':0.,
                                'total':total},
                    clean_after=True)
    
    t = npa[:,0]
    vals = npa[:,1:3]

    return t,vals

def G(x,y,g=5.):
    #osc1 (x1,y1)
    #osc2 (x2,y2)
    x1,y1 = x
    x2,y2 = y

    coord1 = g*(x2-x1)
    coord2 = g*(y2-y1)
    
    return coord1,coord2

def generate_h(phi,lc1,lc2,prc1,prc2):
    """
    lc,prc are the interp1d functions from def main
    """
    N = len(phi)
    
    sol = np.zeros(N)
    
    for i in range(N):
        tot = 0
        for k in range(N):
            sp = phi[i]
            s = phi[k]
            
            g1,g2 = G([lc1(np.mod(s,phi[-1])),lc2(np.mod(s,phi[-1]))],
                      [lc1(np.mod(s+sp,phi[-1])),lc2(np.mod(s+sp,phi[-1]))])
            tot += prc1(np.mod(s,phi[-1]))*g1 + prc2(np.mod(s,phi[-1]))*g2

        sol[i] = tot
    sol *= (phi[-1]/N)/phi[-1]

    return sol


def generate_h_bad(phi,lc1,lc2,prc1,prc2):
    """
    lc,prc are the interp1d functions from def main
    """
    N = len(phi)
    
    sol = np.zeros(N)

    for i in range(N):
        tot = 0
        for k in range(N):
            sp = phi[i]
            s = phi[k]
            
            g1,g2 = G([lc1(np.mod(s,phi[-1])),lc2(np.mod(s,phi[-1]))],
                      [lc1(np.mod(s+sp,phi[-1])),lc2(np.mod(s+sp,phi[-1]))])
            tot += 1+0*prc1(np.mod(s,phi[-1]))*g1 + 2+0*prc2(np.mod(s,phi[-1]))*g2

        sol[i] = tot
    sol *= (phi[-1]/N)/phi[-1]

    return sol


def phase_rhs(y,t,hfun):
    """
    hfun is the lookup table generated using interp1d
    """
    val = hfun(np.mod(-y,1))-hfun(np.mod(y,1))

    return .01*val

def phase2sv():
    pass

def main():
    
    show_num = True

    if show_num:

        # total perturbations
        total_perts = 20

        # phase values where perturbations will be applied

        #dx = 1e-4
        #dy = 0

        pert = 1e-2 # keep perturbation to single variable for now
        print "Calculating iPRC via direct method for perturbations in the x direction..."
        phis = np.linspace(0,1-1./total_perts,total_perts)

        x_prc = np.array([
            oct_phase_reset(phi*2*np.pi, dx=pert, dy=0)
            for phi in phis])
        if pert != 0.0:
            x_prc /= pert
        #x_prc = np.roll(x_prc,int(len(phis)*2./16.))

        #x_prc = np.zeros(len(phis))

        y_prc = np.array([
            oct_phase_reset(phi*2*np.pi, dx=0, dy=pert,ode_ver='slow')
            for phi in phis])
        if pert != 0.0:
            y_prc /= pert

        #y_prc = np.roll(y_prc,int(len(phis)*2./16.))

        # create prc lookup table
    
    #phis = np.linspace(0,2*math.pi,total_perts)
    phi = np.linspace(0,1,500)
    prc = oct_phase_reset_analytic(phi)

    
    prc1 = interp1d(phi,prc[:,0])
    prc2 = interp1d(phi,prc[:,1])
    
    # create lc lookup table
    t,vals = oct_lc()
    lc1 = interp1d(t,vals[:,0])
    lc2 = interp1d(t,vals[:,1])

    # numerics
    #mp.figure(figsize=(6,3))
    #mp.scatter(phis,x_prc,label=r'$z_x$ (numerics)',color='blue',s=20,zorder=5)
    #mp.scatter(phis,y_prc,label=r'$z_y$ (numerics)',color='green',s=20,alpha=.5,zorder=3)
    #mp.title('PRCs')
    #mp.plot(phi,prc1(phi),label='prcx')
    #mp.plot(phi,prc2(phi),label='prcy')
    #mp.plot(phis,oct_phase_reset_analytic(phis)[:,1])

    #mp.xlabel('Phase',fontsize=20)
    #mp.ylabel(r'$z$',fontsize=20)
    #mp.legend(loc=4,fontsize=20)
    #plt.xticks(fontsize=18)

    # theory + numerics
    mp.figure(figsize=(6,3))

    if show_num:
        mp.scatter(np.mod(phis+1./8,1),x_prc,label=r'$z_x$ (numerics)',color='blue',s=20)
        mp.scatter(np.mod(phis+1./8,1),y_prc,label=r'$z_y$ (numerics)',color='green',s=20)
    #mp.title('PRCs')
    mp.plot(phi,prc1(phi),label=r'$z_x$ (theory)',color='black',lw=3)
    mp.plot(phi,prc2(phi),label=r'$z_y$ (theory)',color='gray',lw=3)
    #mp.plot(phis,oct_phase_reset_analytic(phis)[:,1])

    mp.xlabel('Phase',fontsize=15)
    mp.ylabel(r'$z$',fontsize=15)
    #mp.legend(loc=4,fontsize=10)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)


    """
    mp.figure()
    phi2 = np.linspace(0,1,100)
    hvals = generate_h(phi2,lc1,lc2,prc1,prc2)

    h = interp1d(phi2,hvals)
    mp.title('-2hodd for pwc')
    mp.plot(phi2,h(np.mod(-phi2,phi2[-1]))-h(phi2),lw=2)

    t = np.linspace(0,100,10000)
    sol = odeint(phase_rhs,.49,t,args=(h,))
    mp.figure()
    mp.plot(t,sol)
    """
    
    plt.tight_layout()
    
    mp.show()

if __name__ == "__main__":
    main()
