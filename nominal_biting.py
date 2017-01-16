"""
Written by Youngmin Par MS/BS CWRU 2013
This is my attempt to reproduce the system constructed by Shaw et al.
"""
import numpy as np
import matplotlib.pyplot as mp
import math
import sys

from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', serif=['Computer Modern Roman'])
from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate

mu = .1 # "distance" from heteroclinic bifurcation
a = .01 # PWL bifurcation param
rho = 3 # coupling strength

def smooth(a,t,rho=3.):
    """
    a: A list of three elements, each state variable representing a single "node".
    t: Time
    """
    #print a
    #a[1]
    a[a<=0]= 0.0
    return [a[0]-a[0]**2+mu-rho*a[0]*a[1],
            a[1]-a[1]**2+mu-rho*a[1]*a[2],
            a[2]-a[2]**2+mu-rho*a[2]*a[0]]

def vectorField(R,init=[],a=0.01,rho=3.):
    x,y,z = init[0],init[1],init[2]
    return [1. - x - (a + y)*rho, a + y, (-a + z)*(1. - rho)]

def piecewise(x,t,a=0.01,rho=3.):
    """
    The piecewise linear system is defined within a division of the square.
    The system is derived from the smooth system (above), by linearizing about fixed points
    x: A list of three elements.  State variables.
    t: Time
    LaTex for equations when $\mu = 0$:
    Df(1,0,0)(x) = \matrix{c}{-x_1 -x_2 \rho \\ x_2 \\ x_3(1-\rho)}
    Df(0,1,0)(x) = \matrix{c}{-x_1(1-\rho) \\ -x_2-x_3 \rho \\ x_3}
    Df(0,0,1)(x) = \matrix{c}{x_1 \\ x_2(1-\rho) \\ -x_3 - x_1 \rho}
    
    The region about the saddle $(1,0,0)$ is bounded in the following way:
    $y \leq x$, $z \leq x$, and there is no relationship between $z$ and $y$.

    The region about the saddle $(0,1,0)$ is bounded in the following way:
    $y \geq z$, $y \geq x$, and there is no relationship between $x$ and $y$.
    
    The region about the saddle $(0,0,1)$ is bounded in the following way:
    $z \geq x$, $z \geq y$ and there is no relationship between $x$ and $y$.
    """
    #print x
    
    # impose boundary condition
    x[x<=0] = 0.0
    #print x
    # 

    if x[0] >= x[1]+a and x[0] >= x[2]-a:
        #print 'x'
        # saddle at (1,0,0)
        # region 1
        return [-(x[0])-(x[1]+a)*rho+1.,
                 x[1]+a,
                 (x[2]-a)*(1.-rho)]
    elif x[1] > x[0]-a and x[1] >= x[2]+a:
        # saddle at (0,1,0)
        #print 'y', x[0]*(1.-rho),-x[1]-x[2]*rho
        # region 2
        return [(x[0]-a)*(1.-rho),
                -(x[1])-(x[2]+a)*rho+1.,
                x[2]+a]
    elif x[2] > x[0]+a and x[2] > x[1]-a:
        # saddle at (0,0,1)
        #print 'z'
        # region 3
        return [x[0]+a,
                (x[1]-a)*(1.-rho),
                -x[2]-(x[0]+a)*rho+1.]
    

"""
def piecewise_local(x,t,a=0.01,rho=10.):
    # if at boundary, reinject
    if (x[2] <= x[0]) or (x[2] <= x[1]):
        print x,t
        x[0] = .05#x[1]+a
        x[1] = .05#x[0]-a*math.sqrt(2)
        x[2] = .8#x[2]-a*math.sqrt(2)

    return [x[0],
            x[1]*(1.-rho),
            -x[2]-x[0]*rho+1]
"""


def LinInterp(R,Rp1,pt1,pt2,a=0.01):
    """
    :R: region number
    :Rp1: next region -- it must be R+1
    :pt1:
    :pt2:
    Find the intersection of two line segments.  One determined by
    the boundary, and the other determined by pt1 and pt2.
    """
    assert ((Rp1-1)%3 == R)
    
    # slope
    m = (pt1[1]-pt2[1])/(pt1[0]-pt2[0])
    
    # return point of intersection between boundary and LC trajectory
    if R == 0 and Rp1 == 1:
        # the equations are lines, so the intersection may be solved analytically
        x = (m*pt1[0]-a-pt1[1])/(m-1.)
        y = x-a
        return np.array([x,y])
    elif R == 1 and Rp1 == 2:
        y = (m*pt1[0]-a-pt1[1])/(m-1.)
        z = y-a
        return np.array([y,z])
    elif R == 2 and Rp1 == 0:
        z = (m*pt1[0]-a-pt1[1])/(m-1.)
        x = z - a
        return np.array([z,x])
    

def LCinit(a=0.01,rho=10.,init=[0.19, 0.18, 0.01],max_time=500,max_steps=5000000,return_period=False):
    """
    This function returns the times of flight and LC inital conditions for each region.
    :a: bifrucation parameter
    :rho: coupling constant
    :init: initial conditions of poncare map between regions 1 and 2
    :max_time: total time to integrate
    :max_steps: max steps to take for integration
    
    The return maps are 2D planes between regions.

    Between regions 1 and 2, y = x-a.
    Trajectories must enter region 2 at some point in z <= x-a

    Between regions 2 and 3, z = y.
    Trajectories must enter region 3 at some point in y >= x.

    Between regions 3 and 1, z = x.
    Trajectories must enter region 1 at some point in y <= x-a
    """



    # begin initial condition on edge between regions 1 and 2
    #assert (init[1] == init[0]-a and init[2] > 0 and init[2] < init[0])
    t = np.linspace(0, max_time, max_steps)
    vals = integrate.odeint(piecewise,
            init,
            t, args=(a,rho))

    # find crossing points between regions 1 and 2
    # (the a doesn't have to be here)
    crossings12 = ((vals[:-1,0] > vals[:-1,1]+a) * (vals[1:,0] <= vals[1:,1]+a)
                 * (vals[1:,2] < vals[1:,0]))#* (vals[1:,2] < vals[1:,0]-a))

    ## linearly interpolate point on boundary
    # points on first side of boundary
    x1 = vals[1:,0][crossings12][-1]
    y1 = vals[1:,1][crossings12][-1]

    # points on other side of boundary
    x2 = vals[:-1,0][crossings12][-1]
    y2 = vals[:-1,1][crossings12][-1]

    # find midpoint on x/y axis
    midpt = LinInterp(0,1,
                      [x2,y2],
                      [x1,y1],a=a)
    crossing12_fs = np.linalg.norm(midpt-np.array([x1,y1])) / np.linalg.norm(np.array([x2,y2])-np.array([x1,y1]))


    # x crossing point
    crossing12_xs = (crossing12_fs * vals[:-1,0][crossings12][-1]
            + (1-crossing12_fs) * vals[1:,0][crossings12][-1])
    # y crossing point
    crossing12_ys = (crossing12_fs * vals[:-1,1][crossings12][-1]
            + (1-crossing12_fs) * vals[1:,1][crossings12][-1])
    # z crossing point
    crossing12_zs = (crossing12_fs * vals[:-1,2][crossings12][-1]
            + (1-crossing12_fs) * vals[1:,2][crossings12][-1])

    # t crossing point
    t12exact = (crossing12_fs * t[:-1][crossings12][-1]
            + (1-crossing12_fs) * t[1:][crossings12][-1])

    # check for LC
    if crossings12.sum() < 2:
        raise RuntimeError("No complete cycles")

    # find crossing points between regions 2 and 3
    crossings23 = ((vals[:-1,1] > vals[:-1,2]+a) * (vals[1:,1] <= vals[1:,2]+a)
                 * (vals[1:,0] < vals[1:,1]))#* (vals[1:,0] < vals[1:,1]-a))

    ## Find point on boundary
    # points on first side of boundary
    y1 = vals[1:,1][crossings23][-1]
    z1 = vals[1:,2][crossings23][-1]

    # points on other side of boundary
    y2 = vals[:-1,1][crossings23][-1]
    z2 = vals[:-1,2][crossings23][-1]

    # find midpoint on y/z axis
    midpt = LinInterp(1,2,
                      [y2,z2],
                      [y1,z1],a=a)
    crossing23_fs = np.linalg.norm(midpt-np.array([y1,z1])) / np.linalg.norm(np.array([y2,z2])-np.array([y1,z1]))


    #print midpt,np.array([y1,z1])
    # x crossing point
    crossing23_xs = (crossing23_fs * vals[:-1,0][crossings23][-1]
            + (1-crossing23_fs) * vals[1:,0][crossings23][-1])
    # y crossing point
    crossing23_ys = (crossing23_fs * vals[:-1,1][crossings23][-1]
            + (1-crossing23_fs) * vals[1:,1][crossings23][-1])
    # z crossing point
    crossing23_zs = (crossing23_fs * vals[:-1,2][crossings23][-1]
            + (1-crossing23_fs) * vals[1:,2][crossings23][-1])

    # t crossing point
    t23exact = (crossing23_fs * t[:-1][crossings23][-1]
            + (1-crossing23_fs) * t[1:][crossings23][-1])


    if crossings23.sum() < 2:
        raise RuntimeError("No complete cycles")

    # find crossing points between regions 3 and 1
    crossings31 = ((vals[:-1,2] > vals[:-1,0]+a) * (vals[1:,2] <= vals[1:,0]+a)
                 * (vals[1:,1] < vals[1:,0]))#* (vals[1:,1] < vals[1:,0]-a))


    ## Find point on boundary
    # points on first side of boundary
    z1 = vals[1:,2][crossings31][-1]
    x1 = vals[1:,0][crossings31][-1]

    # points on other side of boundary
    z2 = vals[:-1,2][crossings31][-1]
    x2 = vals[:-1,0][crossings31][-1]

    # find midpoint on y/z axis
    midpt = LinInterp(2,0,
                      [z2,x2],
                      [z1,x1],a=a)
    crossing31_fs = np.linalg.norm(midpt-np.array([z1,x1])) / np.linalg.norm(np.array([z2,x2])-np.array([z1,x1]))


    # x crossing point
    crossing31_xs = (crossing31_fs * vals[:-1,0][crossings31][-1]
            + (1-crossing31_fs) * vals[1:,0][crossings31][-1])
    # y crossing point
    crossing31_ys = (crossing31_fs * vals[:-1,1][crossings31][-1]
            + (1-crossing31_fs) * vals[1:,1][crossings31][-1])
    # z crossing point
    crossing31_zs = (crossing31_fs * vals[:-1,2][crossings31][-1]
            + (1-crossing31_fs) * vals[1:,2][crossings31][-1])

    # t crossing point
    t31exact = (crossing31_fs * t[:-1][crossings31][-1]
            + (1-crossing31_fs) * t[1:][crossings31][-1])


    #print vals[1:,0][crossings23],vals[1:,1][crossings23]
    if crossings31.sum() < 2:
        raise RuntimeError("No complete cycles")
        
    """
    t12 = t[1:][crossings12]
    t23 = t[1:][crossings23]
    t31 = t[1:][crossings31]

    #print 't12exact, t12[-1]', t12exact, t12[-1]
    #print 't23exact, t23[-1]', t23exact, t23[-1]
    #print 't31exact, t31[-1]', t31exact, t31[-1]
    """    
    t12 = t[1:][crossings12]
    t23 = t[1:][crossings23]
    t31 = t[1:][crossings31]

    """
    print 't12:',t12[-1]
    print 't12exact:',t12exact
    print 'crossing x approx', vals[1:,0][crossings12][-1]
    print 'crossing x exact', crossing12_xs

    print 'crossing y approx', vals[1:,1][crossings12][-1]
    print 'crossing y exact', crossing12_ys
    
    print 'crossing z approx', vals[1:,2][crossings12][-1]
    print 'crossing z exact', crossing12_zs
    """

    #print t12,t23,t31
    
    # place all the time vectors in a list
    """
    The following steps are necessary because the last element of any of t12
    t23 and t31 could be the final time value.  So in order to find the time of flight
    for each region, I need to find the largest final time value, then subtract
    the other times from that.

    e.g. if the final time is t12[-1], subtract t31[-1] from that to
    get the time of flight for region 1.
    """

    #print 'final_times',final_times
    #print 'final_times original',[t12[-1],t23[-1],t31[-1]]
    final_times =[t12[-1],t23[-1],t31[-1]]#[t12exact,t23exact,t31exact] ##
    # pick index of max time, then subtract the remaining times from this number
    idx_max = int(np.argmax(final_times))


    if idx_max == 2:
        tof3 = final_times[2] - final_times[1]
        tof2 = final_times[1] - final_times[0]
        tof1 = final_times[0] - t31[-2]#(tof2+tof3)/2. #   #
    elif idx_max == 1:
        tof2 = final_times[1] - final_times[0]
        tof1 = final_times[0] - final_times[2]
        tof3 = final_times[2] - t23[-2] #(tof1+tof2)/2 
    elif idx_max == 0:
        tof1 = final_times[0] - final_times[2]
        tof3 = final_times[2] - final_times[1]
        tof2 = final_times[1] - t12[-2] #(tof3+tof1)/2.

    #print 'sum tofi',tof3+tof2+tof1
    #print 't12[-1]-t12[-2]',t12[-1]-t12[-2]



    ## period calculation
    # points on first side of boundary
    z1 = vals[1:,2][crossings31][-1]
    x1 = vals[1:,0][crossings31][-1]

    # points on other side of boundary
    z2 = vals[:-1,2][crossings31][-1]
    x2 = vals[:-1,0][crossings31][-1]

    # find midpoint on y/z axis
    midpt1 = LinInterp(2,0,
                      [z2,x2],
                      [z1,x1],a=a)
    crossing31_fs1 = np.linalg.norm(midpt-np.array([z1,x1])) / np.linalg.norm(np.array([z2,x2])-np.array([z1,x1]))


    # points on first side of boundary
    z1 = vals[1:,2][crossings31][-2]
    x1 = vals[1:,0][crossings31][-2]

    # points on other side of boundary
    z2 = vals[:-1,2][crossings31][-2]
    x2 = vals[:-1,0][crossings31][-2]

    # find midpoint on y/z axis
    midpt2 = LinInterp(2,0,
                      [z2,x2],
                      [z1,x1],a=a)
    crossing31_fs2 = np.linalg.norm(midpt-np.array([z1,x1])) / np.linalg.norm(np.array([z2,x2])-np.array([z1,x1]))

    t_second_to_last = (crossing31_fs2 * t[:-1][crossings31][-2]
            + (1-crossing31_fs2) * t[1:][crossings31][-2])

    t_last = (crossing31_fs1 * t[:-1][crossings31][-1]
            + (1-crossing31_fs1) * t[1:][crossings31][-1])
    
    T = t_last - t_second_to_last

    # return the three times of flight and the three LC initial conditions.
    if not(return_period):
        return ( tof1,tof2,tof3,
             (crossing12_xs,#vals[1:,0][crossings12][-1], # LC init for region 2
             crossing12_ys,#vals[1:,1][crossings12][-1],
             crossing12_zs),#vals[1:,2][crossings12][-1]),
             (crossing23_xs,#vals[1:,0][crossings23][-1], # LC init for region 3
             crossing23_ys,#vals[1:,1][crossings23][-1],
             crossing23_zs),#vals[1:,2][crossings23][-1]),
             (vals[1:,0][crossings31][-1], # LC init for region 1
             vals[1:,1][crossings31][-1],
             vals[1:,2][crossings31][-1]),
             )
    else:
        return ( tof1,tof2,tof3,
             (crossing12_xs,#vals[1:,0][crossings12][-1], # LC init for region 2
             crossing12_ys,#vals[1:,1][crossings12][-1],
             crossing12_zs),#vals[1:,2][crossings12][-1]),
             (crossing23_xs,#vals[1:,0][crossings23][-1], # LC init for region 3
             crossing23_ys,#vals[1:,1][crossings23][-1],
             crossing23_zs),#vals[1:,2][crossings23][-1]),
             (vals[1:,0][crossings31][-1], # LC init for region 1
             vals[1:,1][crossings31][-1],
             vals[1:,2][crossings31][-1]),
             T)
def pwl_biting_phase_reset(phi,init_all=[],tof_all=[], a=0.,rho=3.,
                           dx=0., dy=0.,dz=0., steps_per_cycle = 100000,
                           num_cycles = 10, return_intermediates=False,
                           T = 0.):

    #sys.stdout.write("\r"+str(int((phi/(2*math.pi))*100))+"% percent complete")
    #sys.stdout.flush()
    
    # get times of flight and initial conditions
    if len(init_all) == 0 or len(tof_all) == 0 or T == 0.:
        t1,t2,t3,init12,init23,init31,T = LCinit(a=a,rho=rho,return_period=True)
    else:
        t1,t2,t3 = tof_all
        init12,init23,init31 = init_all
    

    #print T, "total period in pwl_biting_phase_reset"
    steps_before = int(phi/(2*math.pi) * steps_per_cycle) + 1

    # run up to perturbation
    t1 = np.linspace(0, phi/(2*math.pi) * T, steps_before)
    vals1 = integrate.odeint(piecewise,
                   init31,
                   t1, args=(a,rho))

    # apply perturbation
    t2 = np.linspace(phi/(2*math.pi) * T, T * num_cycles,
                     steps_per_cycle * num_cycles - steps_before)
    vals2 = integrate.odeint(piecewise,
                   list(vals1[-1,:] + np.array([dx, dy, dz])),
                   t2, args=(a,rho))

    # return map is set to be between regions 3 and 1.
    crossings = ((vals2[:-1,2] > vals2[:-1,0]+a) * (vals2[1:,2] <= vals2[1:,0]+a)
                 * (vals2[1:,1] < vals2[1:,0]))

    if len(crossings) == 0:
        raise RuntimeError("No complete cycles after the perturbation")

    ## Find point on boundary
    # points on first side of boundary
    z1 = vals2[1:,2][crossings][-1]
    x1 = vals2[1:,0][crossings][-1]

    # points on other side of boundary
    z2 = vals2[:-1,2][crossings][-1]
    x2 = vals2[:-1,0][crossings][-1]

    # find midpoint on y/z axis
    midpt = LinInterp(1,2,
                      [z2,x2],
                      [z1,x1],a=a)

    crossing_fs = np.linalg.norm(midpt-np.array([z1,x1])) / np.linalg.norm(np.array([z2,x2])-np.array([z1,x1]))

    # find fs of last intersection
    #crossing_fs = 1.#( (vals2[:-1,2][crossings] - vals2[:-1,0][crossings])
                   #/ (vals2[1:,2][crossings]-vals2[:-1,2][crossings]) )
    #print crossing_fs
    crossing_times = (crossing_fs * t2[:-1][crossings]
                      + (1-crossing_fs) * t2[1:][crossings])
    crossing_phases = np.fmod(crossing_times, T)/T * 2 * math.pi
    crossing_phases[crossing_phases > math.pi] -= 2*math.pi

    #print crossing_phases    

    if return_intermediates:
        return dict(t1=t1, vals1=vals1, t2=t2, vals2=vals2,
                    crossings=crossings,
                    crossing_times=crossing_times,                    
                    crossing_phases=crossing_phases)

    else:
        return -crossing_phases[-1]



def matrixExponential(R,t,a=0.,rho=3.):
    """
    Return matrix exponential of region R
    :R: Region number
    :t: time (typically the time of flight)
    :a: bifurcation parameter value
    :rho: coupling constant
    """
    if R == 0:
        return np.array([[np.exp(t),0.,0.],
                         [.5* rho * np.exp(-t) * (np.exp(2*t)-1.) ,np.exp(-t),0.],
                        [0.,0.,np.exp(t*(rho-1.))]])
    elif R == 1:
        return np.array([[np.exp(t*(rho-1.)),0.,0.],
                         [0.,np.exp(t),0.],
                         [0.,.5*np.exp(-t) * rho * (np.exp(2*t)-1.),np.exp(-t)]])
    elif R == 2:
        return np.array([[np.exp(-t),0.,.5* rho * np.exp(-t) * (np.exp(2*t)-1.)],
                         [0.,np.exp(t*(rho-1.)),0.],
                         [0.,0.,np.exp(t)]])

def jumpMatrix(R,Rp1,init=[],a=0.01,rho=3.):
    assert ((Rp1-1)%3 == R)
    
    
    if R == 0 and Rp1 == 1:
        eta2 = init[0]; kappa2 = init[1]; nu2 = init[2]
        return [[(-eta2 + kappa2 - kappa2*rho + nu2*rho)/
                 (-1 + eta2 + kappa2 - eta2*rho + nu2*rho + 
                   a*(-1 + 2*rho)),
                 (-1 + a + 2*kappa2 + a*rho + nu2*rho)/
                 (-1 + eta2 + kappa2 - eta2*rho + nu2*rho + 
                   a*(-1 + 2*rho)), 
                 (a*(-2 + rho) - nu2*rho)/
                 (-1 + eta2 + kappa2 - \
                       eta2*rho + nu2*rho + a*(-1 + 2*rho))], 
                [(1 + a + eta2*(-2 + rho) - 
                  2*a*rho - kappa2*rho)/
                 (1 + a - kappa2 + eta2*(-1 + rho) - 
                  2*a*rho - nu2*rho), 
                 (eta2 - kappa2 + 
                  a*(-2 + rho) - eta2*rho)/
                 (-1 + eta2 + kappa2 - eta2*rho + nu2*rho + 
                   a*(-1 + 2*rho)), 
                 ((-a)*(-2 + rho) + nu2*rho)/
                 (-1 + eta2 + kappa2 - \
                       eta2*rho + nu2*rho + a*(-1 + 2*rho))],
                [0, 0, 1]]

    if R == 1 and Rp1 == 2:
        eta3 = init[0]; kappa3 = init[1]; nu3 = init[2]
        return [[1, 0, 0], [(a*(-2 + rho) - eta3*rho)/(-1 + kappa3 + \
nu3 + eta3*rho - kappa3*rho + 
     a*(-1 + 2*rho)), (-kappa3 + nu3 + eta3*rho - \
nu3*rho)/
       (-1 + kappa3 + nu3 + eta3*rho - kappa3*rho + 
     a*(-1 + 2*rho)), 
     (-1 + a + 2*nu3 + 
     a*rho + eta3*rho)/(-1 + kappa3 + nu3 + eta3*\
rho - kappa3*rho +       
     a*(-1 + 2*rho))], [((-a)*(-2 + rho) + eta3*rho)/(-1 \
+ kappa3 + nu3 + eta3*rho - 
          kappa3*rho + a*(-1 + 2*rho)), (1 + 
     a + kappa3*(-2 + rho) - 2*a*rho - nu3*rho)/
       (1 + a - nu3 + kappa3*(-1 + rho) - 
     2*a*rho - eta3*rho), 
     (kappa3 - nu3 + 
     a*(-2 + rho) - kappa3*rho)/(-1 + kappa3 + nu3 + \
eta3*rho - kappa3*rho + 
          a*(-1 + 2*rho))]]

    
    if R == 2 and Rp1 == 0:
        eta1 = init[0]; kappa1 = init[1]; nu1 = init[2]
        return [[(-eta1 + nu1 + 
     a*(-2 + rho) - nu1*rho)/(-1 + eta1 + nu1 + \
kappa1*rho - nu1*rho +           
     a*(-1 + 2*rho)), ((-a)*(-2 + rho) + kappa1*rho)/(-1 \
+ eta1 + nu1 + kappa1*rho - 
          nu1*rho + 
     a*(-1 + 2*rho)), (-1 - nu1*(-2 + rho) + eta1*rho \
+ a*(-1 + 2*rho))/
       (-1 + eta1 + nu1 + kappa1*rho - nu1*rho + 
     a*(-1 + 2*rho))], [0, 1, 0], 
   [(-1 + a + 2*eta1 + 
     a*rho + kappa1*rho)/(-1 + eta1 + nu1 + kappa1*\
rho - nu1*rho +           
     a*(-1 + 2*rho)), (a*(-2 + rho) - kappa1*rho)/(-1 + \
eta1 + nu1 + kappa1*rho - 
          nu1*rho + 
     a*(-1 + 2*rho)), (eta1 - nu1 - eta1*rho + \
kappa1*rho)/
       (-1 + eta1 + nu1 + kappa1*rho - nu1*rho + 
     a*(-1 + 2*rho))]]


def domains(ax,a):
    # DOMAIN EDGES
    #R1shift = np.array([a,-a,a]) # use this if keeping R1 stationary
    R1shift = np.array([0,-a,a])
    R1O = np.array([0,0,0])+R1shift # R2 origin (R-one-Oh)
    RV1 = np.array([[1,0,0], [1,0,1], [1,1,0], [1,1,1]]) + np.array([R1shift,R1shift,R1shift,R1shift]) 

    #R2shift = np.array([a,-a,-a]) # use this if keeping R3 stationary
    R2shift = np.array([a,0,-a])
    R2O = np.array([0,0,0])+R2shift # R3 origin
    RV2 = np.array([[0,1,0],[1,1,0],[0,1,1],[1,1,1]]) + np.array([R2shift,R2shift,R2shift,R2shift])

    #R3shift = np.array([0,0,0])
    R3shift = np.array([-a,a,0])
    R3O = np.array([0,0,0])+R3shift # R1 origin (R-one-Oh)
    RV3 = np.array([[0,0,1],[1,0,1],[0,1,1],[1,1,1]]) + np.array([R3shift,R3shift,R3shift,R3shift])

    # Plot domains
    # red = region 1, green = region 2, blue = region 3
    for i in range(4):
        ax.plot([R1O[0],RV1[i][0]],[R1O[1],RV1[i][1]],[R1O[2],RV1[i][2]],color='red')
        if i >= 1 and i <=2:
            ax.plot([RV1[i][0],RV1[0][0]],[RV1[i][1],RV1[0][1]],[RV1[i][2],RV1[0][2]],color='red')
            ax.plot([RV1[i][0],RV1[3][0]],[RV1[i][1],RV1[3][1]],[RV1[i][2],RV1[3][2]],color='red')

    for i in range(4):
        ax.plot([R2O[0],RV2[i][0]],[R2O[1],RV2[i][1]],[R2O[2],RV2[i][2]],color='green')
        if i >= 1 and i <=2:
            ax.plot([RV2[i][0],RV2[0][0]],[RV2[i][1],RV2[0][1]],[RV2[i][2],RV2[0][2]],color='green')
            ax.plot([RV2[i][0],RV2[3][0]],[RV2[i][1],RV2[3][1]],[RV2[i][2],RV2[3][2]],color='green')

    for i in range(4):
        ax.plot([R3O[0],RV3[i][0]],[R3O[1],RV3[i][1]],[R3O[2],RV3[i][2]],color='blue')
        if i >= 1 and i <=2:
            ax.plot([RV3[i][0],RV3[0][0]],[RV3[i][1],RV3[0][1]],[RV3[i][2],RV3[0][2]],color='blue')
            ax.plot([RV3[i][0],RV3[3][0]],[RV3[i][1],RV3[3][1]],[RV3[i][2],RV3[3][2]],color='blue')


    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def pwl_biting_phase_reset_analytic(a,tof_all=[],init_all=[],dt=0.001):
    # total number of regions
    Rtot = 3
    
    # if there is no input for ToF or LC init, then find them
    if len(tof_all) == 0 or len(init_all) == 0:
        t1,t2,t3,init12,init23,init31 = LCinit(a=a,rho=rho)
        tof_all = [t1,t2,t3];init_all=[init12,init23,init31]
        #print "Times of flight, t1, t2, t3:", t1,t2,t3
        #print "LC init for region 2", init12
        #print "LC init for region 3", init23
        #print "LC init for region 1", init31

    # unpack variables...
    t1,t2,t3 = tof_all
    init12,init23,init31 = init_all

    # ... and conslidate for later use
    LCinitlist = [init31,init12,init23]
    tlist = [t1,t2,t3]

    # determine total steps per region
    steps1 = int(t1*(1./dt))
    steps2 = int(t2*(1./dt))
    steps3 = int(t3*(1./dt))
    steplist = [steps1,steps2,steps3]
    totalsteps = sum(steplist)

    # create empty solution vector
    adjoint_solx = np.zeros(totalsteps)
    adjoint_soly = np.zeros(totalsteps)
    adjoint_solz = np.zeros(totalsteps)

    #t1,t2,t3,init12,init23,init31 = LCinit(a=a,rho=rho)
    for R in range(Rtot):
        if R == 0:
            t = np.linspace(0,tlist[R],steplist[R])

            # calculate M_i * exp_i
            expjumppair1 = np.dot(jumpMatrix(0,1,init=init12,a=a,rho=rho),matrixExponential(0,t1,a=a,rho=rho))
            expjumppair2 = np.dot(jumpMatrix(1,2,init=init23,a=a,rho=rho),matrixExponential(1,t2,a=a,rho=rho))
            expjumppair3 = np.dot(jumpMatrix(2,0,init=init31,a=a,rho=rho),matrixExponential(2,t3,a=a,rho=rho))
            
            # consolidate product
            temp1 = np.dot(expjumppair3,expjumppair2)
            B = np.dot(temp1,expjumppair1)
            print "a:",a
            print "B:", B

            # calculate eigenvalues and eigenvectors
            w,v = np.linalg.eig(B)
            print "eigenvalues:",w
            print "hatz10:",v[:,2]
            # use eigenvector with associated eigenvalue 1 and find unique z10
            hatz10 = np.array(v[:,2])
            T = t1+t2+t3
            z10 = hatz10 * (1./(T*np.dot(hatz10,vectorField(0,init=init31,a=a,rho=rho))))

            t = np.linspace(0,t1,steps1)
            matrixsol = np.dot(matrixExponential(0,t,a=a,rho=rho),z10)
            #print matrixsol
            adjoint_solx[:steps1] = matrixsol[0]
            adjoint_soly[:steps1] = matrixsol[1]
            adjoint_solz[:steps1] = matrixsol[2]

        else:
            M = jumpMatrix(R-1,R,init=LCinitlist[R],a=a,rho=rho)
            zinit = np.dot(M,np.array([adjoint_solx[sum(steplist[:R])-1],adjoint_soly[sum(steplist[:R])-1],np.array([adjoint_solz[sum(steplist[:R])-1]])]))

            t = np.linspace(0,tlist[R],steplist[R])
            matrixsol = np.dot(matrixExponential(R,t,a=a,rho=rho),zinit)
            adjoint_solx[sum(steplist[:R]):sum(steplist[:R+1])] = matrixsol[0]
            adjoint_soly[sum(steplist[:R]):sum(steplist[:R+1])] = matrixsol[1]
            adjoint_solz[sum(steplist[:R]):sum(steplist[:R+1])] = matrixsol[2]

    return adjoint_solx,adjoint_soly,adjoint_solz

def main():
    ## PARAMETERS
    init = [.1,.5,.8] # Initial condition
    finaltime = 100
    timesteps = 100*finaltime
    t = np.linspace(0,finaltime,timesteps)
    #sol = integrate.odeint(smooth,init,t)
    #sol = integrate.odeint(piecewise,init,t,args=(a,rho))
    #sol_local = integrate.odeint(piecewise_local,[.05,.05,.8],t)

    ## PLOTS

    fig = mp.figure()
    ax = fig.gca(projection='3d')
    domains(ax,a)

    #plt.show()
    """
    fig = mp.figure()
    ax = fig.gca(projection='3d')
    domains(ax,a)
    ax.plot(sol[:,0],sol[:,1],sol[:,2])
    ax.view_init(35,35)

    fig2 = mp.figure()
    ax2 = fig2.gca(projection='3d')
    domains(ax2,a)
    ax2.plot(sol[:,0],sol[:,1],sol[:,2])
    ax2.view_init(35,215)
    """

    #fig2 = mp.figure()
    #ax2 = fig2.gca(projection='3d')
    #ax.plot(sol_local[:,0],sol_local[:,1],sol_local[:,2])


    ## ADJOINT METHOD
    #mp.figure(2)
    #mp.plot(t,sol)

    t1,t2,t3,init12,init23,init31,T = LCinit(a=a,rho=rho,return_period=True)
    adjoint_solx,adjoint_soly,adjoint_solz = pwl_biting_phase_reset_analytic(a,
                                                   tof_all=[t1,t2,t3],
                                                   init_all=[init12,init23,init31])
    
    #print "Times of flight, t1, t2, t3:", t1,t2,t3
    print "LC init for region 2", init12
    print "LC init for region 3", init23
    print "LC init for region 1", init31
    """                                                                                                                                                              
    ## DIRECT METHOD
    Original code courtesy of Kendrick Shaw
    """

    total_perts = 10
    phis = np.linspace(0,2*math.pi,total_perts)

    pert = 1e-4
    
    print "Calculating iPRC via direct method for perturbations in the x direction..."
    x_prc = np.array([
            pwl_biting_phase_reset(phi, a=a, rho=rho,
                                 init_all=[init12,init23,init31],
                                 tof_all=[t1,t2,t3], dx=pert, dy=0.,dz=0.,T=T)
            for phi in phis
            ])
    print " "


    print "Calculating iPRC via direct method for perturbations in the y direction..."
    y_prc = np.array([
            pwl_biting_phase_reset(phi, a=a, rho=rho,
                                 init_all=[init12,init23,init31],
                                 tof_all=[t1,t2,t3],
                                 dx=0., dy=pert, dz=0.,T=T)
            for phi in phis
            ])
    print " "
    print "Calculating iPRC via direct method for perturbations in the z direction..."
    z_prc = np.array([
            pwl_biting_phase_reset(phi, a=a, rho=rho,
                                 init_all=[init12,init23,init31],
                                 tof_all=[t1,t2,t3],
                                 dx=0., dy=0., dz=pert,T=T)
            for phi in phis
            ])
    print " "


    #fig1 = plt.figure(figsize=(6,6))
    #axes = fig.add_axes([.1,.1,.8,.8])
    
    # normalize
    if pert != 0.0:
        x_prc = x_prc/pert
        y_prc = y_prc/pert
        z_prc = z_prc/pert

    phases_n = np.linspace(0,1,total_perts)

    mp.figure()
    mp.title("iPRC of PWL SHC Nominal Biting Model: Direct vs Adjoint")
    phases_a = np.linspace(0,1,len(adjoint_solx))

    p3, = mp.plot(phases_a,adjoint_solz,color='.75',ls='--')
    p6, = mp.plot(phases_n,z_prc,marker='D',color='.75',markeredgecolor='.75',ls='None')

    p2, = mp.plot(phases_a,adjoint_soly,color='.5',ls='-')
    p5, = mp.plot(phases_n,y_prc,marker='o',color='.5',markeredgecolor='.5',ls='None')

    p1, = mp.plot(phases_a,adjoint_solx,'k-')
    p4, = mp.plot(phases_n,x_prc,'bo')

    mp.legend([p1,p2,p3,p4,p5,p6],['Adjoint x','Adjoint y','Adjoint z','Direct x','Direct y','Direct z'])




    mp.show()


    
if __name__ == "__main__":
    main()

