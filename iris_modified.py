# Youngmin Park MS CWRU 2013
# yxp30@case.edu
# yop6@pitt.edu

"""
This script generates plots of the iPRC (direct and adjoint) of the modified iris
system with nonuniform saddle values.

Much of this code was originally written by Kendrick M. Shaw and modified by me.

Find these variables with the search function:
:pert =: perturbation size
:a =: bifurcation parameter
:l_cw_SW =: The first of the four saddle values

Jump to these sections with the search function:
:## ADJOINT CALCULATIONS: Analytic iPRC
:## DIRECT METHOD: Direct iPRC
:## PARAMETERS: Beginning of the parameters
:## PLOTS: Plots
:main(): The main function
"""

from scipy.integrate import odeint

import iris
import scipy.optimize as optimize
import numpy as np
import math
import matplotlib.pylab as mp
import sys

import generate_figures

def ToF(u):
    return np.log(1./u)

def LCinit(saddle_val,a=0.,guess=.5,return_all=False):
    # numerically derive LC initial condition for u_1
    l_cw_SW,l_ccw_SW,l_cw_NW,l_ccw_NW,l_cw_NE,l_ccw_NE,l_cw_SE,l_ccw_SE = saddle_val
    L1 = -l_cw_SW
    L2 = -l_cw_NW
    L3 = -l_cw_NE
    L4 = -l_cw_SE

    def return_map(x,a,L1,L2,L3,L4):
        sol = (((x**L1+a)**L2+a)**L3+a)**L4+a-x
        #print sol
        return sol

    def return_map_u2(x,a,L1,L2,L3,L4):
        sol = (((x**L2+a)**L3+a)**L4+a)**L1+a-x
        #print sol
        return sol

    def return_map_u3(x,a,L1,L2,L3,L4):
        sol = (((x**L3+a)**L4+a)**L1+a)**L2+a-x
        #print sol
        return sol

    def return_map_u4(x,a,L1,L2,L3,L4):
        sol = (((x**L4+a)**L1+a)**L2+a)**L3+a-x
        #print sol
        return sol
    
    # implement runtime error catch here
    u1 = optimize.newton(return_map,guess, args=(a,L1,L2,L3,L4))
    #u3 = optimize
    if not(return_all):
       # find limit cycle
        return u1
    else:
        u2 = optimize.newton(return_map_u2,guess, args=(a,L1,L2,L3,L4))#u1**(L1)+a
        u3 = optimize.newton(return_map_u3,guess, args=(a,L1,L2,L3,L4))#u2**(L2)+a
        u4 = optimize.newton(return_map_u4,guess, args=(a,L1,L2,L3,L4))#u3**(L3)+a
        return [u1,u2,u3,u4]

def iris_mod(y, unused_t, a=0., saddle_val=(), X=1., Y=1.):
    """
    :l_cw_k: stable saddle value
    :l_ccw_k: unstable saddle value
    :saddle_values: a tuple containing all saddle values
    """
    l_cw_SW,l_ccw_SW,l_cw_NW,l_ccw_NW,l_cw_NE,l_ccw_NE,l_cw_SE,l_ccw_SE = saddle_val
    if y[0] >= -a/2 and y[1] > a/2: # North East
        s = np.array([X - a/2, Y + a/2])
        l = np.array([l_cw_NE, l_ccw_NE])
    elif y[0] > a/2 and y[1] <= a/2: # South East
        s = np.array([Y + a/2, -X + a/2])
        l = np.array([l_ccw_SE, l_cw_SE])
    elif y[0] <= a/2 and y[1] < -a/2: # South West
        s = np.array([-X + a/2, -Y - a/2])
        l = np.array([l_cw_SW, l_ccw_SW])
    elif y[0] < -a/2 and y[1] >= -a/2: # North West
        s = np.array([-Y - a/2, X - a/2])
        l = np.array([l_ccw_NW, l_cw_NW])
    else:
        return 0*y

    return (y-s)*l;


def iris_mod_phase_reset(phi, saddle_val=(),u=[], a=0., dx=0., dy=0., steps_per_cycle = 500000,
                        num_cycles = 10, return_intermediates=False,X=1.,Y=1.):

    #sys.stdout.write("\r"+str(int((phi/(2*math.pi))*100))+"% percent complete")
    #sys.stdout.flush()
    u1 = u[0]
    u2 = u[1]
    u3 = u[2]
    u4 = u[3]

    # Total period
    T = ToF(u1)+ToF(u2)+ToF(u3)+ToF(u4)
    r0 = True # just roll with it for now
    if r0 == False:
        raise RuntimeError("No limit cycle found")
    else:
        steps_before = int(phi/(2*math.pi) * steps_per_cycle) + 1

        # run up to perturbation
        t1 = np.linspace(0, phi/(2*math.pi) * T, steps_before)
        vals1 = odeint(iris_mod,
                [a/2.,u1-Y-a/2.],
                t1, args=(a,saddle_val))

        # propagate perturbation
        t2 = np.linspace(phi/(2*math.pi) * T, T * num_cycles,
                steps_per_cycle * num_cycles - steps_before)
        vals2 = odeint(iris_mod,
                list(vals1[-1,:] + np.array([dx, dy])),
                t2, args=(a,saddle_val))

        crossings = ((vals2[:-1,0] > a/2) * (vals2[1:,0] <= a/2)
                * (vals2[1:,1] < 0))

        #mp.figure()
        #mp.plot(vals2[:,0],vals2[:,1])
        #mp.show()
        if len(crossings) == 0:
            raise RuntimeError("No complete cycles after the perturbation")


        crossing_fs = ( (vals2[1:,0][crossings] - a/2)
                / (vals2[1:,0][crossings]-vals2[:-1,0][crossings]) )
        crossing_times = (crossing_fs * t2[:-1][crossings]
                + (1-crossing_fs) * t2[1:][crossings])
        crossing_phases = np.fmod(crossing_times, T)/T * 2 * math.pi
        crossing_phases[crossing_phases > math.pi] -= 2*math.pi                                                                                                       

        if return_intermediates:
            return dict(t1=t1, vals1=vals1, t2=t2, vals2=vals2,
                    crossings=crossings,
                    crossing_times=crossing_times,
                    crossing_phases=crossing_phases)
        else:
            return -crossing_phases[-1]

def iris_mod_phase_reset_analytic(saddle_val=[],a=0.,tof_all=[],u_all=[],dt=.001):
    """
    :saddle_val: list of saddle values
    :a: bifurcation parameter value
    :tof_all: all times of flight
    :u_all: all nontrivial initial conditions
    :dt: mesh width
    """

    # unpack variables
    l_cw_SW,l_ccw_SW,l_cw_NW,l_ccw_NW,l_cw_NE,l_ccw_NE,l_cw_SE,l_ccw_SE = saddle_val
    t1,t2,t3,t4 = tof_all
    u1,u2,u3,u4 = u_all

    # derive times of flight and initial conditions if necessary
    if len(tof_all) == 0 or len(u_all) == 0:
        u1,u2,u3,u4 = LCinit(saddle_val,a=a,return_all=True)        
        t1,t2,t3,t4 = (ToF(u1),ToF(u2),ToF(u3),ToF(u4))

    #print (u1-1-a/2),(u2-1-a/2),(1+a/2-u3),(1+a/2-u4)
    # variables for convenience
    L1 = -l_cw_SW
    L2 = -l_cw_NW
    L3 = -l_cw_NE
    L4 = -l_cw_SE

    s1 = u1**(L1)
    s2 = u2**(L2)
    s3 = u3**(L3)
    s4 = u4**(L4)

    # period
    T = t1+t2+t3+t4
    
    # The amount of time an adjoint solution spends in one time may be non-uniform
    # Define solution vector to compensate for this potential discrepancy
    # this method should work as long as the total time is large.

    # total number of solution values per region
    steps1 = int(t1*(1./dt))
    steps2 = int(t2*(1./dt))
    steps3 = int(t3*(1./dt))
    steps4 = int(t4*(1./dt))
    totalsteps = steps1+steps2+steps3+steps4
    
    # create empty solution vectors
    adjoint_solx = np.zeros(totalsteps)
    adjoint_soly = np.zeros(totalsteps)
    
    # define parameters used for iPRC initial condition calculation
    upsilon = u2*u3*u4
    xi = s1*s2*s3*s4*L1*L2*L3*L4
    zeta = s1*L1*(u3*u4 + s2*L2*(u4 + s3*L3))

    # This next section could be put into a loop with some patience, but there are too many moving parts.
    # REGION 1: R = 0
    # iPRC initial condition
    R = 0
    t = np.linspace(0,t1,steps1)

    hatz10 = np.array([(s1*(u1*(u3*u4 + s2*L2*(u4 + s3*L3)) + 
                            s2*s3*s4*L2*L3*L4))/
                       (u2*u3*u4 + 
                        s1*L1*(u3*u4 + s2*L2*(u4 + s3*L3))), 1.])

    minu = np.amin([u1,u2,u3,u4])
    toflist = np.array([t1,t2,t3,t4])
    #weightlist = toflist/sum(list(toflist))
    #weighted_average = sum(list(weightlist*toflist))/sum(list(weightlist))
    #tof_average = sum(list(toflist))/4.
    scale = sum(list(toflist))*np.dot(hatz10,np.array([-L1,u1])) #ToF(u1) *(-L1*((u1*zeta + xi)/(L1*(upsilon + zeta))) + u1)
    z10 = (1./scale)*hatz10 #np.array([(u1*zeta + xi)/(L1*(upsilon + zeta)),1])

    adjoint_solx[:steps1] = np.exp(L1*t)*z10[0]
    adjoint_soly[:steps1] = np.exp(-t)*z10[1]

    # REGION 2: R = 1
    R = 1
    t = np.linspace(0,t2,steps2)

    # calculate initial condition of next iPRC
    # in terms of z10
    M = (1./L2)*np.array([[L2,0.],[-(L1*s1+u2),1.]])
    z20 = np.dot(M,np.array([adjoint_solx[steps1-1],adjoint_soly[steps1-1]]))

    # explicit, not based in terms of z10
    #hatz20 = np.array([-((u1*(u3*u4 + s2*L2*(u4 + s3*L3)) + s2*s3*s4*L2*L3*L4)/(s2*(u1*u2*(u4 + s3*L3) +  s3*s4*(u2 + s1*L1)*L3*L4))), 1])
    #scale = ToF(minu)*np.dot(hatz20,np.array([u2,L2]))

    #(ToF(u2) *    (u2*(-(u1*(u3*u4+s2*L2*(u4 + s3*L3)) + s2*s3*s4*L2*L3*L4)/(s2*(u1*u2*(u4 + s3*L3) + s3*s4*(u2 + s1*L1)*L3*L4))) + L2))
    #z20 = (1./scale)*hatz20 #*np.array([-(u1*(u3*u4+s2*L2*(u4 + s3*L3)) + s2*s3*s4*L2*L3*L4)/(s2*(u1*u2*(u4 + s3*L3) + s3*s4*(u2 + s1*L1)*L3*L4)),1])

    # save solution to array
    adjoint_solx[steps1:steps1+steps2] = np.exp(-t)*z20[0]
    adjoint_soly[steps1:steps1+steps2] = np.exp(L2*t)*z20[1]


    # REGION 3: R = 2
    R = 2
    t = np.linspace(0,t3,steps3)
    
    # calculate initial condition of next iPRC
    # not in terms of z10

    #hatz30 = np.array([(s3*(u1*u2*u3 + s4*(u2*u3 + s1*L1*(u3 + s2*L2))*L4))/(u1*u2*(u4 + s3*L3) + s3*s4*(u2 + s1*L1)*L3*L4), 1])
    #scale = ToF(minu)*np.dot(hatz30,np.array([L3,-u3]))
    #scale = (ToF(minu) *       (L3*(s3*(u1*u2*u3 + s4*(u2*u3 + s1*L1*(u3 + s2*L2))*L4)/(u1*u2*(u4 + s3*L3) + s3*s4*(u2 + s1*L1)*L3*L4)) - u3))
    #z30 = (1./scale)* np.array([s3*(u1*u2*u3 + s4*(u2*u3 + s1*L1*(u3 + s2*L2))*L4)/(u1*u2*(u4 + s3*L3) + s3*s4*(u2 + s1*L1)*L3*L4),1])
    #z30 = (1./scale)*hatz30
    # calculate initial condition of next iPRC

    # in terms of z10
    M = (1./L3)*np.array([[1.,L2*s2+u3],[0,L3]])
    z30 = np.dot(M,np.array([adjoint_solx[steps1+steps2-1],adjoint_soly[steps1+steps2-1]]))

    # save solution to array
    adjoint_solx[steps1+steps2:steps1+steps2+steps3] = np.exp(L3*t)*z30[0]
    adjoint_soly[steps1+steps2:steps1+steps2+steps3] = np.exp(-t)*z30[1]

    # REGION 4: R = 3
    R = 3
    t = np.linspace(0,t4,steps4)

    # calculate initial condition of next iPRC
    # not in terms of z10
    #hatz40 = np.array([-((u1*u2*u3 + s4*(u2*u3 + s1*L1*(u3 + s2*L2))*L4)/(s4*(u2*u3*u4 + s1*L1*(u3*u4 + s2*L2*(u4 + s3*L3))))),1.])

    #scale = ToF(minu)* np.dot(hatz40,np.array([-u4,-L4]))
    #(-u4*(   -(u1*u2*u3 + s4*(u2*u3 + s1*L4*(u3 + s2*L2))*L4)/(s4*(u2*u3*u4 + s1*L1*(u3*u4 + s2*L2*(u4 + s3*L3))))) - L4))
    #z40 = (1./scale)*hatz40
    #np.array([ -(u1*u2*u3 + s4*(u2*u3 + s1*L4*(u3 + s2*L2))*L4)/(s4*(u2*u3*u4 + s1*L1*(u3*u4 + s2*L2*(u4 + s3*L3)))),1])

    # calculate initial condition of next iPRC
    # in terms of z10
    M = (1./L4)*np.array([[L4,0],[-(L3*s3+u4),1.]])
    z40 = np.dot(M,np.array([adjoint_solx[steps1+steps2+steps3-1],adjoint_soly[steps1+steps2+steps3-1.]]))


    # save solution to array
    adjoint_solx[steps1+steps2+steps3:steps1+steps2+steps3+steps4] = np.exp(-t)*z40[0]
    adjoint_soly[steps1+steps2+steps3:steps1+steps2+steps3+steps4] = np.exp(L4*t)*z40[1]

    # Calculate the final jump matrix for fun
    M = (1./(L1))*np.array([[1.,L4*s4+u1],[0,L1]])
    #print 'zf4:', np.array([adjoint_solx[-1],adjoint_soly[-1]])
    #print 'z10:', np.dot(M,np.array([adjoint_solx[-1],adjoint_soly[-1]]))

    #t = np.linspace(0,t1,steps1)
    #z10 = np.dot(M,np.array([adjoint_solx[-1],adjoint_soly[-1]]))
    #adjoint_solx[:steps1] = np.exp(L1*t)*z10[0]
    #adjoint_soly[:steps1] = np.exp(-t)*z10[1]
    
    
    return adjoint_solx,adjoint_soly


def main():
    ## PARAMETERS
    X = 1.
    Y = 1.
    a = 1e-2
    #a = 1e-3
    
    #l_cw_SW = -2.5
    #l_cw_NW = -1.2
    #l_cw_NE = -3.
    #l_cw_SE = -4.

    l_cw_SW = -2.5
    l_cw_NW = -1.5
    l_cw_NE = -3.
    l_cw_SE = -4.
    
    l_ccw_SW = 1.
    l_ccw_NW = 1.
    l_ccw_NE = 1.
    l_ccw_SE = 1.
    saddle_val = (l_cw_SW,l_ccw_SW,l_cw_NW,l_ccw_NW,l_cw_NE,l_ccw_NE,l_cw_SE,l_ccw_SE)

    u1,u2,u3,u4 = LCinit(saddle_val,a,return_all=True) # get all LC initial conditions (local coordinate u)
    #print LCinit(saddle_val,a)


    #print (u1-1-a/2),(u2-1-a/2),(1+a/2-u3),(1+a/2-u4)
    # variables for convenience
    L1 = -l_cw_SW
    L2 = -l_cw_NW
    L3 = -l_cw_NE
    L4 = -l_cw_SE

    s1 = u1**(L1)
    s2 = u2**(L2)
    s3 = u3**(L3)
    s4 = u4**(L4)

    # Phase portrait
    generate_figures.iris_mod_fig(a)

    """
    tmax = 100
    maxsteps = tmax*1000
    t = np.linspace(0,tmax,maxsteps)
    #vals = odeint(iris_mod,[X+a/2.-LCinit(saddle_val,a),a/2.],t,args=(a,saddle_val))
    vals = odeint(iris_mod,[a/2,u1-Y-a/2],t,args=(a,saddle_val))
    mp.figure()
    mp.title("Phase Portrait of Modified Iris System")
    mp.plot(vals[:,0],vals[:,1])
    mp.xlabel('x')
    mp.ylabel('y')
    """
    
    ## ADJOINT CALCULATIONS
    """
    see  glass-pasternack-2d-cyclic-attractor.py for algorithm
    THIS SECTION IS FOR THE ADJOINT METHOD.  SEE THE NEXT SECTION FOR THE DIRECT METHOD.
    """
    t1,t2,t3,t4 = [ToF(u1),ToF(u2),ToF(u3),ToF(u4)]
    adjoint_solx,adjoint_soly = iris_mod_phase_reset_analytic(
        saddle_val=saddle_val,a=a,
        tof_all=[t1,t2,t3,t4],u_all=[u1,u2,u3,u4],
        dt=.001)

    """
    ## DIRECT METHOD
    Original code courtesy of Kendrick Shaw
    """
    total_perts = 100
    phis = np.linspace(0,2*math.pi,total_perts)

    pert = 1e-4#1e-2
    print "Calculating iPRC via direct method for perturbations in the x direction..."
    x_prc = np.array([
            iris_mod_phase_reset(phi, a=a, saddle_val=saddle_val, u=[u1,u2,u3,u4], dx=pert, dy=0.)
            for phi in phis
            ])
    print " "
    print "Calculating iPRC via direct method for perturbations in the y direction..."
    y_prc = np.array([
            iris_mod_phase_reset(phi, a=a, saddle_val=saddle_val, u=[u1,u2,u3,u4],dx=0., dy=pert)
            for phi in phis
            ])
    print " "

    if pert != 0.:
        x_prc = x_prc/pert
        y_prc = y_prc/pert


    ## PLOTS
    mp.figure()
    mp.title("iPRC of Modified Iris System: Direct vs Adjoint")
    phases_a = np.linspace(0,1,len(adjoint_solx))
    p1, = mp.plot(phases_a,adjoint_solx*(math.pi/2.)) # analytic
    p2, = mp.plot(phases_a,adjoint_soly*(math.pi/2.)) 
    p3, = mp.plot(np.linspace(0,1,total_perts),x_prc,'bo',markeredgecolor='b') # direct
    p4, = mp.plot(np.linspace(0,1,total_perts),y_prc,'go',markeredgecolor='g')
    mp.legend([p1,p2,p3,p4],['Adjoint x','Adjoint y','Direct x','Direct y'])


    """
    xtest_prc = np.array([
            iris.phase_reset(phi, a=a, dx=pert, dy=0.,l_cw=-2,l_ccw=1)
            for phi in phis
            ])
    xtest2_prc = np.array([
            iris.analytic_phase_reset(phi, a=a, dx=pert, dy=0.,l_cw=-2,l_ccw=1)
            for phi in phis
            ])
    mp.figure()
    
    total_pert_vector = np.linspace(0,1,total_perts)
    total_pert_vector2 = np.linspace(0,1,total_perts)
    mp.plot(total_pert_vector,xtest_prc/pert,'bo')
    mp.plot(total_pert_vector,xtest2_prc/pert)
    mp.plot(phases_a,adjoint_solx*(math.pi/2.)) # this scaling is the cause of the discrepancy.  Multiply my solutions by math.pi/2.
    mp.plot(total_pert_vector2/4.,(u1**(L1*(1-total_pert_vector2)))/(ToF(u1)*(u1-L1*u1**L1)))
    
    """
    mp.show()
    

if __name__ == '__main__':
    main()
