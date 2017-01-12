# Youngmin Park MS CWRU 2013
# yxp30@case.edu
# yop6@pitt.edu

"""
This script generates plots of Example 1 from Glass and Pasternack 1978 (also see Fig. 2).
The example is a model of feedback inhibition in 2 dimensions.  In verbal terms,
"x_1 stimulates the production of x_2 and x_2 inhibits the production of x_1."


Find these variables with the search function:
:pert =: perturbation size


Jump to these sections with the search function:
:## ADJOINT CALCULATIONS: Analytic iPRC
:## DIRECT METHOD: Direct iPRC
:## PARAMETERS: Beginning of the parameters
:## PHASE SPACE: Gather data for phase plane plots
:## PLOTS: Plots
:main(): The main function
"""

from scipy.integrate import odeint
import numpy as np
import math
import matplotlib.pylab as mp
import sys
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', serif=['Computer Modern Roman'])
from matplotlib import pyplot as plt


def checkrho(fixedpts):
    """
    If we let h_1(x_1) = \frac{\rho}{1+r x_1} where
    \rho = \frac{a_4 b_3 a_2 b_1}{b_4 a_3 b_2 a_1} and
    r = \frac{1}{a_1} + \frac{b_1}{a_1 b_2} + \frac{b_1 a_2}{a_1 b_2 a_3} + \frac{b_1 a_2 b_3}{a_1 b_2 a_3 b_4}
    then h is a Poincare map along the x axis.  Limit cycles exist when \rho > 1.

    I think r above, as printed in the paper, might be incorrect.  It should be
    r = -\frac{1}{a_1} + \frac{b_1}{a_1 b_2} - \frac{b_1 a_2}{a_1 b_2 a_3} + \frac{b_1 a_2 b_3}{a_1 b_2 a_3 b_4}
    """
    a1,b1,a2,b2,a3,b3,a4,b4 = fixedpts

    # \rho and r as described above
    # We will use these parameters to check for limit cycles
    rho = (a4*b3*a2*b1)/(b4*a3*b2*a1)

    #r = 1./a1 + b1/(a1*b2) + (b1*a2)/(a1*b2*a3) + (b1*a2*b3)/(a1*b2*a3*b4)
    # rho: Scalar input as described in main()
    if rho > 1:
        return True
    else:
        return True

def boolvar(x):
    """
    x: any scalar we wish to turn into a boolean.
    Rules determined by Lewis and Glass 1991, Eq. (6)
    """
    if x < 0:
        return 0
    else:
        return 1

def D2Attractor(y,t,fixedpt1,fixedpt2,fixedpt3,fixedpt4):
    """
    y: initial condition to ODE
    t: time
    fixedpti: tuple containing the ith fixed point; (ai,bi) = fixedpti
    
    Equations:
    \frac{dy_i}{dt} = \Lambda_i(\tilde{y}_1,\tilde{y}_2,\ldots,\tilde{y}_N)-y_i
    """

    # lambdai is only used for readability.
    if boolvar(y[0]) == 1 and boolvar(y[1]) == 1: # first quadrant
        a1,b1 = fixedpt1
        lambda1 = a1
        lambda2 = b1
        return [lambda1-y[0], lambda2-y[1]]
    elif boolvar(y[0]) == 0 and boolvar(y[1]) == 1: # second quadrant
        a2,b2 = fixedpt2
        lambda1 = a2
        lambda2 = b2
        return [lambda1-y[0], lambda2-y[1]]
    elif boolvar(y[0]) == 0 and boolvar(y[1]) == 0: # third quadrant
        a3,b3 = fixedpt3
        lambda1 = a3
        lambda2 = b3
        return [lambda1-y[0], lambda2-y[1]]
    elif boolvar(y[0]) == 1 and boolvar(y[1]) == 0: # fourth quadrant
        a4,b4 = fixedpt4
        lambda1 = a4
        lambda2 = b4
        return [lambda1-y[0], lambda2-y[1]]

def LCinit(R,fixedpts):
    """
    This function finds the initial value of the limit cycle of a given region R.
    Note that it only finds either the ordinate or abscissa, but not both.
    For example, in region 0, when viewed as a point in space, the function looks like

    \partial_{4/1}^\gamma = (LCinit(0,fixedpts),0)

    At the next limit cycle crossing along the positive y-axis (x_2), LCinit is the abscissa.

    R: region number, ranging from 0 to Rtot-1
    fixedpts = (a1,b1,a2,b2,a3,b3,a4,b4)
    """
    # unpack fixed points
    a1,b1,a2,b2,a3,b3,a4,b4 = fixedpts
    if R == 0: # equivalent to region 1
        # return the initial value of the LC of region 1
        return (a2*a4*b1*b3 - a1*a3*b2*b4)/(a2*b1*b3 - a2*b1*b4 + a3*b1*b4 - a3*b2*b4)
    elif R == 1: # equivalent to region 2
        # return the initial value of the LC of region 2
        return (a2*a4*b1*b3 - a1*a3*b2*b4)/(a2*a4*b3 - a1*(a2*(b3 - b4) + a3*b4))
    elif R == 2:
        return (a2*a4*b1*b3 - a1*a3*b2*b4)/(a4*(b1 - b2)*b3 + a1*b2*(b3 - b4))
    elif R == 3:
        return (a2*a4*b1*b3 - a1*a3*b2*b4)/(a2*a4*b1 - a3*a4*b1 - a1*a3*b2 + a3*a4*b2)


def ToF(R,fixedpts):
    """
    This function finds the time of flight for each region.
    R: region number, ranging from 0 to Rtot-1
    fixedpts = (a1,b1,a2,b2,a3,b3,a4,b4)
    """
    # unpack fixed points
    a1,b1,a2,b2,a3,b3,a4,b4 = fixedpts
    if R == 0:
        return np.log((a1-LCinit(R,fixedpts))/a1)
    if R == 1:
        return np.log((b2-LCinit(R,fixedpts))/b2)
    if R == 2:
        return np.log((a3-LCinit(R,fixedpts))/a3)
    if R == 3:
        return np.log((b4-LCinit(R,fixedpts))/b4)

def jumpMatrix(R,Rp1,fixedpts):
    """
   This functon returns the jump matrix for boundaries between regions R and R+1
    R: region number
    Rp1: adjacent region number
    """
    assert ((Rp1 - 1)%4 == R) # only look at adjacent regions

    # unpack fixed points
    a1,b1,a2,b2,a3,b3,a4,b4 = fixedpts 

    # return jump matrix
    if R == 0 and Rp1 == 1:
        return np.array([[a1/a2,(b1-b2)/a2],[0.,1.]])
    if R == 1 and Rp1 == 2:
        return np.array([[1.,0.],[(a2-a3)/b3,b2/b3]])
    if R == 2 and Rp1 == 3:
        return np.array([[a3/a4,(b3-b4)/a4],[0.,1.]])
    if R == 3 and Rp1 == 0:
        return np.array([[1.,0.],[(a4-a1)/b1,b4/b1]])


def glass2d_phase_reset(phi, fixedpts, dx=0., dy=0., steps_per_cycle = 100000,
                        num_cycles = 10, return_intermediates=False):
    
    r0 = checkrho(fixedpts) #check for limit cycle
    
    # console output indicating completion (feel free to comment or delete)
    #sys.stdout.write("\r"+str(int((phi/(2*math.pi))*100))+"% percent complete")
    #sys.stdout.flush()

    # unpack fixed points
    a1,b1,a2,b2,a3,b3,a4,b4 = fixedpts

    # total period
    T = ToF(0,fixedpts)+ToF(1,fixedpts)+ToF(2,fixedpts)+ToF(3,fixedpts)

    if r0 == False:
        raise RuntimeError("No limit cycle")
    else:
        steps_before = int(phi/(2*math.pi) * steps_per_cycle) + 1

        t1 = np.linspace(0, phi/(2*math.pi) * T, steps_before)
        vals1 = odeint(D2Attractor,
                [LCinit(0,fixedpts),0],
                t1, args=((a1,b1),(a2,b2),(a3,b3),(a4,b4)))

        t2 = np.linspace(phi/(2*math.pi) * T, T * num_cycles,
                steps_per_cycle * num_cycles - steps_before)
        vals2 = odeint(D2Attractor,
                list(vals1[-1,:] + np.array([dx, dy])),
                t2, args=((a1,b1),(a2,b2),(a3,b3),(a4,b4)))
        """
        crossings = ((vals2[:-1,0] > 0) * (vals2[1:,0] <= 0) 
                * (vals2[1:,1] > 0))

        if len(crossings) == 0:
            raise RuntimeError("No complete cycles after the perturbation")
        crossing_fs = ( (vals2[1:,0][crossings] - 0)
                / (vals2[1:,0][crossings]-vals2[:-1,0][crossings]) )
        """
        crossings = ((vals2[:-1,1] <= 0) * (vals2[1:,1] > 0) 
                * (vals2[1:,0] > 0))
        if len(crossings) == 0:
            raise RuntimeError("No complete cycles after the perturbation")
        crossing_fs = ( (vals2[1:,1][crossings] - 0)
                / (vals2[1:,1][crossings]-vals2[:-1,1][crossings]) )
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


def glass_2d_phase_reset_analytic(fixedpts=[],dt=0.001):
    assert (len(fixedpts) != 0)
    """
    The algorithm is as follows:
    1. Find initial condition of LC
    2. Find time of flight for current region
    3. calculation adjoint in that region using known initial condition
    4. Apply jumpy matrix to final point
    5. repeat steps 1-4

    The solution to the adjoint equation is the same for all regions,
    \begin{matrix} e^{t_k} & 0 \\ 0 & e^{t_k} \end{matrix}, \forall k=1,2,3,4.
    """

    # The amount of time an adjoint solution spends in one time may be non-uniform
    # Define solution vector to compensate for this potential discrepancy
    # this method should work as long as the total time is large
    Rtot = 4 # total number of regions

    a1,b1,a2,b2,a3,b3,a4,b4 = fixedpts
    t1,t2,t3,t4 = (ToF(0,fixedpts),ToF(1,fixedpts),ToF(2,fixedpts),ToF(3,fixedpts))

    steps1 = int(t1*(1./dt))
    steps2 = int(t2*(1./dt))
    steps3 = int(t3*(1./dt))
    steps4 = int(t4*(1./dt))
    steplist = [steps1,steps2,steps3,steps4]

    totalsteps = sum(steplist)

    adjoint_solx = np.zeros(totalsteps) # create empty solution vector
    adjoint_soly = np.zeros(totalsteps) # create empty solution vector
    for R in range(Rtot):
        if R == 0:
            # iPRC initial condition
            t = np.linspace(0,ToF(R,fixedpts),steplist[R])
            #tof_average = sum([t1,t2,t3,t4])/4
            #mintof = np.amin([ToF(0,fixedpts),ToF(1,fixedpts),ToF(2,fixedpts),ToF(3,fixedpts)])
            #maxtof = np.amax([ToF(0,fixedpts),ToF(1,fixedpts),ToF(2,fixedpts),ToF(3,fixedpts)])
            toflist = np.array([t1,t2,t3,t4])
            T = sum(list(toflist))
            #weights = toflist/T
            #weighted_average = sum(list(toflist*weights))/sum(list(weights))
            z10 = (1./(T*((-b1/a1)*(a1 - LCinit(R,fixedpts)) + b1)))*np.array([-b1/a1, 1.]) 
            #print z10
            adjoint_solx[:steps1] = np.exp(t)*z10[0]
            adjoint_soly[:steps1] = np.exp(t)*z10[1]


        else:
            M = jumpMatrix(R-1,R,fixedpts) # jump matrix

            # calculate initial condition of next iPRC given jumpMatrix
            zinit = np.dot(M,np.array([adjoint_solx[sum(steplist[:R])-1],adjoint_soly[sum(steplist[:R])-1]])) 

            # print adjoint_solx[maxpts*R-1],R
            t = np.linspace(0,ToF(R,fixedpts),steplist[R])
            # save solution to array
            adjoint_solx[sum(steplist[:R]):sum(steplist[:R+1])] = np.exp(t)*zinit[0]
            adjoint_soly[sum(steplist[:R]):sum(steplist[:R+1])] = np.exp(t)*zinit[1]

    return adjoint_solx,adjoint_soly


def main():
    ## PARAMETERS

    # Fixed points
    a1,b1 = [-5.,11.]
    a2,b2 = [-10.,-4.]
    a3,b3 = [6.,-10.]
    a4,b4 = [10.,5.]
    fixedpts = (a1,b1,a2,b2,a3,b3,a4,b4) # pack fixed point parameters

    print "Does a limit cycle exist?", checkrho(fixedpts) # check for limit cycles

    ## PHASE SPACE

    # initial conditions for vals1
    y0 = [.5,0] # initial condition
    tmax = 20
    maxsteps = tmax*10000
    t = np.linspace(0,ToF(0,fixedpts)+ToF(1,fixedpts)+ToF(2,fixedpts)+ToF(3,fixedpts),maxsteps)

    # initial conditions for vals2
    y1 = [10,0]

    # integrate
    vals1 = odeint(D2Attractor, y0, 1.5*t, args=((a1,b1),(a2,b2),(a3,b3),(a4,b4)))
    vals2 = odeint(D2Attractor, y1, 1.5*t, args=((a1,b1),(a2,b2),(a3,b3),(a4,b4)))    
    lcval = odeint(D2Attractor, [LCinit(0,fixedpts),0], t, args=((a1,b1),(a2,b2),(a3,b3),(a4,b4)))

    ## ADJOINT CALCULATIONS
    adjoint_solx,adjoint_soly = glass_2d_phase_reset_analytic(fixedpts)

    ## DIRECT METHOD
    """
    Original code courtesy of Kendrick M. Shaw CWRU
    """

    # total perturbations
    total_perts = 30

    # phase values where perturbations will be applied
    phis = np.linspace(0,2*math.pi,total_perts)
    #dx = 1e-4
    #dy = 0

    pert = 1#1e-2 # keep perturbation to single variable for now
    print "Calculating iPRC via direct method for perturbations in the x direction..."
    x_prc = np.array([
            glass2d_phase_reset(phi, fixedpts, dx=pert, dy=0)
            for phi in phis
            ])
    print " "
    print "Calculating iPRC via direct method for perturbations in the y direction..."
    y_prc = np.array([
            glass2d_phase_reset(phi, fixedpts, dx=0, dy=pert)
            for phi in phis
            ])
    print " "


    ## PLOTS
    """
    
    """



    ## Phase portrait
    fig = plt.figure(figsize=(6,6))
    axes = fig.add_axes([.1,.1,.8,.8])

    # draw axes
    axes.plot([-15,15],[0,0],color='0.5') # x-axis
    axes.plot([0,0],[-15,15],color='0.5') # y-axis
    axes.text(11,.5,'x')
    axes.text(-1,11,'y')

    # solution curve
    plt.title("Phase Portrait of 2D Glass Network (Feedback Inhibition)")
    axes.plot(vals1[:,0],vals1[:,1],lw=1,color='b')
    axes.plot(vals2[:,0],vals2[:,1],lw=1,color='r')
    axes.plot(lcval[:,0],lcval[:,1],lw=3,color='purple')
    #mp.plot(vals2[:,0],vals2[:,1],color='red')
    axes.set_xlim(-12,12)
    axes.set_ylim(-12,12)
    axes.set_xticks([])
    axes.set_yticks([])

    # fixed points    
    axes.plot([0,a1],[LCinit(1,fixedpts),b1],'k--')
    axes.plot([LCinit(2,fixedpts),a2],[0,b2],'k--')
    axes.plot([0,a3],[LCinit(3,fixedpts),b3],'k--')
    axes.plot([LCinit(0,fixedpts),a4],[0,b4],'k--')
    
    # fixed point labels
    fixedptlist = [[a1,b1],[a2,b2],[a3,b3],[a4,b4]]
    valign = ['bottom','top','bottom','bottom']
    halign = ['right','left','left','right']
    
    for i in range(len(fixedptlist)):
            axes.plot(fixedptlist[i][0],fixedptlist[i][1],'ko',markeredgecolor='k')
            axes.text(fixedptlist[i][0],fixedptlist[i][1],'$(a_'+str(i+1)+',b_'+str(i+1)+')$',
                      horizontalalignment=halign[i],verticalalignment=valign[i])
            #axes.text(fixedptlist[i][0],fixedptlist[i][1],'$(a_'+str(i)+',b_'+str(i)+') = ('+str(a1)+','+str(b1)+')$',
            #horizontalalignment=halign[i],verticalalignment=valign[i])


    ## iPRC
    fig2 = plt.figure(figsize=(6,6))
    axes2 = fig2.add_axes([.1,.1,.8,.8])
    #mp.figure()
    plt.title('iPRC of 2D Glass Network (Inhibitory Feedback): Analytical vs Direct')
    #mp.title('iPRC of 2D Glass Network (Inhibitory Feedback): Analytical vs Direct')
    p2, = axes2.plot(np.linspace(0,1,len(adjoint_solx)),adjoint_soly,color='.5',lw=2) # analytic
    p1, = axes2.plot(np.linspace(0,1,len(adjoint_soly)),adjoint_solx,color='b',lw=2) 
    p4, = axes2.plot(np.linspace(0,1,total_perts),y_prc/pert/(2*math.pi),marker='o',linestyle='None',color='.5',markeredgecolor='.5') # direct
    p3, = axes2.plot(np.linspace(0,1,total_perts),x_prc/pert/(2*math.pi),'bo')
    mp.legend([p1,p2,p3,p4],['Adjoint x', 'Adjoint y', 'Direct x','Direct y'])
    #mp.xlim((0,1))
    #mp.ylim((0,1))


    # debugging
    """
    debug = glass2d_phase_reset(0, fixedpts, dx=0, dy=pert,return_intermediates=True)
    mp.figure()
    mp.plot(debug['vals2'][:,0],debug['vals2'][:,1])
    print LCinit(1,fixedpts)
    """

    mp.show()

if __name__ == "__main__":
    main()
