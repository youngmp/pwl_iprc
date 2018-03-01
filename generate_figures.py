import iris
import iris_modified
import nominal_biting
import glass_pasternack_2d
import math
import multiprocessing
import tempfile

import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import odeint
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', serif=['Computer Modern Roman'])
import matplotlib
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath \usepackage{bm}']
from matplotlib import pyplot as plt
import matplotlib.pylab as mp

from matplotlib.patches import Circle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d

# code from http://stackoverflow.com/questions/11140163/python-matplotlib-plotting-a-3d-cube-a-sphere-and-a-vector
#draw a vector
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import pw_const as pwc
from xppcall import xpprun

pi = np.pi

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)



# code from iris.py was written by Kendrick Shaw

def iris_mod_fig(a, border=1., label_theta0=False,
                 X=1.,Y=1.,maxsteps=10000):
    
    # make sure these params match those in iris_mod_prc_fig
    l_cw_SW = -2.5
    l_cw_NW = -1.5
    l_cw_NE = -3.
    l_cw_SE = -4.

    l_ccw_SW = 1.
    l_ccw_NW = 1.
    l_ccw_NE = 1.
    l_ccw_SE = 1.
    saddle_val = (l_cw_SW,l_ccw_SW,l_cw_NW,l_ccw_NW,l_cw_NE,l_ccw_NE,l_cw_SE,l_ccw_SE)

    fig = plt.figure(figsize=(8,8))
    axes = plt.axes([0.,0.,1.,1.])

    # draw iris shape
    iris.draw_iris(axes, a=a, l_ccw=1, l_cw=-2, X=X, Y=Y, offset=(0.,0.))

    # print regin numbers
    if a == 0.:
        axes.text(-1.8,-1.8,"1",fontsize=40)
        axes.text(-1.8,1.6,"2",fontsize=40)
        axes.text(1.7,1.6,"3",fontsize=40)
        axes.text(1.7,-1.8,"4",fontsize=40)
    axes.set_xlim([-2.2,2.2])
    axes.set_ylim([-2.2,2.2])

    # LCinit
    if a != 0 and a <= .24:
        u1,u2,u3,u4 = iris_modified.LCinit(saddle_val,a=a,return_all=True)
        if a == 0.05:
            shift = .8
        elif a == .2:
            shift = .4
        else:
            shift = .1
        x0 = [-a/2, a/2 + Y - u3-shift]
        
        axes.annotate(r'$\theta = 0$',
                      xy=(a/2, -X + u1 - a/2), xycoords='data',
                      xytext=(15,15), textcoords='offset points',
                      arrowprops=dict(arrowstyle='->',
                                      connectionstyle='angle,angleA=180,angleB=240,rad=10')
                      )

        # period
        t1,t2,t3,t4 = (iris_modified.ToF(u1),iris_modified.ToF(u2),iris_modified.ToF(u3),iris_modified.ToF(u4))
        T = t1+t2+t3+t4

        # draw sample trajectory, not limit cycle
        t = np.linspace(0,4*T,maxsteps)
        vals = odeint(iris_modified.iris_mod,x0,t,args=(a,saddle_val))

    elif a == 0:
        x0 = [-a/2, a/2 + Y - 0.9]
        t = np.linspace(0,100,maxsteps)
        vals = odeint(iris_modified.iris_mod,x0,t,args=(a,saddle_val))    

    elif a >= .255:
        x0 = [-a/2, a/2 + Y - .2]
        t = np.linspace(0,50,maxsteps)
        vals = odeint(iris_modified.iris_mod,x0,t,args=(a,saddle_val))    

    axes.annotate(r'',
                  xy=x0, xycoords='data',
                  xytext=x0 + np.r_[-1.0e-3, 0.5e-3], textcoords='data',
                  arrowprops=dict(arrowstyle='->', color='b',
                                  connectionstyle='arc3,rad=0')
                  )
    
    
    axes.plot(vals[:,0],vals[:,1])
    axes.axis('off')
    axes.set_xticks([])
    axes.set_yticks([])
    
    return fig

def glass_2d_fig():
    fig = plt.figure(figsize=(7,6))
    axes = plt.axes([.1,.1,.8,.8])

    # fixed points
    a1,b1 = [-5.,11.]
    a2,b2 = [-10.,-4.]
    a3,b3 = [6.,-10.]
    a4,b4 = [10.,5.]

    # consoldate fixed points
    fixedpts = (a1,b1,a2,b2,a3,b3,a4,b4)
    
    maxsteps = 100000
    # total integration time
    T = glass_pasternack_2d.ToF(0,fixedpts)+\
        glass_pasternack_2d.ToF(1,fixedpts)+\
        glass_pasternack_2d.ToF(2,fixedpts)+\
        glass_pasternack_2d.ToF(3,fixedpts)
    t = np.linspace(0,T,maxsteps)
    
    # draw axes
    axes.plot([-15,15],[0,0],color='0.5') # x-axis
    axes.plot([0,0],[-15,15],color='0.5') # y-axis
    axes.text(11,.5,'x')
    axes.text(-1,11,'y')

    # calculate inside trajectory
    yin = [.5,0.]
    valsin = odeint(glass_pasternack_2d.D2Attractor,
                    yin,
                    1.5*t,
                    args=((a1,b1),(a2,b2),(a3,b3),(a4,b4)))


    # calculate outside trajectory
    yout = [10.,0.]
    valsout = odeint(glass_pasternack_2d.D2Attractor,
                     yout,
                     1.5*t,
                     args=((a1,b1),(a2,b2),(a3,b3),(a4,b4)))


    # calculate limit cycle
    lcval = odeint(glass_pasternack_2d.D2Attractor,
                   [glass_pasternack_2d.LCinit(0,fixedpts),0],
                   t,
                   args=((a1,b1),(a2,b2),(a3,b3),(a4,b4)))

    # draw fixed points and lines to them from limit cycle
    axes.plot([0,a1],[glass_pasternack_2d.LCinit(1,fixedpts),b1],'k--')
    axes.plot([glass_pasternack_2d.LCinit(2,fixedpts),a2],[0,b2],'k--')
    axes.plot([0,a3],[glass_pasternack_2d.LCinit(3,fixedpts),b3],'k--')
    axes.plot([glass_pasternack_2d.LCinit(0,fixedpts),a4],[0,b4],'k--')

    # draw theta = 0
    axes.annotate(r'$\theta = 0$',
                  xy=(glass_pasternack_2d.LCinit(0,fixedpts),0), xycoords='data',
                  xytext=(15,-15), textcoords='offset points',
                  ha='left',
                  arrowprops=dict(arrowstyle='->',
                                  connectionstyle='arc3,rad=-.5')
                  )


    # draw dashed line from outer trajectory in region 1 to fixed point (a_1,b_1)
    crossingsout = ((valsout[:-1,0] > 0) * (valsout[1:,0] <= 0) *
                 (valsout[1:,1]>0))
    axes.plot([0,a1],[np.amax(valsout[1:,1][crossingsout]),b1],ls='--',color='.6')

    # draw dashed line from inner trajectory in region 1 to fixed point (a_1,b_1)
    crossingsin = ((valsin[:-1,0] > 0) * (valsin[1:,0] <= 0) *
                 (valsin[1:,1]>0))
    axes.plot([0,a1],[np.amin(valsin[1:,1][crossingsin]),b1],ls='--',color='.6')


    # plot inside
    axes.plot(valsin[:,0],valsin[:,1],lw=1,color='b')

    # plot outside
    axes.plot(valsout[:,0],valsout[:,1],lw=1,color='r')

    # plot limit cycle
    axes.plot(lcval[:,0],lcval[:,1],lw=3,color='purple')


    # draw fixed point labels
    fixedptlist = [[a1,b1],[a2,b2],[a3,b3],[a4,b4]]
    valign = ['bottom','top','bottom','bottom']
    halign = ['right','left','left','right']
    for i in range(len(fixedptlist)):
            axes.plot(fixedptlist[i][0],fixedptlist[i][1],'ko',markeredgecolor='k')
            axes.text(fixedptlist[i][0],fixedptlist[i][1],'$(a_'+str(i+1)+',b_'+str(i+1)+')$',
                      horizontalalignment=halign[i],verticalalignment=valign[i])
            #axes.text(fixedptlist[i][0],fixedptlist[i][1],'$(a_'+str(i)+',b_'+str(i)+') = ('+str(a1)+','+str(b1)+')$',
            #horizontalalignment=halign[i],verticalalignment=valign[i])


    # draw trajectory arrows

    # limit cycle:
    # index of time crossing:
    lc_time_choice = .16*T
    best_idx = np.argmin(np.abs(t - lc_time_choice))

    x = lcval[:,0][best_idx]
    y = lcval[:,1][best_idx]
    dx = lcval[:,0][best_idx] - lcval[:,0][best_idx-1]
    dy = lcval[:,1][best_idx] - lcval[:,1][best_idx-1]
    axes.arrow(x,y,dx,dy,
               head_width=.6,
               head_length=.6,
               fc='purple',
               ec='purple',
               )

    
    # inside trajectory
    inside_time_choice = .3*T
    best_idx = np.argmin(np.abs(t - inside_time_choice))

    x = valsin[:,0][best_idx]
    y = valsin[:,1][best_idx]
    dx = valsin[:,0][best_idx] - valsin[:,0][best_idx-1]
    dy = valsin[:,1][best_idx] - valsin[:,1][best_idx-1]
    axes.arrow(x,y,dx,dy,
               head_width=.3,
               head_length=.3,
               fc='blue',
               ec='blue',
               )


    # outside trajectory
    outside_time_choice = .2*T
    best_idx = np.argmin(np.abs(t - outside_time_choice))

    x = valsout[:,0][best_idx]
    y = valsout[:,1][best_idx]
    dx = valsout[:,0][best_idx] - valsout[:,0][best_idx-1]
    dy = valsout[:,1][best_idx] - valsout[:,1][best_idx-1]
    axes.arrow(x,y,dx,dy,
               head_width=.3,
               head_length=.3,
               fc='red',
               ec='red',
               )


    # set limits and clean up axes
    axes.set_xlim(-12,12)
    axes.set_ylim(-12,12)
    axes.set_xticks([])
    axes.set_yticks([])

    return fig


def nominal_biting_fig(a_vals=[.02,.001],rho=3.,maxsteps=10000):
    fig = plt.figure(figsize=(10,5))

    counter = 1
    for a in a_vals:
        ax = fig.add_subplot(1,2,counter,projection='3d')

        # calculate limit cycle
        t1,t2,t3,init12,init23,init31 = nominal_biting.LCinit(a=a,rho=rho)
        t = np.linspace(0,t1+t2+t3,maxsteps)
        sol = odeint(nominal_biting.piecewise,init31,t,args=(a,rho))

        R1shift = np.array([0,-a,a])
        R1O = np.array([0,0,0])+R1shift # R2 origin (R-one-
        RV1 = np.array([[1,0,0], [1,0,1], [1,1,0], [1,1,1]]) + np.array([R1shift,R1shift,R1shift,R1shift])

        # R2shift = np.array([a,-a,-a]) # use this if keeping R3 
        R2shift = np.array([a,0,-a])
        R2O = np.array([0,0,0])+R2shift # R3 
        RV2 = np.array([[0,1,0],[1,1,0],[0,1,1],[1,1,1]]) + np.array([R2shift,R2shift,R2shift,R2shift])
        
        # R3shift = np.array([0,0,
        R3shift = np.array([-a,a,0])
        R3O = np.array([0,0,0])+R3shift # R1 origin (R-one-
        RV3 = np.array([[0,0,1],[1,0,1],[0,1,1],[1,1,1]]) + np.array([R3shift,R3shift,R3shift,R3shift])

        # define color regions
        # chosen according to Fig. 16 of Okabe and Ito 2002
        # these colors are the least ambiguous between
        # color blind and non-color blind
        color_region1 = '#CC79A7' # reddish purple
        color_region2 = '#009E73' # blueish green
        color_region3 = '#D55E00' # vermillion

        linewidth = 1.5


        # a = Arrow3D([0,1],[0,1],[0,1], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")

        # plot eigenvectors (direction of stable/stable/unstable manifolds)
        # region 1
        xshift = np.array([1.,0.,0.])
        s1r1_begin = np.array([.99,0.,0.])+R1shift
        s1r1_end = np.array([1.,0.,0.])+R1shift

        s2r1_begin = np.array([0.,0.,.1])+R1shift + xshift
        s2r1_end = np.array([0.,0.,0.])+R1shift + xshift

        ur1_begin = .3*np.array([0.,0.,0.])+R1shift + xshift
        ur1_end = .3*np.array([0.-rho/2.,1.,0.])+R1shift + xshift

        s1r1 = Arrow3D([s1r1_begin[0],s1r1_end[0]],[s1r1_begin[1],s1r1_end[1]],[s1r1_begin[2],s1r1_end[2]],
                       mutation_scale=20, lw=1.5, arrowstyle="-|>",color=color_region1)

        s2r1 = Arrow3D([s2r1_begin[0],s2r1_end[0]],[s2r1_begin[1],s2r1_end[1]],[s2r1_begin[2],s2r1_end[2]],
                       mutation_scale=20, lw=1.5, arrowstyle="-|>",color=color_region1)

        ur1 = Arrow3D([ur1_begin[0],ur1_end[0]],[ur1_begin[1],ur1_end[1]],[ur1_begin[2],ur1_end[2]],
                      mutation_scale=20, lw=1.5, arrowstyle="-|>",color=color_region1)

        ax.add_artist(s1r1)
        ax.add_artist(ur1)
        ax.add_artist(s2r1)


        # region 2
        yshift = np.array([0.,1.,0.])
        s1r2_begin = np.array([0.,.99,0.])+R2shift
        s1r2_end = np.array([0.,1.,0.])+R2shift

        s2r2_begin = np.array([.01,0.,0.])+R2shift  + yshift
        s2r2_end = np.array([0.,0.,0.])+R2shift + yshift

        ur2_begin = .3*np.array([0.,0.,0.])+R2shift + yshift
        ur2_end = .3*np.array([0.,-rho/2.,1.])+R2shift + yshift

        s1r2 = Arrow3D([s1r2_begin[0],s1r2_end[0]],[s1r2_begin[1],s1r2_end[1]],[s1r2_begin[2],s1r2_end[2]],
                      mutation_scale=20, lw=1.5, arrowstyle="-|>",color=color_region2)
        s2r2 = Arrow3D([s2r2_begin[0],s2r2_end[0]],[s2r2_begin[1],s2r2_end[1]],[s2r2_begin[2],s2r2_end[2]],
                       mutation_scale=20, lw=1.5, arrowstyle="-|>",color=color_region2)
        ur2 = Arrow3D([ur2_begin[0],ur2_end[0]],[ur2_begin[1],ur2_end[1]],[ur2_begin[2],ur2_end[2]],
                      mutation_scale=20, lw=1.5, arrowstyle="-|>",color=color_region2)

        ax.add_artist(s1r2)
        ax.add_artist(s2r2)
        ax.add_artist(ur2)

        # region 3
        zshift = np.array([0.,0.,1.]) 
        s1r3_begin = np.array([0.,0.,.99]) + R3shift
        s1r3_end = np.array([0.,0.,1.]) + R3shift 

        s2r3_begin = np.array([0.,.01,0.]) + R3shift + zshift
        s2r3_end = np.array([0.,0.,0.]) + R3shift + zshift

        ur3_begin = .3*np.array([0.,0.,0.]) + R3shift + zshift
        ur3_end = .3*np.array([1.,0.,-rho/2.]) + R3shift + zshift

        s1r3 = Arrow3D([s1r3_begin[0],s1r3_end[0]],[s1r3_begin[1],s1r3_end[1]],[s1r3_begin[2],s1r3_end[2]],
                      mutation_scale=20, lw=1.5, arrowstyle="-|>",color=color_region3)
        s2r3 = Arrow3D([s2r3_begin[0],s2r3_end[0]],[s2r3_begin[1],s2r3_end[1]],[s2r3_begin[2],s2r3_end[2]],
                       mutation_scale=20, lw=1.5, arrowstyle="-|>",color=color_region3)
        ur3 = Arrow3D([ur3_begin[0],ur3_end[0]],[ur3_begin[1],ur3_end[1]],[ur3_begin[2],ur3_end[2]],
                      mutation_scale=20, lw=1.5, arrowstyle="-|>",color=color_region3)

        ax.add_artist(s1r3)
        ax.add_artist(s2r3)
        ax.add_artist(ur3)

        # draw limit cycle arrow
        lc_idx = len(sol[:,0])-1
        lc_begin = np.array([sol[:,0][lc_idx-1],sol[:,1][lc_idx-1],sol[:,2][lc_idx-1]])
        lc_end = init31
        lcarrow = Arrow3D([lc_begin[0],lc_end[0]],[lc_begin[1],lc_end[1]],[lc_begin[2],lc_end[2]],
                      mutation_scale=20, lw=1.5, arrowstyle="-|>",color='k')
        ax.add_artist(lcarrow)


        # these loops are deplorable but necessary to draw the edges in the correct order
        for i in range(4):
                ax.plot([R2O[0],RV2[i][0]],[R2O[1],RV2[i][1]],[R2O[2],RV2[i][2]],color=color_region2,lw=linewidth)

        for i in range(4):
            if i <= 3:
                ax.plot([R3O[0],RV3[i][0]],[R3O[1],RV3[i][1]],[R3O[2],RV3[i][2]],color=color_region3,lw=linewidth)

        for i in range(4):
            if i <= 3:
                ax.plot([R1O[0],RV1[i][0]],[R1O[1],RV1[i][1]],[R1O[2],RV1[i][2]],color=color_region1,lw=linewidth)

        ax.plot(sol[:,0],sol[:,1],sol[:,2],'k',lw=2)
        for i in range(4):
            if i >= 1 and i <=2:
                ax.plot([RV2[i][0],RV2[0][0]],[RV2[i][1],RV2[0][1]],[RV2[i][2],RV2[0][2]],color=color_region2,lw=linewidth)
                ax.plot([RV2[i][0],RV2[3][0]],[RV2[i][1],RV2[3][1]],[RV2[i][2],RV2[3][2]],color=color_region2,lw=linewidth)


        for i in range(4):
            if i >= 1 and i <=2:
                ax.plot([RV3[i][0],RV3[0][0]],[RV3[i][1],RV3[0][1]],[RV3[i][2],RV3[0][2]],color=color_region3,lw=linewidth)
                ax.plot([RV3[i][0],RV3[3][0]],[RV3[i][1],RV3[3][1]],[RV3[i][2],RV3[3][2]],color=color_region3,lw=linewidth)
            if i == 3:
                ax.plot([R3O[0],RV3[i][0]],[R3O[1],RV3[i][1]],[R3O[2],RV3[i][2]],color=color_region3,lw=linewidth)

        for i in range(4):
            if i >= 1 and i <=2:
                ax.plot([RV1[i][0],RV1[0][0]],[RV1[i][1],RV1[0][1]],[RV1[i][2],RV1[0][2]],color=color_region1,lw=linewidth)
                ax.plot([RV1[i][0],RV1[3][0]],[RV1[i][1],RV1[3][1]],[RV1[i][2],RV1[3][2]],color=color_region1,lw=linewidth)
            if i == 3:
                ax.plot([R1O[0],RV1[i][0]],[R1O[1],RV1[i][1]],[R1O[2],RV1[i][2]],color=color_region1,lw=linewidth)





        # draw saddle points
        saddle1 = np.array([1.,0.,0.]) + R1shift
        saddle2 = np.array([0.,1.,0.]) + R2shift
        saddle3 = np.array([0.,0.,1.]) + R3shift


        ax.plot([saddle1[0]],[saddle1[1]],[saddle1[2]],
                color='white',
                markeredgecolor=color_region1,
                ms=7,marker='o')
        ax.plot([saddle2[0]],[saddle2[1]],[saddle2[2]],
                color='white',
                markeredgecolor=color_region2,
                ms=7,marker='o')
        ax.plot([saddle3[0]],[saddle3[1]],[saddle3[2]],
                color='white',
                markeredgecolor=color_region3,
                ms=7,marker='o')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        

        #ax.y_axis.set_pane_color([1.,1.,1.,])
        ax.w_xaxis.set_pane_color((0.,0.,0.,))

        # nominal_biting.domains(ax,a)
        ax.view_init(35,35)


        # set font size
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, ax.zaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels() +
                     ax.get_zticklabels()):
            item.set_fontsize(15)
        counter += 1

    """
        # plot view rotated 180 degrees along x-y plane
    ax2 = fig.add_subplot(122,projection='3d')
    ax2.plot(sol[:,0],sol[:,1],sol[:,2],'k',lw=2)
    nominal_biting.domains(ax2,a)
    ax2.view_init(35,215)

    # set font size
    for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label, ax2.zaxis.label] +
                 ax2.get_xticklabels() + ax2.get_yticklabels() +
                 ax2.get_zticklabels()):
        item.set_fontsize(15)

    # bring plots close together
    # http://stackoverflow.com/questions/6541123/improve-subplot-size-spacing-with-many-subplots-in-matplotlib
    """
    fig.tight_layout()

    return fig

def nominal_biting_prc_fig(a):
    """
    Nominal biting PRC figure
    """
    # coupling constant
    rho = 3.

    fig = plt.figure(figsize=(6,5))
    axes = plt.axes([.15,.1,.75,.8])

    t1,t2,t3,init12,init23,init31,T = nominal_biting.LCinit(a=a,rho=rho,return_period=True)
    adjoint_solx,adjoint_soly,adjoint_solz = nominal_biting.pwl_biting_phase_reset_analytic(a,
                                                   tof_all=[t1,t2,t3],
                                                   init_all=[init12,init23,init31])

    print "Times of flight, t1, t2, t3:", t1,t2,t3
    print "LC init for region 2", init12
    print "LC init for region 3", init23
    print "LC init for region 1", init31


    total_perts = 75
    phis = np.linspace(0,2*math.pi,total_perts)

    pert = 1e-4

    #scaling
    scale = 1./(2.*math.pi)

    x_prc = np.array([
            nominal_biting.pwl_biting_phase_reset(phi, a=a, rho=rho,
                                 init_all=[init12,init23,init31],
                                 tof_all=[t1,t2,t3], dx=pert, dy=0.,dz=0.,T=T)
            for phi in phis
            ])

    y_prc = np.array([
            nominal_biting.pwl_biting_phase_reset(phi, a=a, rho=rho,
                                 init_all=[init12,init23,init31],
                                 tof_all=[t1,t2,t3],
                                 dx=0., dy=pert, dz=0.,T=T)
            for phi in phis
            ])


    z_prc = np.array([
            nominal_biting.pwl_biting_phase_reset(phi, a=a, rho=rho,
                                 init_all=[init12,init23,init31],
                                 tof_all=[t1,t2,t3],
                                 dx=0., dy=0., dz=pert,T=T)
            for phi in phis
            ])



    if pert != 0.0:
        x_prc = x_prc/pert
        y_prc = y_prc/pert
        z_prc = z_prc/pert

    phases_n = np.linspace(0,1,total_perts)

    #mp.title("iPRC of PWL SHC Nominal Biting Model: Direct vs Adjoint")
    phases_a = np.linspace(0,1,len(adjoint_solx))

    # Plot figures
    p3, = axes.plot(phases_a,adjoint_solz,color='.7',ls='--')
    p6, = axes.plot(phases_n,z_prc*scale,marker='D',
                    color='#a8d4a8',
                    markeredgecolor='#a8d4a8',
                    ls='None',
                    ms=3.)

    p2, = axes.plot(phases_a,adjoint_soly,color='.5',ls='-')
    p5, = axes.plot(phases_n,y_prc*scale,marker='s',
                    color='.5',
                    markeredgecolor='.5',
                    ls='None',
                    ms=3.) # color was set to .5

    p1, = axes.plot(phases_a,adjoint_solx,'k-')
    p4, = axes.plot(phases_n,x_prc*scale,'bo',
                    ls='None',
                    ms=3.)

    # set font size
    for item in ([axes.title, axes.xaxis.label, axes.yaxis.label] +
                 axes.get_xticklabels() + axes.get_yticklabels()):
        item.set_fontsize(15)
    axes.set_ylabel(r'$z(\theta)$')
    axes.set_xlabel(r'$\theta$')
    #mp.legend([p1,p2,p3,p4,p5,p6],['Adjoint x','Adjoint y','Adjoint z','Direct x','Direct y','Direct z'])
    return fig

def draw_fancy_iris_mod(ax, a=0., saddle_val=[], X=1., Y=1.,
        x0=None,
        #x0_rev=None,
        tmax=100,
        scale=1., offset=(0.,0.)):

    assert (len(saddle_val) != 0)
    # draw the iris
    iris.draw_iris(ax, a, l_ccw=1, l_cw=-2, X=1., Y=1., offset=(0.,0.))

    offset = np.asarray(offset)
    max_step = 0.5

    # draw the unstable limit cycle
    #r0u = iris_fixedpoint(a, l_ccw, l_cw, X, Y, guess=Y)
    
    """
    if u1 != None:
        if a/2 + X - u3 > 0:
            vals = integrate.odeint(iris_modified.iris_mod,
                    [-a/2, a/2 + Y - u3],
                    np.linspace(0,
                                iris_modified.ToF(u1)+
                                iris_modified.ToF(u2)+
                                iris_modified.ToF(u3)+
                                iris_modified.ToF(u4), 1000),
                    args=(a, saddle_val, X, Y))
            ax.plot(vals[:,0]+offset[0], vals[:,1]+offset[1], 'r-', lw=1)
        else:
            pointsize = 0.04 * scale
            ax.add_patch(plt.Circle( (0., 0.) + offset, pointsize,
                fc = (1,1,1), fill=True))
    """
    # draw the stable limit cycle

    if a != 0:
        u1,u2,u3,u4 = iris_modified.LCinit(saddle_val,a,return_all=True)
        if u3 != None:
            vals = odeint(iris_modified.iris_mod,
                    [-a/2, a/2 + Y - u3],
                    np.linspace(0, 
                                iris_modified.ToF(u1)+
                                iris_modified.ToF(u2)+
                                iris_modified.ToF(u3)+
                                iris_modified.ToF(u4), 1000),
                    args=(a, saddle_val, X, Y))
            lc_color = ['b','k'][x0 != None and np.isnan(x0)]
            ax.plot(vals[:,0]+offset[0], vals[:,1]+offset[1], lc_color, lw=2)

    # draw sample trajectory
    """
    if x0 == None:
        if a != 0 and r0u != None:
            x0 = [-a/2, a/2 + Y - (0.9*r0u + 0.1*r0s)]
        elif a == 0:
            x0 = [-a/2, a/2 + Y - 0.9]
        else:
            x0 = [-a/2, 0.9*Y]
    if np.isfinite(x0).all():
        vals = odeint(iris, x0, np.linspace(0,tmax,10000),
                args=(a, l_ccw, l_cw, X, Y))
        good = ((vals[1:,:] - vals[:-1,:])**2).sum(axis=1) < max_step**2
        good *= np.abs(vals[:-1,:]).max(axis=1) >= a/2 # ignore the inner square
        vals = np.resize(vals, (len(good), 2))
        ax.plot(vals[good,0]+offset[0], vals[good,1]+offset[1], lw=0.5)
    """

def iris_mod_prc_fig(a_vals=[1e-2,.1,.2,.24],border=0.3):
    # define saddle values
    l_cw_SW = -2.5#-2.5
    l_cw_NW = -1.5#-1.5
    l_cw_NE = -3.#-3.
    l_cw_SE = -4.#-4.

    l_ccw_SW = 1.
    l_ccw_NW = 1.
    l_ccw_NE = 1.
    l_ccw_SE = 1.

    # consolidate saddle values
    saddle_val = (l_cw_SW,l_ccw_SW,l_cw_NW,l_ccw_NW,l_cw_NE,l_ccw_NE,l_cw_SE,l_ccw_SE)

    X = 1.
    Y = 1.

    # perturbation size and numerical phase values
    total_pert = 80
    n_phis = np.linspace(0, 2*math.pi, total_pert)
    #dx = 1e-4
    #dy = 0.
    pert = 1e-4
    mag = pert #math.sqrt(dx**2 + dy**2)
    phasescale = 4 / (2 * math.pi)
    newscale = 1./(2.*math.pi)


    fig = plt.figure(figsize=(6,6))

    width = 1./len(a_vals)
    padding = 0.2*width

    for i in range(len(a_vals)):
        a = a_vals[i]

        axes = plt.axes((2*padding, 1-(i+1) * width+padding,
            1 - width - 2*padding, width - 1.5*padding))

        # find iris initial condition
        u1,u2,u3,u4 = iris_modified.LCinit(saddle_val,a,return_all=True)

    
        # find times of flight
        t1,t2,t3,t4 = [iris_modified.ToF(u1),
                       iris_modified.ToF(u2),
                       iris_modified.ToF(u3),
                       iris_modified.ToF(u4)]
        #print "t1-t4", t1,t2,t3,t4,'and u1-u4', u1,u2,u3,u4,'a',a

        # total period
        T = t1+t2+t3+t4

        # calculate analytic iPRC
        a_prc,a_prc_o = iris_modified.iris_mod_phase_reset_analytic(
            saddle_val=saddle_val,
            a=a,
            tof_all=[t1,t2,t3,t4],
            u_all=[u1,u2,u3,u4],
            dt=.001)
        


        # calculate PRC for perturbations in x direction
        x_prc = np.array([
                iris_modified.iris_mod_phase_reset(phi, 
                                              a=a,
                                              saddle_val=saddle_val,
                                              u=[u1,u2,u3,u4],
                                              dx=pert, dy=0.)
                for phi in n_phis
                ])

        # calculate PRC for perturbations in y direction

        y_prc = np.array([
                iris_modified.iris_mod_phase_reset(phi,
                                              a=a,
                                              saddle_val=saddle_val,
                                              u=[u1,u2,u3,u4],
                                              dx=0., dy=pert)
                for phi in n_phis
                ])


        # draw vertical lines denoting boundary crossings
        ymin = 5*np.amin(a_prc)
        ymax = 5*np.amax(a_prc)
        # ffd5d5 super light red
        # ff8c8c lighter red
        # ff6262 light red
        axes.vlines(t1/T * 2*math.pi,ymin,ymax,color='#ff6262',linestyles='dotted')
        axes.vlines((t1+t2)/T * 2*math.pi,ymin,ymax,color='#ff6262',linestyles='dotted')
        axes.vlines((t1+t2+t3)/T * 2*math.pi,ymin,ymax,color='#ff6262',linestyles='dotted')
        axes.vlines((t1+t2+t3+t4)/T * 2*math.pi,ymin,ymax,color='#ff6262',linestyles='dotted')

        # analytic y-prc
        axes.plot(np.linspace(0,2*math.pi,len(a_prc)), a_prc_o, color='0.8')

        # draw numerical y-prc
        axes.plot(np.linspace(0,2*math.pi,len(y_prc)),
                  y_prc/mag * newscale,
                  marker="o",linestyle='None',
                  markeredgecolor='0.8', color='0.8',
                  ms=3.)

        # analytic x-prc
        axes.plot(np.linspace(0,2*math.pi,len(a_prc)), a_prc, color='k')

        # numerical x-prc
        axes.plot(np.linspace(0,2*math.pi,len(x_prc)),
                  x_prc/mag * newscale,
                  marker="o",linestyle='None',color='blue',
                  ms=3.)



        # clean up axes
        axes.set_xlim(0, 2*math.pi)
        plt.xticks(np.arange(5.)*math.pi/2,
                ['$0$', '$0.25$', '$0.5$', '$0.75$', '$1$'])

        # make the y-axis symmetric around zero
        ymaxabs = np.max(np.abs([a_prc,a_prc_o])*1.1)#axes.get_ylim()))
        axes.set_ylim(-ymaxabs, ymaxabs)


        # draw phase plane for reference
        axes = plt.axes((1-width, 1-(i+1) * width, width, width))
        draw_fancy_iris_mod(axes, a, saddle_val, X, Y,
                                 scale=3.0, x0=np.nan)

        axes.set_xlim(-2*X-border, 2*X+border)
        axes.set_ylim(-2*Y-border, 2*Y+border)
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_frame_on(False)

    return fig

def glass_2d_prc_fig():
    # create figure
    fig = plt.figure(figsize=(6,5))
    axes = fig.add_axes([0.15, 0.1, 0.75, 0.8])

    # define parameters

    a1,b1 = [-5.,11.]
    a2,b2 = [-10.,-4.]
    a3,b3 = [6.,-10.]
    a4,b4 = [10.,5.]

    """
    # homogenous parameters lead to very good fits.
    a1,b1 = [-5.,10.]
    a2,b2 = [-10.,-5.]
    a3,b3 = [5.,-10.]
    a4,b4 = [10.,5.]
    """
    fixedpts = (a1,b1,a2,b2,a3,b3,a4,b4)

    # calculate analytic adjoint
    adjoint_solx,adjoint_soly = glass_pasternack_2d.glass_2d_phase_reset_analytic(fixedpts)
    # plot
    a_plot_phis = np.linspace(0,1,len(adjoint_solx))
    axes.plot(a_plot_phis, adjoint_soly,color='.5')

    total_perts = 80
    pert = 1e-2
    phis = np.linspace(0,2*math.pi,total_perts)
    
    #scale
    scale = 1./(2.*math.pi)

    # calculate iPRC for perturbations in the x direction
    x_prc = np.array([
            glass_pasternack_2d.glass2d_phase_reset(phi, fixedpts, dx=pert, dy=0)
            for phi in phis
            ])
    if pert != 0.0:
        x_prc = x_prc/pert

    # calculate iPRC for perturbations in the y direction
    y_prc = np.array([
            glass_pasternack_2d.glass2d_phase_reset(phi, fixedpts, dx=0, dy=pert)
            for phi in phis
            ])
    if pert != 0.0:
        y_prc = y_prc/pert
        
    # plot numerical y-iPRC
    n_plot_phis = np.linspace(0,1,total_perts)
    axes.plot(n_plot_phis,y_prc*scale,marker='o',
              linestyle='None',color='.5',
              markeredgecolor='.5',ms=3.)

    # plot analytic x-iPRC then numerical x-iPRC
    axes.plot(a_plot_phis, adjoint_solx,color='k')
    axes.plot(n_plot_phis,x_prc*scale,'bo',ms=3.)


    # set font size
    for item in ([axes.title, axes.xaxis.label, axes.yaxis.label] +
                 axes.get_xticklabels() + axes.get_yticklabels()):
        item.set_fontsize(15)

    axes.set_ylim([np.amin([y_prc*scale,x_prc*scale])-.025,np.amax([y_prc*scale,x_prc*scale])+.025])
    axes.set_ylabel(r'$z(\theta)$')
    axes.set_xlabel(r'$\theta$')


    return fig

def glass_pert_plane_fig(phase_of_pert=1.5*math.pi-.5, num_cycles=1.5, pert_size=2.):
    """
    A failed attempted at showing the effect of perturbations from the phase plane perspective.
    """
    fig = plt.figure(figsize=(8,8))
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # make sure that these parameters are the same between 
    # all the glass figures, i.e., glass_2d_prc_fig
    # and glass_2d_fig.

    # fixed points
    a1,b1 = [-5.,11.]
    a2,b2 = [-10.,-4.]
    a3,b3 = [6.,-10.]
    a4,b4 = [10.,5.]

    # consolidate fixed points for use in functions
    fixedpts = (a1,b1,a2,b2,a3,b3,a4,b4)

    # find period and limit cycle initial condition
    T = glass_pasternack_2d.ToF(0,fixedpts)+\
        glass_pasternack_2d.ToF(1,fixedpts)+\
        glass_pasternack_2d.ToF(2,fixedpts)+\
        glass_pasternack_2d.ToF(3,fixedpts)

    y0 = glass_pasternack_2d.LCinit(0,fixedpts)
    
    # get perturbed trajectory
    prc_dict = glass_pasternack_2d.glass2d_phase_reset(phase_of_pert,fixedpts,
                                                       dx=pert_size,steps_per_cycle=10000,
                                                       num_cycles=num_cycles,
                                                       return_intermediates=True)    

    maxsteps = 10000
    maxtime = prc_dict['t2'][-1]
    t = np.linspace(prc_dict['t1'][-1],maxtime,maxsteps)

    vals_lc = odeint(glass_pasternack_2d.D2Attractor,
                     prc_dict['vals1'][-1],
                     t,
                     args=((a1,b1),(a2,b2),(a3,b3),(a4,b4)))


    axes.plot(vals_lc[:,0],vals_lc[:,1],color='.5',ls='--')

    # plot first portion of limit cycle through t1
    # and plot perturbed limit cycle trajectory through t2
    #axes.plot(prc_dict['vals1'][:,0],prc_dict['vals1'][:,1],color='k',lw=2)
    #axes.plot([prc_dict['t1'][-1],prc_dict['t2'][0]],
    #          [prc_dict['vals1'][:,0][-1],prc_dict['vals2'][:,0][0]],color='k',lw=2)
    axes.plot(prc_dict['vals2'][:,0],prc_dict['vals2'][:,1],color='k',lw=2)

    return fig

def glass_phase_fig(num_cycles=3.5):
    fig = plt.figure(figsize=(6,8))
    axes = fig.add_subplot(211)
    # make sure that these parameters are the same between 
    # all the glass figures, i.e., glass_2d_prc_fig
    # and glass_2d_fig.

    # fixed points
    a1,b1 = [-5.,11.]
    a2,b2 = [-10.,-4.]
    a3,b3 = [6.,-10.]
    a4,b4 = [10.,5.]

    # consolidate fixed points for use in functions
    fixedpts = (a1,b1,a2,b2,a3,b3,a4,b4)

    # find period and limit cycle initial condition
    T = glass_pasternack_2d.ToF(0,fixedpts)+\
        glass_pasternack_2d.ToF(1,fixedpts)+\
        glass_pasternack_2d.ToF(2,fixedpts)+\
        glass_pasternack_2d.ToF(3,fixedpts)

    y0 = glass_pasternack_2d.LCinit(0,fixedpts)

    # total time of integration
    maxsteps = 10000
    maxtime = T*num_cycles
    t = np.linspace(0,maxtime,maxsteps)

    # calculate limit cycle
    vals_lc = odeint(glass_pasternack_2d.D2Attractor,
                     [y0,0.],
                     t,
                     args=((a1,b1),(a2,b2),(a3,b3),(a4,b4)))

    # plot x_1 and x_2 state variables
    axes.plot(t,vals_lc[:,1],'.75',lw=2)
    axes.plot(t,vals_lc[:,0],'k',lw=2)

    # annotate period
    max_lc =np.amax( vals_lc[:,0])
    axes.annotate('',xy=(T,np.amax(max_lc)),
                xytext=(2*T,np.amax(max_lc)),
                ha="right", va="center",
                size=15,
                arrowprops=dict(arrowstyle='<->',
                                fc="w", ec="k",
                                connectionstyle="arc3,rad=0.05",
                                )
                )
    
    # find period to single decimal place
    approx_period = int(T*10.)/10.

    axes.annotate(r'$T \approx '+str(approx_period)+'$',
                  xy=(T,max_lc),
                  xytext=(T+T/3.,max_lc+.3),
                  size=15)
    # set axis limit
    axes.set_xlim([0,t[-1]])
    #axes.set_ylabel('$x_1(t)$')

    # resize plot labels
    for item in ([axes.title, axes.xaxis.label, axes.yaxis.label] +
                 axes.get_xticklabels() + axes.get_yticklabels()):
        item.set_fontsize(20)

    # theta values subplot
    axes = fig.add_subplot(212)

    # define theta values and plot
    thetavals = np.linspace(0,t[-1]/T,len(vals_lc))%1
    axes.plot(t,thetavals,'k',lw=2)

    # set theta plot lables and limits
    axes.set_ylabel(r'$\theta$')
    axes.set_xlim([0,t[-1]])
    axes.set_xlabel('$t$')

    # resize plot labels
    for item in ([axes.title, axes.xaxis.label, axes.yaxis.label] +
                 axes.get_xticklabels() + axes.get_yticklabels()):
        item.set_fontsize(20)


    return fig

def glass_pert_displacemnt_fig(phase_of_pert=1.5*math.pi-.5, num_cycles=3.5, pert_size=2.):
    """
    show an example of asymptotic effects of perturbations
    """
    
    fig = plt.figure(figsize=(7,5))
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # make sure that these parameters are the same between 
    # all the glass figures, i.e., glass_2d_prc_fig
    # and glass_2d_fig.

    # fixed points
    a1,b1 = [-5.,11.]
    a2,b2 = [-10.,-4.]
    a3,b3 = [6.,-10.]
    a4,b4 = [10.,5.]    

    # consolidate fixed points for use in functions
    fixedpts = (a1,b1,a2,b2,a3,b3,a4,b4)

    # find period and limit cycle initial condition
    T = glass_pasternack_2d.ToF(0,fixedpts)+\
        glass_pasternack_2d.ToF(1,fixedpts)+\
        glass_pasternack_2d.ToF(2,fixedpts)+\
        glass_pasternack_2d.ToF(3,fixedpts)

    y0 = glass_pasternack_2d.LCinit(0,fixedpts)

    # get perturbed trajectory
    prc_dict = glass_pasternack_2d.glass2d_phase_reset(phase_of_pert,fixedpts,
                                                       dx=pert_size,steps_per_cycle=10000,
                                                       num_cycles=num_cycles,
                                                       return_intermediates=True)

    # calculate the "would-have-been' limit cycle
    maxsteps = 10000
    maxtime = prc_dict['t2'][-1]
    t = np.linspace(prc_dict['t1'][-1],maxtime,maxsteps)

    vals_lc = odeint(glass_pasternack_2d.D2Attractor,
                     prc_dict['vals1'][-1],
                     t,
                     args=((a1,b1),(a2,b2),(a3,b3),(a4,b4)))
    
    # plot 'would-have-been' limit cycle 
    axes.plot(t,vals_lc[:,0],color='.5',ls='--')

    # plot first portion of limit cycle through t1
    # and plot perturbed limit cycle trajectory through t2
    axes.plot(prc_dict['t1'],prc_dict['vals1'][:,0],color='k',lw=2)
    #axes.plot([prc_dict['t1'][-1],prc_dict['t2'][0]],
    #          [prc_dict['vals1'][:,0][-1],prc_dict['vals2'][:,0][0]],color='k',lw=2)
    axes.plot(prc_dict['t2'],prc_dict['vals2'][:,0],color='k',lw=2)

    # draw perturbation
    axes.annotate('',
                  #xy=(prc_dict['t1'][-1],prc_dict['vals1'][:,0][-1]-2*pert_size),
                  xy=(prc_dict['t2'][0],prc_dict['vals2'][:,0][0]),
                  xycoords='data',
                  #xytext=(prc_dict['t1'][-1],prc_dict['vals1'][:,0][-1]-3*pert_size),textcoords='data',
                  xytext=(prc_dict['t1'][-1],prc_dict['vals1'][:,0][-1]),textcoords='data',
                  va='center',ha='center',
                  arrowprops=dict(arrowstyle="simple",
                                  connectionstyle="angle3",
                                  color='red',
                                  fc='w'),
                  )
    # label perturbation
    """
    axes.annotate('Impulse',
                  #xy=(prc_dict['t1'][-1],prc_dict['vals1'][:,0][-1]-2*pert_size),
                  xy=(prc_dict['t2'][0],prc_dict['vals2'][:,0][0]),
                  xycoords='data',
                  #xytext=(prc_dict['t1'][-1],prc_dict['vals1'][:,0][-1]-3*pert_size),textcoords='data',
                  xytext=(prc_dict['t1'][-1]-pert_size/16.,prc_dict['vals1'][:,0][-1]+pert_size/2.),textcoords='data',
                  va='center',ha='right',
                  #arrowprops=dict(arrowstyle="simple",
                  #                connectionstyle="arc3",
                  #                color='red'),
                  )
                  
    """

    # draw arrows showing phase difference
    final_cross_time_pert = prc_dict['crossing_times'][-1]
    crossing_idx = prc_dict['crossings']
    one_max_pert_val = prc_dict['vals2'][:,0][crossing_idx][-1]
    final_cross_time_lc = int(num_cycles)*T
    axes.annotate('',
                  xy=(final_cross_time_pert,one_max_pert_val),
                  xycoords='data',
                  #xytext=(prc_dict['t1'][-1],prc_dict['vals1'][:,0][-1]-3*pert_size),textcoords='data',
                  xytext=(final_cross_time_lc,one_max_pert_val),textcoords='data',
                  va='center',ha='left',
                  arrowprops=dict(arrowstyle="<->",
                                  connectionstyle="arc3,rad=0.2",
                                  ),
                  )
    
    # draw phase difference label
    x_position = (final_cross_time_pert + final_cross_time_lc)/2.
    axes.annotate(r'$\Delta \theta$',
                  size=15,
                  xy=(x_position,one_max_pert_val+.3),
                  xycoords='data',
                  xytext=(x_position,one_max_pert_val+.3),textcoords='data',
                  va='bottom',ha='center',
                  )


    # set axis limit
    axes.set_xlim(0,prc_dict['t2'][-1])
    
    # set axis labels
    axes.set_xlabel('$t$')
    axes.set_ylabel('$x_1(t)$')

    for item in ([axes.title, axes.xaxis.label, axes.yaxis.label] +
                 axes.get_xticklabels() + axes.get_yticklabels()):
        item.set_fontsize(15)


    return fig


def tlnet_fig():
    t1 = np.loadtxt('tlnet_phase_and_full_t.txt')
    tlnet_full1 = np.loadtxt('tlnetphase_full.txt')
    tlnetphase1 = np.loadtxt('tlnetphase_theory.txt')
    tlnetsol1 = np.loadtxt('tlnet_full_sol.txt')

    # phase shifted by almost half period
    t2 = np.loadtxt('tlnet_phase_and_full_t2.txt')
    tlnet_full2 = np.loadtxt('tlnetphase_full2.txt')
    tlnetphase2 = np.loadtxt('tlnetphase_theory2.txt')
    tlnetsol2 = np.loadtxt('tlnet_full_sol2.txt')

    dt = t1[2]-t1[1]
    cutoff = int(20/dt)

    fig = plt.figure(figsize=(10,10))

    ax1 = fig.add_subplot(321)
    # first 20 time units of ex 1

    # LC 2
    ax1.plot(t1[:cutoff],tlnetsol1[5,:cutoff],color='#8CD6FF',lw=2,ls='--',dashes=(5,1))
    ax1.plot(t1[:cutoff],tlnetsol1[4,:cutoff],color='#40A8E3',lw=3,ls='--',dashes=(5,1))


    # LC 1
    ax1.plot(t1[:cutoff],tlnetsol1[2,:cutoff],color='#FFBE8A',lw=2)
    ax1.plot(t1[:cutoff],tlnetsol1[1,:cutoff],color='#EB8D42',lw=3)


    ax1.plot(t1[:cutoff],tlnetsol1[3,:cutoff],color='#0072B2',lw=4,ls='--',dashes=(5,1)) # LC 2
    ax1.plot(t1[:cutoff],tlnetsol1[0,:cutoff],color='#D55E00',lw=4) # LC 1

    ax1.set_ylabel(r'\textbf{Activity}',fontsize=20)
    ax1.set_xlabel(r'\textbf{t}',fontsize=20)
    ax1.set_title(r'(a)',x=0,y=1.05,fontsize=20)

    # set tick label font size
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(15) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(15) 



    
    ax2 = fig.add_subplot(322)
    # first 20 time units of ex2

    # LC 2
    ax2.plot(t2[:cutoff],tlnetsol2[5,:cutoff],color='#8CD6FF',lw=2,ls='--',dashes=(5,1))
    ax2.plot(t2[:cutoff],tlnetsol2[4,:cutoff],color='#40A8E3',lw=3,ls='--',dashes=(5,1))


    # LC 1
    ax2.plot(t2[:cutoff],tlnetsol2[2,:cutoff],color='#FFBE8A',lw=2)
    ax2.plot(t2[:cutoff],tlnetsol2[1,:cutoff],color='#EB8D42',lw=3)

    
    ax2.plot(t2[:cutoff],tlnetsol2[3,:cutoff],color='#0072B2',lw=4,ls='--',dashes=(5,1)) #LC 2
    ax2.plot(t2[:cutoff],tlnetsol2[0,:cutoff],color='#D55E00',lw=4) #LC 1

    ax2.set_xlabel(r'\textbf{t}',fontsize=20)
    ax2.set_title(r'(b)',x=0,y=1.05,fontsize=20)

    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(15) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(15) 


    ax3 = fig.add_subplot(323)
    # last 20 time units of ex 1

    ax3.plot(t1[-cutoff:],tlnetsol1[5,-cutoff:],color='#8CD6FF',lw=2,ls='--',dashes=(5,1))
    ax3.plot(t1[-cutoff:],tlnetsol1[4,-cutoff:],color='#40A8E3',lw=3,ls='--',dashes=(5,1))


    ax3.plot(t1[-cutoff:],tlnetsol1[2,-cutoff:],color='#FFBE8A',lw=2)
    ax3.plot(t1[-cutoff:],tlnetsol1[1,-cutoff:],color='#EB8D42',lw=3)

    ax3.plot(t1[-cutoff:],tlnetsol1[3,-cutoff:],color='#0072B2',lw=4,ls='--',dashes=(5,1)) # LC 2
    ax3.plot(t1[-cutoff:],tlnetsol1[0,-cutoff:],color='#D55E00',lw=4) # LC 1

    ax3.set_ylabel(r'\textbf{Activity}',fontsize=20)
    ax3.set_xlabel(r'\textbf{t}',fontsize=20)
    ax3.set_title(r'(c)',x=0,y=1.05,fontsize=20)

    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(15) 
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(15) 


    ax4 = fig.add_subplot(324)
    # last 20 time units of ex 2

    # LC 2
    ax4.plot(t2[-cutoff:],tlnetsol2[5,-cutoff:],color='#8CD6FF',lw=2,ls='--',dashes=(5,1))
    ax4.plot(t2[-cutoff:],tlnetsol2[4,-cutoff:],color='#40A8E3',lw=3,ls='--',dashes=(5,1))

    # LC 1
    ax4.plot(t2[-cutoff:],tlnetsol2[2,-cutoff:],color='#FFBE8A',lw=2)
    ax4.plot(t2[-cutoff:],tlnetsol2[1,-cutoff:],color='#EB8D42',lw=3)

    ax4.plot(t2[-cutoff:],tlnetsol2[3,-cutoff:],color='#0072B2',lw=4,ls='--',dashes=(5,1)) # LC 2
    ax4.plot(t2[-cutoff:],tlnetsol2[0,-cutoff:],color='#D55E00',lw=4) # LC 1

    ax4.set_xlabel(r'\textbf{t}',fontsize=20)
    ax4.set_title(r'(d)',x=0,y=1.05,fontsize=20)

    for tick in ax4.xaxis.get_major_ticks():
        tick.label.set_fontsize(15) 
    for tick in ax4.yaxis.get_major_ticks():
        tick.label.set_fontsize(15) 


    ax5 = fig.add_subplot(325)
    ax5.scatter(t1[0:-1:500],tlnet_full1[0:-1:500],color='black')
    ax5.plot(t1,tlnetphase1,color='#80bfff',lw=3,ls='--',dashes=(5,1))
    #ax5.plot(t1,tlnetphase1,color='#3399ff',lw=3)
    
    
    ax5.set_xlim(0,t1[-1])
    ax5.set_ylim(-.4,0)
    ax5.set_ylabel(r'\textbf{Phase Difference}',fontsize=20)
    ax5.set_xlabel(r'\textbf{t}',fontsize=20)
    ax5.set_title(r'(e)',x=0,y=1.05,fontsize=20)

    for tick in ax5.xaxis.get_major_ticks():
        tick.label.set_fontsize(15) 
    for tick in ax5.yaxis.get_major_ticks():
        tick.label.set_fontsize(15) 


    ax6 = fig.add_subplot(326)
    ax6.scatter(t2[0:-1:500],np.mod(tlnet_full2[0:-1:500],1),color='black',label='Numerical Phase Difference')
    ax6.plot(t2,tlnetphase2,color='#80bfff',label='Predicted Phase Difference',lw=3,ls='--',dashes=(5,1))

    ax6.set_xlim(0,t2[-1])
    ax6.set_ylim(.2,.55)
    #ax6.set_ylabel('Phase Difference')
    ax6.set_xlabel(r'\textbf{t}',fontsize=20)
    ax6.set_title(r'(f)',x=0,y=1.05,fontsize=20)

    ax6.legend(loc='lower center')

    for tick in ax6.xaxis.get_major_ticks():
        tick.label.set_fontsize(15) 
    for tick in ax6.yaxis.get_major_ticks():
        tick.label.set_fontsize(15) 


    plt.tight_layout()

    return fig


def oct_domain_fig():
    b = np.sqrt(2)
    a = np.sqrt(2)-1
    c = 1./b
    d = 1./a

    lw=1
    color='gray'


    fig = plt.figure(figsize=(5,5))

    # domain
    ax = fig.add_subplot(111)
    
    x = a + np.arange(0,10)
    y = x-np.sqrt(2)
    ax.plot(x,y,color,lw=lw)

    x = -a + np.arange(0,10)
    y = -np.ones(len(x))
    ax.plot(x,y,color,lw=lw)

    x = np.arange(-1,10)
    y = -x-np.sqrt(2)
    ax.plot(x,y,color,lw=lw)

    y = a-np.arange(0,10)
    x = -np.ones(len(y))
    ax.plot(x,y,color,lw=lw)

    x = -a-np.arange(0,10)
    y = x + np.sqrt(2)
    ax.plot(x,y,color,lw=lw)

    x = a-np.arange(0,10)
    y = np.ones(len(x))
    ax.plot(x,y,color,lw=lw)

    x = np.arange(1,-10,-1)
    y = -x + np.sqrt(2)
    ax.plot(x,y,color,lw=lw)
    
    y = -a + np.arange(0,10)
    x = np.ones(len(y))
    ax.plot(x,y,color,lw=lw)


    # solutions

    # example trajectory 1 inside
    npa,vn = xpprun('limit_cycle_pw_const_coupled.ode',
                    inits={'x1':-1+.5,'y1':2.41421-.5},
                    parameters={'meth':'euler',
                                'dt':.001,
                                'eps':0.,
                                'total':1},
                    clean_after=True)
    
    t = npa[:,0]
    vals = npa[:,1:3]
    ax.plot(vals[:,0],vals[:,1],lw=1,color='blue')
    arrowx1 = vals[int(len(t)/16.),0]
    arrowy1 = vals[int(len(t)/16.),1]
    ax.arrow(arrowx1,arrowy1,.1,0.,head_width=0.1,fc='blue',ec='blue')

    # example trajectory 2 outisde
    npa,vn = xpprun('limit_cycle_pw_const_coupled.ode',
                    inits={'x1':-1-.5,'y1':2.41421+.5},
                    parameters={'meth':'euler',
                                'dt':.001,
                                'eps':0.,
                                'total':1.2},
                    clean_after=True)
    
    t = npa[:,0]
    vals = npa[:,1:3]
    ax.plot(vals[:,0],vals[:,1],lw=1,color='red')
    arrowx1 = vals[int(len(t)/16.),0]
    arrowy1 = vals[int(len(t)/16.),1]
    ax.arrow(arrowx1,arrowy1,.1,0.,head_width=0.1,fc='red',ec='red')


    # limit cycle
    t,lc = pwc.oct_lc()
    ax.plot(lc[:,0],lc[:,1],color='purple',lw=2)


    # first arrow
    arrowx1 = lc[int(len(t)/16.),0]
    arrowy1 = lc[int(len(t)/16.),1]
    ax.arrow(arrowx1,arrowy1,.1,0.,head_width=0.15,fc='purple',ec='purple')

    # second arrow 
    arrowx2 = lc[int(len(t)*(5./8.-1./16)),0]
    arrowy2 = lc[int(len(t)*(5./8.-1./16)),1]
    ax.arrow(arrowx2,arrowy2,-.1,0.,head_width=0.15,fc='purple',ec='purple')


    # zero phase label
    ax.annotate(r'$\theta = 0$',
                fontsize=20,
                xy=(1., .9), xycoords='data',
                xytext=(-70,-50), textcoords='offset points',
                arrowprops=dict(arrowstyle='->',
                                connectionstyle='angle,angleA=90,angleB=220,rad=10')
            )

    
    # region labels
    ax.text(1-.35,1,'1.',fontsize=20)
    ax.text(1+.3,0.+.3,'2.',fontsize=20)
    ax.text(1.05,-1.+.25,'3.',fontsize=20)
    ax.text(0.3,-1.-.35,'4.',fontsize=20)
    ax.text(-0.8,-1-.2,'5.',fontsize=20)
    ax.text(-1-0.35,-.35,'6.',fontsize=20)
    ax.text(-1-0.2,.65,'7.',fontsize=20)
    ax.text(-.4,1.3,'8.',fontsize=20)

    #ax.plot()

    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
    #ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])



    return fig


def oct_coupled_fig():
    
    fig = plt.figure(figsize=(10,3))

    simt = np.loadtxt('oct_ex_x1.dat')[:,0]
    x1 = np.loadtxt('oct_ex_x1.dat')[:,1]
    y1 = np.loadtxt('oct_ex_y1.dat')[:,1]

    x2 = np.loadtxt('oct_ex_x2.dat')[:,1]
    y2 = np.loadtxt('oct_ex_y2.dat')[:,1]

    # LCs
    lct = np.loadtxt('oct_x1.dat')[:,0]
    lc1_table = np.loadtxt('oct_x1.dat')[:,1]
    lc2_table = np.loadtxt('oct_y1.dat')[:,1]

    ax1 = fig.add_subplot(131)


    cutoff = int(5/.0001)
    # LC 2 y coord, LC 1 y coord
    ax1.plot(simt[:cutoff],y2[:cutoff],color='#8CD6FF',ls='--',dashes=(5,1))
    ax1.plot(simt[:cutoff],y1[:cutoff],color='#FFBE8A')

    # LC 2 x coord, LC 1 x coord
    ax1.plot(simt[:cutoff],x2[:cutoff],color='#0072B2',ls='--',dashes=(5,1))
    ax1.plot(simt[:cutoff],x1[:cutoff],color='#D55E00')

    ax1.set_xlabel(r'$\bm{t}$')
    ax1.set_ylim(-2.7,2.7)


    ax2 = fig.add_subplot(132)
    cutoff = int(95/.0001)
    # LC 2 y coord, LC 1 y coord
    ax2.plot(simt[cutoff:],y2[cutoff:],color='#8CD6FF',ls='--',dashes=(5,1))
    ax2.plot(simt[cutoff:],y1[cutoff:],color='#FFBE8A')

    # LC 2 x coord, LC 1 x coord
    ax2.plot(simt[cutoff:],x2[cutoff:],color='#0072B2',ls='--',dashes=(5,1))
    ax2.plot(simt[cutoff:],x1[cutoff:],color='#D55E00')

    ax2.set_xlabel(r'$\bm{t}$')
    ax2.set_ylim(-2.7,2.7)

    ### H FUNCTION STUFF
    #phis = np.linspace(0,2*math.pi,total_perts)
    phi = np.linspace(0,1,1000)
    prc = pwc.oct_phase_reset_analytic(phi)
    
    prc1 = interp1d(phi,prc[:,0])
    prc2 = interp1d(phi,prc[:,1])
    
    # create lc lookup table
    t,vals = pwc.oct_lc()
    lc1 = interp1d(t,vals[:,0])
    lc2 = interp1d(t,vals[:,1])

    phi2 = np.linspace(0,1,100)
    hvals = pwc.generate_h(phi2,lc1,lc2,prc1,prc2)

    hvals_bad = pwc.generate_h_bad(phi2,lc1,lc2,prc1,prc2)
    
    h = interp1d(phi2,hvals)
    h_bad = interp1d(phi2,hvals_bad)
    
    ax3 = fig.add_subplot(133)
    t = np.linspace(0,100,10000)
    sol = odeint(pwc.phase_rhs,.49,t,args=(h,))
    sol_bad = odeint(pwc.phase_rhs,.49,t,args=(h_bad,))

    
    # EXTRACT PHASES
    phase1 = np.zeros(len(simt))
    phase2 = np.zeros(len(simt))

    """
    v1 = (lc1_table-x1[0])**2. + (lc2_table-y1[0])**2.
    i1 = np.argmin(v1)
    x1size = 1.*len(lc1_table)
    phase1[0] = i1/x1size

    v2 = (lc1_table-x2[0])**2. + (lc2_table-y2[0])**2.
    i2 = np.argmin(v2)
    phase2[0] = i2/x1size

    for i in range(1,len(simt)):
        v1 = (lc1_table-x1[i])**2. + (lc2_table-y1[i])**2.
        i1 = np.argmin(v1)
        phase1[i] = i1/x1size

        v2 = (lc1_table-x2[i])**2. + (lc2_table-y2[i])**2.
        i2 = np.argmin(v2)
        phase2[i] = i2/x1size

    """

    v1 = np.arctan2(y1[0],x1[0])
    phase1[0] = (v1+pi)/(2*np.pi)

    v2 = np.arctan2(y2[0],x2[0])
    phase2[0] = (v2+pi)/(2*np.pi)

    for i in range(1,len(simt)):
        v1 = np.arctan2(y1[i],x1[i])
        phase1[i] = (v1+pi)/(2*np.pi)

        v2 = np.arctan2(y2[i],x2[i])
        phase2[i] = (v2+pi)/(2*np.pi)


    """
    crossings = ((y1[:-1] >= 0) * (y1[1:] < 0) 
                 * (x1[1:] > 0))

    crossings2 = ((y2[:-1] >= 0) * (y2[1:] < 0) 
                 * (x2[1:] > 0))

    crossing_times = simt[:-1][crossings][:1]
    crossing_times2 = simt[:-1][crossings2]

    crossing_phases = np.mod(crossing_times2-crossing_times+.5,1)-.5
    print len(crossing_phases)
    print len(crossings),len(crossings2)
    print len(crossing_times),len(crossing_phases)
    ax3.scatter(crossing_times2,crossing_phases)

    """

    phi = np.mod(phase1-phase2+.5,1)-.5
    #ax3.scatter(simt[0:-1:50],phi[0:-1:50],color='black',label='Numerical Phase Difference')
    ax3.scatter(simt[:-1:500],phi[:-1:500],color='black',label='Numerical Phase Difference')

    ax3.plot(t,sol,color='#80bfff',lw=3,ls='--',dashes=(5,1))
    ax3.plot(t,sol_bad,color='gray',lw=3,ls='--',dashes=(5,1))
    ax3.set_ylabel(r'\textbf{Phase Difference}')
    ax3.set_xlabel(r'$\bm{t}$')
    ax3.set_ylim(-.01,.51)
    ax3.set_xlim(0,100)
    
    plt.tight_layout()
    #plt.show()

    return fig

def oct_prc():

    phi = np.linspace(0,1,1000)
    prc = pwc.oct_phase_reset_analytic(phi)
    fig = plt.figure()
    ax = fig.add_subplot(111)


    ax.plot(phi,prc[:,1],lw=2,color='.5')
    ax.plot(phi,prc[:,0],lw=3,color='black')
    
    ax.set_ylabel(r'$\bm{z(\theta)}$')
    ax.set_xlabel(r'$\bm{\theta}$')


    return fig
    

def generate_figure(function, args, filenames, title="", title_pos=(0.5,0.95)):
    tempfile._name_sequence = None;
    fig = function(*args)
    fig.text(title_pos[0], title_pos[1], title, ha='center')
    if type(filenames) == list:
        for name in filenames:
            fig.savefig(name)
    else:
        fig.savefig(filenames)



def main():


    figures = [
        #(glass_pert_displacemnt_fig,[],['fig1_glass_pert_displacement_fig.pdf']),
        #(glass_2d_fig,[],['fig4_glass_2d_fig.pdf']),
        #(glass_2d_prc_fig,[],['fig5_glass_2d_prc_fig.pdf']),
        #(iris_mod_fig,[0., 0.2],['fig6_iris_mod_a_fig.pdf']),
        #(iris_mod_fig,[0.05, 0.2],['fig6_iris_mod_b_fig.pdf']),
        #(iris_mod_fig,[0.2, 0.2],['fig6_iris_mod_c_fig.pdf']),
        #(iris_mod_fig,[0.33, 0.2],['fig6_iris_mod_d_fig.pdf']),
        #(iris_mod_prc_fig,[],['fig7_iris_mod_prc_fig.pdf']),
        #(nominal_biting_fig,[],['fig8_nominal_biting_fig_comined.pdf']),
        #(nominal_biting_prc_fig,[.01],['fig9_nominal_biting_prc_fig.pdf']),
        #(tlnet_fig,[],['fig10_tlnet_fig.png','tlnet_fig.pdf']),
        #(oct_domain_fig,[],['fig11_oct_fig.pdf']),
        #(oct_prc,[],['fig12_oct_prc.pdf']),
        (oct_coupled_fig,[],['fig13_oct_coupled.pdf']),
    ]

    for fig in figures:
        generate_figure(*fig)

    """
    # set up one process per figure
    processes = [multiprocessing.Process(target=generate_figure, args=args)
            for args in figures]

    # start all of the processes
    for p in processes:
        p.start()

    # wait for everyone to finish
    for p in processes:
        p.join()

    #plt.show()
    """

if __name__ == '__main__':
    main()

