#!/usr/bin/python
import numpy as np
from scipy import integrate
from scipy import optimize
import math


def iris(y, unused_t, a=0., l_ccw=0.2, l_cw=-1., X=1., Y=1.):
    # determine the saddle whose neighborhood we are in
    if y[0] >= -a/2 and y[1] > a/2: # upper right
        s = np.array([X - a/2, Y + a/2])
        l = np.array([l_cw, l_ccw])
    elif y[0] > a/2 and y[1] <= a/2: # lower right
        s = np.array([Y + a/2, -X + a/2])
        l = np.array([l_ccw, l_cw])
    elif y[0] <= a/2 and y[1] < -a/2: # lower left
        s = np.array([-X + a/2, -Y - a/2])
        l = np.array([l_cw, l_ccw])
    elif y[0] < -a/2 and y[1] >= -a/2: # lower right
        s = np.array([-Y - a/2, X - a/2])
        l = np.array([l_ccw, l_cw])
    else:
        return 0*y
        #return -y
        #raise ValueError(
        #        "({0},{1}) is in the center a by a square."
        #            .format(y[0], y[1])
        #        )

    return (y-s)*l;


def sine_system(y, unused_t, mu=-0.2, alpha=0.23333, k=1.):
    r"""Compute the gradient of the sine system.

    Computes the gradient of sine system:

    .. math::

        \frac{d\mathbf{y}}{dt} = 
            \left( \begin{array}{cc}
            1 & -\mu \\
            \mu & 1 \\
            \end{array} \right)
            \left( \begin{array}{c}
            \cos(y_0) \sin(y_1) + \alpha \sin(2 k \; y_0) \\
            -\sin(y_0) \cos(y_1) + \alpha \sin(2 k \; y_1) \\
            \end{array} \right)

    :param y: the (two dimensional) point where the gradient is sampled
    :param unused_t: the simulation time when the gradient is sampled
    :type unused_t: float
    :param mu: a parameter controlling scaling and rotation of the system 
    :param alpha: a parameter controlling the local strength of the central 
        focus's attraction/repulsion
    :param k: a parameter controlling the number of limit cycles
    :type k: int

    :rtype: A two dimensional vector
    """
    #print y
    f = np.cos(y[0])*np.sin(y[1]) + alpha*np.sin(2*k*y[0])
    g = -np.sin(y[0])*np.cos(y[1]) + alpha*np.sin(2*k*y[1])
    return np.array([f - mu*g, g + mu*f])


def dwell_time(y0, l_u=0.1, l_s=-1., X=1., Y=1.):
    if y0 <= 0:
        return 1e100
    else:
        return 1./l_u * math.log(Y/y0);


def exit_position(y0, l_u=0.1, l_s=-1., X=1., Y=1.):
    return X * math.exp(l_s * dwell_time(y0, l_u, l_s, X, Y));


def iris_fixedpoint(a=0., l_ccw=0.2, l_cw=-1., X=1., Y=1., guess=1e-6):
    try:
        r0 = optimize.newton(
                lambda x: exit_position(x, l_ccw, l_cw, X, Y) + a + Y - X - x, 
                guess)
        if abs(exit_position(r0, l_ccw, l_cw, X, Y) + a + Y - X - r0) > 1e-7:
            raise RuntimeError # Why is this reported as convergent?
        return r0

    except RuntimeError:
        return None # no limit cycle


def iris_period(a=0., l_ccw=0.2, l_cw=-1., X=1., Y=1.):
    r0 = iris_fixedpoint(a, l_ccw, l_cw, X, Y)
    if r0 == None: # no limit cycle
        return None
    else:
        return 4 * dwell_time(r0, l_ccw, l_cw, X, Y)


def sine_limit_cycle(mu=-0.2, alpha=0.23333, k=1., max_time=200, 
        max_steps=100000):

    # run for a while
    t = np.linspace(0, max_time, max_steps)
    vals = integrate.odeint(sine_system, 
            [0, -math.pi/2], 
            t, args=(mu, alpha, k))
    
    # calculate the most recent time a new cycle was started
    x_section = 0.
    crossings = ((vals[:-1,0] > x_section) * (vals[1:,0] <= x_section) 
            * (vals[1:,1] < 0))
    if crossings.sum() < 2:
        raise RuntimeError("No complete cycles")

    # linearly interpolate between the two nearest points
    crossing_fs = ((vals[1:,0][crossings] - x_section)
            / (vals[1:,0][crossings]-vals[:-1,0][crossings]) )
    crossing_ys = (crossing_fs * vals[:-1,1][crossings] 
            + (1-crossing_fs) * vals[1:,1][crossings])
    crossing_times = (crossing_fs * t[:-1][crossings] 
            + (1-crossing_fs) * t[1:][crossings])

    return ( crossing_times[-1] - crossing_times[-2], crossing_ys[-1],
            abs(crossing_ys[-1]- crossing_ys[-2]) )


def sine_phase_reset(phi, dx=0., dy=0., mu=-0.2, alpha=0.23333, k=1.,
        steps_per_cycle = 10000, num_cycles = 10, return_intermediates=False,
        y0 = None, T = None):

    if y0 is None or T is None:
        T, y0, error = sine_limit_cycle(mu, alpha, k)

    steps_before = int(phi/(2*math.pi) * steps_per_cycle) + 1
    
    # run up to the perturbation
    t1 = np.linspace(0, phi/(2*math.pi) * T, steps_before)  
    vals1 = integrate.odeint(sine_system, 
            [0, y0], 
            t1, args=(mu, alpha, k))
    
    # run after the perturbation
    t2 = np.linspace(phi/(2*math.pi) * T, T * num_cycles, 
            steps_per_cycle * num_cycles - steps_before) 
    vals2 = integrate.odeint(sine_system, 
            list(vals1[-1,:] + np.array([dx, dy])), 
            t2, args=(mu, alpha, k))

    # calculate the most recent time a new cycle was started
    x_section = 0.
    crossings = ((vals2[:-1,0] > x_section) * (vals2[1:,0] <= x_section) 
            * (vals2[1:,1] < 0))
    if len(crossings) == 0:
        raise RuntimeError("No complete cycles after the perturbation")
    crossing_fs = ((vals2[1:,0][crossings] - x_section)
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


def phase_reset(phi, dx=0., dy=0., a=0., l_ccw=0.2, l_cw=-1., X=1., Y=1.,
        steps_per_cycle = 10000, num_cycles = 10, return_intermediates=False):
    r0 = iris_fixedpoint(a, l_ccw, l_cw, X, Y)
    T = iris_period(a, l_ccw, l_cw, X, Y)
    if r0 == None:
        raise RuntimeError("No limit cycle found")
    else:

        steps_before = int(phi/(2*math.pi) * steps_per_cycle) + 1
        
        # run up to the perturbation
        t1 = np.linspace(0, phi/(2*math.pi) * T, steps_before)  
        vals1 = integrate.odeint(iris, 
                [a/2, -a/2 - Y + r0], 
                t1, args=(a, l_ccw, l_cw, X, Y))
        
        # run after the perturbation
        t2 = np.linspace(phi/(2*math.pi) * T, T * num_cycles, 
                steps_per_cycle * num_cycles - steps_before) 
        vals2 = integrate.odeint(iris, 
                list(vals1[-1,:] + np.array([dx, dy])), 
                t2, args=(a, l_ccw, l_cw, X, Y))

        # calculate the most recent time a new cycle was started
        crossings = ((vals2[:-1,0] > a/2) * (vals2[1:,0] <= a/2) 
                * (vals2[1:,1] < 0))
        if len(crossings) == 0:
            raise RuntimeError("No complete cycles after the perturbation")
        #crossing_times = t2[1:][crossings]
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


def analytic_phase_reset_old(phi, dx=0., dy=0., 
        a=0., l_ccw=0.2, l_cw=-1., X=1., Y=1.):
    
    # this is an older expression derived with several assumptions
    assert(X == 1.)
    assert(Y == 1.)
    assert(l_cw == -1.)

    r0 = iris_fixedpoint(a, l_ccw, l_cw, X, Y)
    T = iris_period(a, l_ccw, l_cw, X, Y)
    if r0 == None:
        raise RuntimeError("No limit cycle found")
    else:
        quad1or2 = np.fmod(phi, 2 * math.pi) < math.pi
        quad1or3 = np.fmod(phi, math.pi) < math.pi/2
        du = (quad1or3 * dy + (1 - quad1or3) * dx) * (1 - 2*quad1or2)
        ds = (quad1or3 * dx + (1 - quad1or3) * dy) * (
                1 - 2*quad1or2*quad1or3 - 2*(1 - quad1or2)*(1 - quad1or3))
        t = np.fmod(phi, math.pi/2)/(math.pi/2) * T/4
        Q = 1/(l_ccw * r0) * np.exp(-l_ccw * t)
        dt0 = -Q * du
        dr = np.exp(-T/4) * (X * -dt0 + np.exp(t) * ds) 
        return (
                (dt0 
                + -1./(l_ccw * r0) 
                * 1./(1 - 1./(l_ccw * r0) * (r0/Y)**(1/l_ccw)) * dr)
                / T * 2*math.pi
                )
        #return dt0/3 + 2*-1./(l_ccw * r0) * 1./(1 + Q) * dr


def analytic_phase_reset(phi, dx=0., dy=0., 
        a=0., l_ccw=0.2, l_cw=-1., X=1., Y=1.):
    
    # non-dimensionalize things
    dx /= X
    dy /= Y
    L = -l_cw/l_ccw

    ui = iris_fixedpoint(a, l_ccw, l_cw, X, Y)
    T = l_ccw * iris_period(a, l_ccw, l_cw, X, Y)/4
    if ui == None:
        raise RuntimeError("No limit cycle found")
    else:
        quad1or2 = np.fmod(phi, 2 * math.pi) < math.pi
        quad1or3 = np.fmod(phi, math.pi) < math.pi/2
        du = (quad1or3 * dy + (1 - quad1or3) * dx) * (1 - 2*quad1or2)
        ds = (quad1or3 * dx + (1 - quad1or3) * dy) * (
                1 - 2*quad1or2*quad1or3 - 2*(1 - quad1or2)*(1 - quad1or3))
        phi1 = np.fmod(phi, math.pi/2)
        return -math.pi/(2*T*(1.-L*ui**(L-1)))*(
            ui**-(1-2*phi1/math.pi) * du 
            + ui**(L*(1-2*phi1/math.pi)-1) * ds 
            )


def saddle_isochron(x, y,
        a=0., l_ccw=0.2, l_cw=-1., X=1., Y=1., numperiods = 100):
    r0 = iris_fixedpoint(a, l_ccw, l_cw, X, Y)
    T = iris_period(a, l_ccw, l_cw, X, Y)

    if r0 == None:
        raise RuntimeError("No limit cycle found")
    else:
        # first, trace everything to an edge
        dwell_times = 1./l_ccw * np.log(Y/np.abs(y));
        times = T/4 - dwell_times
        # determine the edge and position along the edge by which we renter
        # the saddle
        edge = np.sign(y)
        rs = x * np.exp(l_cw * dwell_times) + a*edge;  
        
        # then run things forward for the appropriate number of periods
        for i in range(4*numperiods - 1):            
            dwell_times = 1./l_ccw * np.log(Y/np.abs(rs));
            times += T/4 - dwell_times
            # determine the edge and position along the edge by which we renter
            # the saddle
            rs, edge = (
                    edge * X * np.exp(l_cw * dwell_times) + a * np.sign(rs), 
                    np.sign(rs)
                    )
        result = 2 * math.pi * times / T 

        # discard the phases of any points that haven't converged within 
        # some epsilon of the limit cycle
        epsilon = 1e-3
        converged = np.abs(rs - r0) < epsilon
        result[~converged] = np.nan

        return result


def iris_isochron(x, y,
        a=0., l_ccw=0.2, l_cw=-1., X=1., Y=1., numperiods = 100):

    # identify the points that fall into each of the saddles
    s0 = (x <=  a/2) * (y <  -a/2)
    s1 = (x <  -a/2) * (y >= -a/2)
    s2 = (x >= -a/2) * (y >   a/2)
    s3 = (x >   a/2) * (y <=  a/2)

    # remap the coordinates onto a single saddle 
    # (mapping extra coordinates to (0,0))
    saddle_x = np.zeros_like(x)
    saddle_y = np.zeros_like(y)
    saddle_x[s0] =  (x[s0] - a/2 + X)
    saddle_y[s0] =  (y[s0] + a/2 + Y)
    saddle_x[s1] = -(y[s1] + a/2 - Y)
    saddle_y[s1] =  (x[s1] + a/2 + X)
    saddle_x[s2] = -(x[s2] + a/2 - X)
    saddle_y[s2] = -(y[s2] - a/2 - Y)
    saddle_x[s3] =  (y[s3] - a/2 + Y)
    saddle_y[s3] = -(x[s3] - a/2 - X)

    # trim the parts that overflow the edge of the saddles
    overflow = (np.abs(saddle_x) > X) + (np.abs(saddle_y) > Y)
    saddle_x[overflow] = 0.
    saddle_y[overflow] = 0.

    # calculate the isochrons for the single saddle
    result = saddle_isochron(saddle_x, saddle_y, 
        a, l_ccw, l_cw, X, Y, numperiods)

    # add an adjustment for distance around the iris
    result[s1] += math.pi / 2
    result[s2] += math.pi
    result[s3] += 3 * math.pi/2

    return result


import matplotlib.pyplot as plt
import subprocess

def draw_saddle_neighborhood(ax, x, y, width, height, x_stable, y_stable, 
        scale=1):
    facecolor = (0.92, 0.92, 0.92)
    ax.add_patch(plt.Rectangle((x, y), width, height, fc = facecolor))
    pointsize = 0.04 * scale
    headwidth = 2 * pointsize
    headlength = 2 * 1.61 * pointsize
    linewidth = 0.1 * scale
    xs = x + width/2
    ys = y + height/2
    dx = width/2 - pointsize - headlength
    dy = height/2 - pointsize - headlength
    
    arrows = []
    if x_stable:
        arrows += [
                [x, ys, dx, 0],
                [x + width, ys, -dx, 0],
                ]
    else:
        arrows += [
                [xs + pointsize, ys, dx, 0],
                [xs - pointsize, ys, -dx, 0],
                ]

    if y_stable:
        arrows += [
                [xs, y, 0, dy],
                [xs, y + height, 0, -dy],
                ]
    else:
        arrows += [
                [xs, ys + pointsize, 0, dy],
                [xs, ys - pointsize, 0, -dy],
                ]

       
    for arrow in arrows:
        ax.arrow(arrow[0], arrow[1], arrow[2], arrow[3],
                head_width=headwidth, head_length=headlength,
                linewidth=linewidth,
                facecolor = (0,0,0))
    
    ax.add_patch(plt.Circle( (x + width/2., y + height/2.), pointsize, 
        fc = (1,1,1), fill=True))


def draw_iris(ax, a=0., l_ccw=0.2, l_cw=-1., X=1., Y=1., offset=(0.,0.)):
    draw_saddle_neighborhood(ax, -a/2+offset[0], a/2+offset[1], 2*X, 2*Y, True,
            False)
    draw_saddle_neighborhood(ax, a/2+offset[0], a/2-2*X+offset[1], 2*Y, 2*X,
            False, True)
    draw_saddle_neighborhood(ax, a/2-2*X+offset[0], -a/2-2*Y+offset[1], 2*X,
            2*Y, True, False)
    draw_saddle_neighborhood(ax, -a/2-2*Y+offset[0], -a/2+offset[1], 2*Y, 2*X,
            False, True)


def draw_fancy_iris(ax, a=0., l_ccw=0.2, l_cw=-1., X=1., Y=1., 
        x0=None, 
        #x0_rev=None, 
        tmax=100,
        scale=1., offset=(0.,0.)):
    # draw the iris
    draw_iris(ax, a, l_ccw, l_cw, X, Y, offset)
    
    offset = np.asarray(offset)
    max_step = 0.5


    # draw the unstable limit cycle
    r0u = iris_fixedpoint(a, l_ccw, l_cw, X, Y, guess=Y)
    if r0u != None:
        if a/2 + X - r0u > 0:
            vals = integrate.odeint(iris, 
                    [-a/2, a/2 + Y - r0u], 
                    np.linspace(0, 
                        4 * dwell_time(r0u, l_ccw, l_cw, X, Y), 1000), 
                    args=(a, l_ccw, l_cw, X, Y))
            ax.plot(vals[:,0]+offset[0], vals[:,1]+offset[1], 'r-', lw=1)
        else:
            pointsize = 0.04 * scale
            ax.add_patch(plt.Circle( (0., 0.) + offset, pointsize, 
                fc = (1,1,1), fill=True))

    # draw the stable limit cycle
    if a != 0:
        r0s = iris_fixedpoint(a, l_ccw, l_cw, X, Y, guess=1e-6*X)
        if r0s != None:
            vals = integrate.odeint(iris, 
                    [-a/2, a/2 + Y - r0s], 
                    np.linspace(0, 4 * dwell_time(r0s, l_ccw, l_cw, X, Y), 1000), 
                    args=(a, l_ccw, l_cw, X, Y))
            lc_color = ['b','k'][x0 != None and np.isnan(x0)]
            ax.plot(vals[:,0]+offset[0], vals[:,1]+offset[1], lc_color, lw=2)

    # draw a sample trajectory
    if x0 == None:
        if a != 0 and r0u != None:
            x0 = [-a/2, a/2 + Y - (0.9*r0u + 0.1*r0s)]
        elif a == 0:
            x0 = [-a/2, a/2 + Y - 0.9]
        else:
            x0 = [-a/2, 0.9*Y]
    if np.isfinite(x0).all():
        vals = integrate.odeint(iris, x0, np.linspace(0,tmax,10000), 
                args=(a, l_ccw, l_cw, X, Y))
        good = ((vals[1:,:] - vals[:-1,:])**2).sum(axis=1) < max_step**2
        good *= np.abs(vals[:-1,:]).max(axis=1) >= a/2 # ignore the inner square
        vals = np.resize(vals, (len(good), 2))
        ax.plot(vals[good,0]+offset[0], vals[good,1]+offset[1], lw=0.5)


def draw_fancy_sine_system(ax, 
        mu=-0.2, alpha=0.23333, k=1., 
        max_time=200, max_steps=100000, scale=1.):

    # draw the saddles
    pointsize = 0.04 * scale
    ax.add_patch(plt.Circle( (math.pi/2., math.pi/2.), pointsize, 
        fc = (1,1,1), fill=True))
    ax.add_patch(plt.Circle( (-math.pi/2., math.pi/2.), pointsize, 
        fc = (1,1,1), fill=True))
    ax.add_patch(plt.Circle( (math.pi/2., -math.pi/2.), pointsize, 
        fc = (1,1,1), fill=True))
    ax.add_patch(plt.Circle( (-math.pi/2., -math.pi/2.), pointsize, 
        fc = (1,1,1), fill=True))

    # draw the central unstable focus
    ax.add_patch(plt.Circle( (0,0), pointsize, 
        fc = (1,1,1), fill=True))
    
    # draw the stable limit cycle
    if mu < 0:
        T, y0, error = sine_limit_cycle(mu, alpha, k, max_time, max_steps)
        vals = integrate.odeint(sine_system, 
                [0, y0], 
                np.linspace(0, T, 1000), 
                args=(mu, alpha, k))
        ax.plot(vals[:,0], vals[:,1], 'k', lw=2)


def animate_iris(filename="iris_animation.mp4", 
        a=(0.001, 0.8), l_ccw=0.1, l_cw=-1., X=1., Y=1., 
        num_frames = 60, fps=10, num_cycles = 1):

    def s(f, pair):
        return (1 - f) * pair[0] + f * pair[1]

    fig = plt.figure()
    
    # for simplicity, extend all parameters into a start and end value 
    if not hasattr(a, '__getitem__'):
        a = (a, a)
    if not hasattr(l_ccw, '__getitem__'):
        l_ccw = (l_ccw, l_ccw)
    if not hasattr(l_cw, '__getitem__'):
        l_cw = (l_cw, l_cw)
    if not hasattr(X, '__getitem__'):
        X = (X, X)
    if not hasattr(Y, '__getitem__'):
        Y = (Y, Y)
    for i in range(num_frames):
        f = (1. - math.cos(num_cycles * 2*math.pi * i/num_frames)) / 2
        fig.clear()
        ax = fig.add_subplot(111, aspect="equal")
        draw_fancy_iris(ax, s(f,a), s(f,l_ccw), s(f,l_cw), s(f,X), s(f,Y))
        max_size = 2*max(max(X),max(Y)) + max(a)
        plt.xlim(-max_size, max_size)
        plt.ylim(-max_size, max_size)
        fig.savefig("frame%04d.png" % i)
    
    #subprocess.call(["/usr/bin/mencoder", "mf://frame*.png", "-mf", 
    #    "type=png:fps=30", "-ovc", "lavc", "-lavcopts", 
    #    "vcodec=wmv2", "-oac", "copy", "-o", "animation.mpg"]) 
    subprocess.call(["/usr/bin/mencoder", 
        "mf://frame*.png", "-mf", "type=png:fps=%d" % fps, 
        "-of", "lavf", "-lavfopts", "format=mp4", "-vf-add", "harddup", 
        "-oac", "lavc", "-ovc", "lavc", "-lavcopts",
        "aglobal=1:vglobal=1:vcodec=mpeg4:vbitrate=1000:keyint=25",
        "-oac", "copy", "-o", filename]) 
