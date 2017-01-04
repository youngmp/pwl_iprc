import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

theta = 1.
delta=.5

tarr = np.loadtxt('x1.dat')[:,0]

lc1arr = np.loadtxt('x1.dat')[:,1]
lc2arr = np.loadtxt('x2.dat')[:,1]
lc3arr = np.loadtxt('x3.dat')[:,1]

z1arr = np.loadtxt('z1.dat')[:,1]
z2arr = np.loadtxt('z2.dat')[:,1]
z3arr = np.loadtxt('z3.dat')[:,1]

lc1 = interp1d(tarr,lc1arr)
lc2 = interp1d(tarr,lc2arr)
lc3 = interp1d(tarr,lc3arr)

z1 = interp1d(tarr,z1arr)
z2 = interp1d(tarr,z2arr)
z3 = interp1d(tarr,z3arr)

def W(eps=.25,delta=.5):
    """
    return 3x3 weight matrix
    """
    W = np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
    W[W==1]=-1.+eps
    W[W==0.]=-1.-delta

    di = np.diag_indices(3)
    W[di] = 0

    #W[np.array([0,1,2])] = 0

    return W

def heav(x):
    if x > 0.:
        return 1.
    else:
        return 0.
    
def h(phi):
    """
    h function
    """
    #phi = np.linspace(0,tarr[-1],100)
    t = np.linspace(0,tarr[-1],100) # integration variable
    w = W()


    
    j = np.array([0,1,2])

    tot = 0.

    for k in range(len(t)):
        sum1 = w[0,0]*lc1(t[k])+w[0,1]*lc2(t[k])+w[0,2]*lc3(t[k])
        sum2 = w[1,0]*lc1(t[k])+w[1,1]*lc2(t[k])+w[1,2]*lc3(t[k])
        sum3 = w[2,0]*lc1(t[k])+w[2,1]*lc2(t[k])+w[2,2]*lc3(t[k])

        sum4 = (-1-delta)*(lc1(np.mod(t[k]+phi,tarr[-1]))+\
               lc2(np.mod(t[k]+phi,tarr[-1]))+\
               lc3(np.mod(t[k]+phi,tarr[-1])))

        v1 = z1(t[k])*heav(sum1+theta)*sum4
        v2 = z2(t[k])*heav(sum2+theta)*sum4
        v3 = z3(t[k])*heav(sum3+theta)*sum4

        tot += v1+v2+v3

    return tot*tarr[-1]/(tarr[-1]*len(t))

def heav_arr(x):
    return 0.5 * (np.sign(x) + 1)


def tlent_rhs():
    pass

def main():
    # with given parameters, period is 11.25
    w = W()
    print w
    
    #print 'test'

    fig = plt.figure()
    ax = plt.subplot(111)
    phi = np.linspace(0,tarr[-1],len(tarr))

    #ax.plot(phi,h(phi))
    ax.plot(phi/phi[-1],h(-phi)-h(phi))
    ax.set_title('-2hodd')
    ax.set_xlim(tarr[0],tarr[-1]/tarr[-1])
    ax.set_xlabel('phase (fraction of period)')

    if True:
        a = np.zeros((len(phi),2))
        a[:,0] = phi
        a[:,1] = h(-phi)-h(phi)
        np.savetxt('tlnet-2hodd.dat',a)

    sum4 = -(lc1(np.mod(tarr+phi,tarr[-1]))+\
             lc2(np.mod(tarr+phi,tarr[-1]))+\
             lc3(np.mod(tarr+phi,tarr[-1])))

    """
    ax.plot(z1(tarr)*heav_arr(w[0,0]*lc1(tarr)+w[0,1]*lc2(tarr)+w[0,2]*lc3(tarr)+theta)*sum4)
    ax.plot(z2(tarr)*heav_arr(w[1,0]*lc1(tarr)+w[1,1]*lc2(tarr)+w[1,2]*lc3(tarr)+theta)*sum4)
    ax.plot(z3(tarr)*heav_arr(w[2,0]*lc1(tarr)+w[2,1]*lc2(tarr)+w[2,2]*lc3(tarr)+theta)*sum4)
    """

    """
    ax.plot(tarr,lc1(tarr))
    ax.plot(tarr,lc2(tarr))
    ax.plot(tarr,lc3(tarr))
    """

    """
    ax.plot(tarr,z1(tarr))
    ax.plot(tarr,z2(tarr))
    ax.plot(tarr,z3(tarr))
    """
    plt.show()

if __name__ == "__main__":
    main()
