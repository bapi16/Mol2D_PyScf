import numpy as np
from scipy.special import hyp1f1, i0e as be, i1e, ive

#start_time = time.time()

def E(i,j,t,Qx,a,b):
    ''' Recursive definition of Hermite Gaussian coefficients.
        Returns a float.
        a: orbital exponent on Gaussian 'a' (e.g. alpha in the text)
        b: orbital exponent on Gaussian 'b' (e.g. beta in the text)
        i,j: orbital angular momentum number on Gaussian 'a' and 'b'
        t: number nodes in Hermite (depends on type of integral,
           e.g. always zero for overlap integrals)
        Qx: distance between origins of Gaussian 'a' and 'b'=Ax-Bx
    '''
    p = a + b
    q = a*b/p
    if (t < 0) or (t > (i + j)):
        # out of bounds for t
        return 0.0
    elif i == j == t == 0:
        # base case
        return np.exp(-q*Qx*Qx) # K_AB
    elif j==0:
        # decrement index i
        return (1/(2*p))*E(i-1,j,t-1,Qx,a,b) - \
               (q*Qx/a)*E(i-1,j,t,Qx,a,b)    + \
               (t+1)*E(i-1,j,t+1,Qx,a,b)
    else:
        # decrement index j
        return (1/(2*p))*E(i,j-1,t-1,Qx,a,b) + \
               (q*Qx/b)*E(i,j-1,t,Qx,a,b)    + \
               (t+1)*E(i,j-1,t+1,Qx,a,b)

def overlap(a,lmn1,A,b,lmn2,B):
    ''' Evaluates overlap integral between two Gaussians
        Returns a float.
        a:    orbital exponent on Gaussian 'a' (e.g. alpha in the text)
        b:    orbital exponent on Gaussian 'b' (e.g. beta in the text)
        lmn1: int tuple containing orbital angular momentum (e.g. (1,0,0))
              for Gaussian 'a'
        lmn2: int tuple containing orbital angular momentum for Gaussian 'b'
        A:    list containing origin of Gaussian 'a', e.g. [1.0, 2.0, 0.0]
        B:    list containing origin of Gaussian 'b'
    '''
    l1,m1 = lmn1 # shell angular momentum on Gaussian 'a'
    l2,m2 = lmn2 # shell angular momentum on Gaussian 'b'
    S1 = E(l1,l2,0,A[0]-B[0],a,b) # X
    S2 = E(m1,m2,0,A[1]-B[1],a,b) # Y
    return S1*S2*np.pi/(a+b)

def O(a,l1,A,b,l2,B):
    """this function evaluates the 1D overlap that we will use during the kinetic integrals
        a:    orbital exponent on Gaussian 'a' (e.g. alpha in the text)
        b:    orbital exponent on Gaussian 'b' (e.g. beta in the text)
       l1: angular momentum for Gaussian 'a'
        l2: angular momentum for Gaussian 'b'
        A:    origin of Gaussian 'a'
        B:    origin of Gaussian 'b'
    """
    S1 = E(l1,l2,0,A-B,a,b)
    return S1*np.power(np.pi/(a+b),0.5)

def S(a,b):
    s = 0.0
    for i in range(len(a.coefs)):
        for j in range(len(b.coefs)):
            s+= a.norm[i]*b.norm[j]*a.coefs[i]*b.coefs[j]*overlap(a.exps[i],a.shell,a.origin,b.exps[j],b.shell,b.origin)
            #print(s)
    return s


def kinetic(a,lmn1,A,b,lmn2,B):
    ''' Evaluates kinetic energy integral between two Gaussians
        Returns a float.
        a:   array of orbital exponent on Gaussian 'a' (e.g. alpha in the text) #in our case is an array since we work with anisotropic
        b:    array orbital exponent on Gaussian 'b' (e.g. beta in the text)
        lmn1: int array containing orbital angular momentum (e.g. [1,0,0]) for Gaussian 'a'
        lmn2: int array containing orbital angular momentum for Gaussian 'b'
        A:    list containing origin of Gaussian 'a', e.g. [1.0, 2.0, 0.0]
        B:    list containing origin of Gaussian 'b'
    '''
    nx,ny=lmn1
    mx,my=lmn2
    Ax,Ay=A
    Bx,By=B

    squarebracket1=0.5*nx*mx*O(a,nx-1,Ax,b,mx-1,Bx)-a*mx*O(a,nx+1,Ax,b,mx-1,Bx)-b*nx*O(a,nx-1,Ax,b,mx+1,Bx)+2*a*b*O(a,nx+1,Ax,b,mx+1,Bx)
    squarebracket2=0.5*ny*my*O(a,ny-1,Ay,b,my-1,By)-a*my*O(a,ny+1,Ay,b,my-1,By)-b*ny*O(a,ny-1,Ay,b,my+1,By)+2*a*b*O(a,ny+1,Ay,b,my+1,By)

    return O(a,ny,Ay,b,my,By)*squarebracket1+O(a,nx,Ax,b,mx,Bx)*squarebracket2
    #print(kinetic(a,lmn1,A,b,lmn2,B))
def T(a,b):
    '''Evaluates kinetic energy between two contracted Gaussians
       Returns float.
       Arguments:
       a: contracted Gaussian 'a', BasisFunction object
       b: contracted Gaussian 'b', BasisFunction object
    '''
    t = 0.0
    for i in range(len(a.coefs)):
        for j in range(len(b.coefs)):
            t += a.norm[i]*b.norm[j]*a.coefs[i]*b.coefs[j]*kinetic(a.exps[i],a.shell,a.origin,b.exps[j],b.shell,b.origin)
    return t
def R(t, u, n, p, PCx, PCy, PC,z):
    val = 0.0
    #print(u," ",t)
    if u == t == 0:
        if n == -1:
            val+=i1e(z)
        else:
            val+=ive(n,z)
    elif u == 0:
        if t > 1:
            val += -p * 0.5 * ((t-1)* (R(t - 2, u, n - 1, p, PCx, PCy, PC,z) + 2 * R(t - 2, u, n, p, PCx, PCy, PC,z) + R(t - 2, u, n + 1, p, PCx, PCy, PC,z)))
            val += -p*0.5*PCx * (R(t-1, u, n - 1, p, PCx, PCy, PC,z) + 2 * R(t-1, u, n, p, PCx, PCy, PC,z) + R(t-1, u, n + 1, p, PCx, PCy, PC,z))
    else:
        if u > 1:
            val += -p * 0.5 * ((u-1) * (R(t, u - 2, n - 1, p, PCx, PCy, PC,z) + 2 * R(t, u - 2, n, p, PCx, PCy, PC,z) + R(t, u-2, n + 1, p, PCx, PCy, PC,z)))
            val +=  -p*0.5*PCy * (R(t, u-1, n - 1, p, PCx, PCy, PC,z) + 2 * R(t, u-1, n, p, PCx, PCy, PC,z) + R(t, u-1, n + 1, p, PCx, PCy, PC,z))
    return val


#def gaussian_product_center(a, A, b, B):
    #return (a * A + b * B) / (a + b)

def nuclear_attraction(a, lmn1, A, b, lmn2, B, C):
    l1, m1 = lmn1
    l2, m2 = lmn2
    p = a + b
    Ax,Ay=A
    Bx,By=B
    Cx,Cy=C
    #P = gaussian_product_center(a, A, b, B) # Gaussian composite center
    Px=(a*Ax+b*Bx)/(a+b)
    Py=(a*Ay+b*By)/(a+b)
    PC=np.sqrt((Px-Cx)**2+(Py-Cy)**2)
    val = 0.0
    for t in range(l1 + l2+1):
        for u in range(m1 + m2+1):
            val += E(l1, l2, t, Ax - Bx, a, b) * E(m1, m2, u, Ay - By, a, b) * R(t, u, 0, p, Px - Cx, Py - Cy, PC, -p * PC * PC * 0.5)
    val *= np.pi * np.sqrt(np.pi / p)
    return val

def V(a,b,C):
    """this function evaluats the electron-nuclei interaction integral between two contracted gaussian
    a: is the first gaussian
    b: is the second gaussian
    C: is the origin of the nuclei"""
    N=0
    for i in range(len(a.coefs)):
        for j in range(len(b.coefs)):
            N=N+a.norm[i]*b.norm[j]*a.coefs[i]*b.coefs[j]*nuclear_attraction(a.exps[i],a.shell,a.origin,b.exps[j],b.shell,b.origin,C)
        #print(N)
    return N

def electron_repulsion(a,lmn1,A,b,lmn2,B,c,lmn3,C,d,lmn4,D):
    Ax,Ay=A
    Bx,By=B
    Cx,Cy=C
    Dx,Dy = D

    l1, m1 = lmn1
    l2, m2 = lmn2
    l3, m3 = lmn3
    l4, m4 = lmn4

    Px = (a*Ax+b*Bx)/(a+b)
    Qx = (c*Cx+d*Dx)/(c+d)
    Py = (a*Ay+b*By)/(a+b)
    Qy = (c*Cy+d*Dy)/(c+d)

    p = a + b
    q = c + d

    Delx = Qx - Px
    Dely = Qy - Py

    Delta = np.sqrt(Delx*Delx + Dely*Dely)
    sigma = (p+q)/(4*p*q)

    #val = 0
    temp1 = 0
    #temp2 = 0
    for t in range(l1+l2+1):
        for u in range(m1+m2+1):
            for tp in range(l3+l4+1):
                for up in range(m3+m4+1):
                    temp1+= ((np.power(np.pi, 2))/(p*q))*np.sqrt(np.pi/(4*sigma))*E(l1,l2,t,A[0]-B[0],a,b) * E(m1,m2,u,A[1]-B[1],a,b) * np.power(-1,t+u)*E(l3,l4,tp,C[0]-D[0],c,d) * E(m3,m4,up,C[1]-D[1],c,d) * R(t+tp,u+up,0,(1/(4*sigma)),Delx,Dely,Delta,(-1)*(Delta**2)/(8*sigma))


    return temp1

def U(a,b,c,d):
    N=0
    for i in range(len(a.coefs)):
        for j in range(len(b.coefs)):
            for k in range(len(c.coefs)):
                for l in range(len(d.coefs)):
                    N += a.norm[i]*b.norm[j]*a.coefs[i]*b.coefs[j]*c.norm[k]*d.norm[l]*c.coefs[k]*d.coefs[l]*electron_repulsion(a.exps[i],a.shell,a.origin,b.exps[j],b.shell,b.origin,c.exps[k],c.shell,c.origin,d.exps[l],d.shell,d.origin)
        #print(N)
    return N
