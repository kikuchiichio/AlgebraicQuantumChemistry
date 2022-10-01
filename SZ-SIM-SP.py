#!/usr/bin/env python
# coding: utf-8

# These Python programs compute the electronic structure of H3+ in the equilateral triangle. 
# 
# The atoms are indexed by A, B, and C, or a, b, and c.
# The STO-3G are determined by the parameters ZETA, COEFF, and EXPON. 
# 
# 
# # Computation of molecular integrals of p-shells
# 
# In this Python program, we try to compute the molecular integrals of p-shells.
# 
# ## Cautions against the false singularity of analytic formulas
# 
# In principle, we could get molecular integrals of p-shells analyticaly from the differentiation of those of s-shells. However, the practice does not go well as the theory suggests. When we compute the three- or four-center inegrals, it often happens that several centers are coincident. These situations might cause the zero in the denominator of the terms in the analyfic expression of three- or four-center inegrals, while this zero should theoretically be cancelled by another zero in the numerator.  We should prepare the analytic formulas for such special cases.
# 
# ##  Four-center integrals of p-shells
# 
# By trying lazy numerical substitutions to the the four-center integrals, we observe that the numerical divergense shall happen
# for four-centerintegrals of the type $[A,B|A,B]$. The cause of this divergence is what I have explained just now.
# The possible solution to this matter is to compute the integral of the form
# 
# $[A,B|A+T,B+T]$ 
# 
# with a slight shift $T$ of the coordinates in the last two centers, where  $T$ should be a symbol and led to the limit ($T\rightarrow 0$). 
# We would compute the $T$-series expansion of  $[A,B|A+T,B+T]$ by 
# 
# $C_0+ C_1  T + C_2  T^2+\cdots$  
# 
# and get the constant term C0, which is the correct value of the integral in question.
# 
# In pity, lurking numerical errors often give rise to the $T$-series expansion of the following form:
# 
# $\cdots + D_2 T^{-2} + D_1 T^{-1} + C_0+ C_1  T + C_2  T^2 + \cdots $,
# 
# which include the terms of negative exponent terms and negligibly small coefficients. Hence we should detect and prune the negative exponent terms.
# 
# 
# I show another example of similar phenomena. Imagine a square molecule:
# 
#         2--3
#         |  |
#         0--1
# 
# We place the equivalent $p_z$-orbitals on the sites 0, 1, 2, and 3, and find $[03|12]$ has the terms of the form 0/0. 
# If uncautiously we substitute the real values of the coorinates and the exponents in the analytic formula, we get an incorrect divergence, which runs toward 1.0e+46! This absurdity is caused by the geometry with $R_0+R_3=R_2+R_1$.
# 
# We should prepare the special formula for this case, too.
# 
# 
# ## Three-center integrals of p-shells
# 
# $\left(A\left|\frac{1}{|r-C|}\right|B\right)$ becomes falsely singular, when three centers  coincide at the same point.  The avoidance of the falise singularity is similar as in the case of four-center integrals.
# 
# ## Two-center integrals of p-shells
# 
# We do not find any false singularity.
# 
# ##  A toy-model of Hatree-Fock computation of a system composed of pz-orbital
# 
# This Python program enables you to do it, if you correctly prepare the molecular integrals.
# 
# 
# ## A design to compute the electron integrals with all possible type of shells.
# 
# Each shell is indicated by a key (i,j,k), namely, (0,0,0),(1,0,0),(0,1,0) and so on. A key shows the exponents of Cartesian Gaussians
# $
# (x-Ax)^i(y-Ay)^j(z-Az)\exp(-a|r-A|^2)
# $
# 
# Let io1,io2,io3,io4 be the orbitals  involved in the two-electron integral $[io1,io2|io3,io4]$.
# 
# 
# for io1 in ORBITALS:
#   for io2 in ORBITALS:
#      for io3 in ORBITALS:
#         for io4 in ORBITALS: 
#            key1=key(io1)
#            key2=key(io2)
#            key3=key(io3)
#            key4=key(io4)
#            
#            Inquire the formula of the integral [io1,io2|io3, io44] by (key1,key2,key3,key4)
#            If the integral has already been prepared, use it;
#            Otherwise, compute the formula, and register it in a dictionary.
#            
#            There are limiting cases [io1,io2|io1,io2] and [io1,io1|io1,io1].
#            They cause false singularities at the substitution of parameters.
#            Expect those cases; prepare the formulas for them, independently of the general cases [AB|CD].
#            
#            

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

#
#  SUNDRY DEFINITIONS OF WORKING VARIABLES.
#
import networkx as nx

IG=nx.DiGraph() 
from sympy import symbols, Function,expand,sqrt
import numpy as np
import copy
from itertools import combinations_with_replacement, permutations,product

S12,T11,T12,T22,V11A,V12A,V22A,V11B,V12B,V22B,V1111,V2111,V2121,V2211,V2221,V2222=[0 for _ in range(16)]

S=[[0 for _ in range(2)] for _ in range(2)]
X=[[0 for _ in range(2)] for _  in range(2)]
NM=[[0 for _ in range(2)] for _  in range(2)]
XT=[[0 for _ in range(2)] for _ in  range(2)]
H=[[0 for _ in range(2)] for _ in  range(2)]
F=[[0 for _ in range(2)] for _ in  range(2)]
P=[[0 for _ in range(2)] for _  in range(2)]
G=[[0 for _ in range(2)] for _ in  range(2)]
F2=[[0 for _ in range(2)] for _ in  range(2)]
P2=[[0 for _ in range(2)] for _  in range(2)]
G2=[[0 for _ in range(2)] for _ in  range(2)]

C=[[0 for _ in range(2)] for _ in  range(2)]
FPRIME=[[0 for _ in range(2)] for _  in range(2)]
CPRIME=[[0 for _ in range(2)] for _  in range(2)]
P=[[0 for _ in range(2)] for _  in range(2)]
ODLPP=[[0 for _ in range(2)] for _  in range(2)]
E=[[0 for _ in range(2)] for _  in range(2)]
TT=[[[[0 for _ in range(2)] for _  in range(2)] for _ in range(2)] for _ in range(2)]

IOP=2
N=3
ZETA1=2.0925
ZETA2=1.24
ZA=2.0
ZB=1.0
#
#  Atomic distance R
#
#R=symbols("R",positive=True)
#
#  Wavefunction (x,y) for alpha spin
#  Wavefunction (v,w) for beta spin
#
v, w = symbols("v w")
x, y = symbols("x y")


#
# In the following, we compute the atomic orbital integrals by the Gaussian orbital 
#  exp(-A*|r-RA|^2),exp(-A*|B-RB|^2),exp(-C*|r-RC|^2),exp(-D*|r-RD|^2)
#

import sympy as sympy
from sympy import symbols, Function,expand,core,sqrt
import numpy as np


def F0(ARG):
#
# BOYS F0 FUNCTIOM
#
    PI = sympy.pi
    if  type(ARG)==float and ARG < 1.0e-6:
        return 1 -ARG/3.
    if  type(ARG)==sympy.core.numbers.Zero and ARG < 1.0e-6:
        return 1 -ARG/3.
    else:
        #print("F0:ARG",ARG)
        if ARG!=sympy.S.Zero:
            return sympy.sqrt(PI/ARG)*sympy.erf(sqrt(ARG))/2
        else:
            return 1
    
def S000000(A,B,RA,RB):
#
#   OVERLAP INTEGRAL between exp(-A*|r-RA|^2) and exp(-B*|r-RB|^2)
#
    RAB2=sum([(c1-c2)**2 for c1,c2 in zip(RA,RB)])
    PI=np.pi
    print(A,B,RAB2,(PI/(A+B))**1.5)
    return (PI/(A+B))**1.5*sympy.exp(-A*B*RAB2/(A+B))

def getS(A,B,RAB2):
#
# Calculates the overlap matrix between exp(-A*|r-RA|^2) and exp(-A*|B-RB|^2)
# WHEN RAB2=|RA-RB|^2
#
    PI=np.pi
    print(A,B,RAB2,(PI/(A+B))**1.5)
    return (PI/(A+B))**1.5*sympy.exp(-A*B*RAB2/(A+B))

def K000000(A,B,RA,RB):
    RAB2=sum([(c1-c2)**2 for c1,c2 in zip(RA,RB)])
    PI=np.pi
    return A*B/(A+B)*(3.0-2.0*A*B*RAB2/(A+B))*(PI/(A+B))**1.5*sympy.exp(-A*B*RAB2/(A+B))   

def getT(A,B,RAB2):
#
# Calculates the kinetic energy
#  between exp(-A*|r-RA|^2) and exp(-B*|r-RB|^2)
# WHEN RAB2=|RA-RB|^2
#
    PI=np.pi
    return A*B/(A+B)*(3.0-2.0*A*B*RAB2/(A+B))*(PI/(A+B))**1.5*sympy.exp(-A*B*RAB2/(A+B))

def V(A,B,RAB2,RCP2,ZC):
#
#  CALCULATES UN-NORMALIZED NUCLEAR ATTRACTION INTEGRALS
#
    PI=np.pi
    V=2.0*PI/(A+B)*F0((A+B)*RCP2)*sympy.exp(-A*B*RAB2/(A+B))
    return -V*ZC

def TWOE(A,B,C,D,RAB2,RCD2,RPQ2):
#
# CALCULATES TWO-ELECTRON INTEGRALS FOR UN-NORMALIZED PRIMITIVES
# A,B,C,D ARE THE EXPONENTS ALPHA, BETA, ETC.
# RAB2 EQUALS SQUARED DISTANCE BETWEEN CENTER A AND CENTER B, ETC.
#
# The integrand is the product of those functions.
#   exp(-A*|r1-RA|^2),exp(-B*|r1-RB|^2),1/|r1-r2|, exp(-C*|r2-RC|^2),exp(-D*|r2-RD|^2)
# when RB=RA and RD=RC.
#
#
    PI=np.pi
    return 2.0*(PI**2.5)/((A+B)*(C+D)*np.sqrt(A+B+C+D))*F0((A+B)*(C+D)*RPQ2/(A+B+C+D))*sympy.exp(-A*B*RAB2/(A+B)-C*D*RCD2/(C+D))

    
def BOYS(N,X):
#
#  BOYS FUNCTION F_N(X)
#
    if N==0:
        return F0(X)
    else:
        Z=Symbol("Z")
        f=F0(Z)
        for _ in range(N):
            f=f.diff(Z)*(-1)
        return f.subs(Z,X)
    
def KXYZAB(a,b,RA,RB):
    mu=a*b/(a+b)
    RAB=[(c1-c2)**2 for c1,c2 in zip(RA,RB)]
    RAB2=sum(RAB)
    return sympy.exp(-mu*RAB2)

def T0000(a,b,c,d,RA,RB,RC,RD):
#
# CALCULATES TWO-ELECTRON INTEGRALS FOR UN-NORMALIZED PRIMITIVES
# a,b,c,d ARE THE EXPONENTS ALPHA, BETA, ETC.
# RAB2 EQUALS SQUARED DISTANCE BETWEEN CENTER A AND CENTER B, ETC.
#
# The integrand is the product of those functions.
#   exp(-a*|r1-RA|^2),exp(-b*|r1-RB|^2),1/|r1-r2|, exp(-c*|r2-RC|^2),exp(-d*|r2-RD|^2)
#
#
    def MF0(ARG):
    #
    # BOYS F0 FUNCTIOM
    #
        PI = sympy.pi
        if  type(ARG)==float and ARG < 1.0e-6:
            return 1 -ARG/3.
        if  type(ARG)==sympy.core.numbers.Zero and ARG < 1.0e-6:
            return 1 -ARG/3.
        else:
            #print("F0:ARG",ARG)
            if ARG!=sympy.S.Zero:
                return sympy.sqrt(PI/ARG)*sympy.erf(sqrt(ARG))/2
            else:
                return 1
    p=a+b
    q=c+d
    RP=[(a*c1+b*c2)/(a+b) for c1,c2 in zip(RA,RB)]
    RQ=[(c*c1+d*c2)/(c+d) for c1,c2 in zip(RC,RD)]
    alpha=p*q/(p+q)
    PI=sympy.pi
    RPQ=[(c1-c2)**2 for c1,c2 in zip(RP,RQ)]
    RPQ2=sum(RPQ)
    return 2*PI**(2.5)/p/q/sympy.sqrt(p+q)*KXYZAB(a,b,RA,RB)*KXYZAB(c,d,RC,RD)*MF0(alpha*RPQ2)

def V000000(a,b,RA,RB,RC):
#
# CALCULATES ONE-ELECTRON INTEGRALS FOR UN-NORMALIZED PRIMITIVES
#
#  The integrand is the product of those functions.
#   exp(-a*|r-RA|^2),exp(-b*|r-RB|^2),1/|r-RC|
#  
    p=a+b
    PI=sympy.pi
    RP=[(a*c1+b*c2)/(a+b) for c1,c2 in zip(RA,RB)]
    RPC=[(c1-c2)**2 for c1,c2 in zip(RP,RC)]
    RPC2=sum(RPC)
    return 2*PI/p*KXYZAB(a,b,RA,RB)*BOYS(0,p*RPC2)

def TN0000(N,a,b,c,d,RA,RB,RC,RD):
#
#  This function is the simplest case of the so-Called auxilary function THETA(N;000 000 000 000)
#  in Obara-Saika Schme for two-electron integrals.
#
    if N<0:
        return 0
    p=a+b
    q=c+d
    RP=[(a*c1+b*c2)/(a+b) for c1,c2 in zip(RA,RB)]
    RQ=[(c*c1+d*c2)/(c+d) for c1,c2 in zip(RC,RD)]
    alpha=p*q/(p+q)
    PI=sympy.pi
    RPQ=[(c1-c2)**2 for c1,c2 in zip(RP,RQ)]
    RPQ2=sum(RPQ)
    return 2*PI**(2.5)/p/q/sympy.sqrt(p+q)*KXYZAB(a,b,RA,RB)*KXYZAB(c,d,RC,RD)*BOYS(N,alpha*RPQ2)


#
#  TEST
#
AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
z1,z2,z3,z4=symbols("z1 z2 z3 z4")
f=T0000(z1,z2,z3,z4,[AX,AY,AZ],[BX,BY,BZ],[CX,CY,CZ],[DX,DY,DZ])
print(f)


#
# TEST
#
((S000000(z1,z2,[AX,AY,AZ],[BX,BY,BZ]).diff(AX)).diff(AX)).subs([(BX,AX),(BY,AY),(BZ,AZ)])


# #
# # PERFORMANCE TEST CONCERNING DIFFERENTIATION (TO GENERATE TWO-ELECTRON INTERGRALS OF P-ORBITALS)
# #
# #
# import time,random
# ALL=[]
# for va in [AX,AY,AZ]:
#     for vb in [BX,BY,BZ]:
#         for vc in [CX,CY,CZ]:
#             for vd in [DX,DY,DZ]:
#                 z1,z2,z3,z4=symbols("z1 z2 z3 z4")
#                 z1,z2,z3,z4=[random.random() for _ in range(NORBITALS)]
#                 #z1,z2,z3,z4=[z1,z2,random.random(),random.random()]
#                 f0=T0000(z1,z2,z3,z4,[AX,AY,AZ],[BX,BY,BZ],[CX,CY,CZ],[DX,DY,DZ])
#                 #print(f)
#                 #print(va,vb,vc,vd)
#                 start=time.time()
# 
#                 f1=f0.diff(va)
#                 
#                 f2=f1.diff(vb)
#                 f3=f2.diff(vc)
#                 f4=f3.diff(vd)
#                 
#                 end=time.time()
# 
#                 print("time=",end-start)
#                 ALL.append([f0,f1,f2,f3,f4])




from sympy import Function, Symbol
TN=Function("TN0000")
UNDEF=Function("UNDEF")

from sympy import Number
class FMA(Function):
    """
    FMA(x, y, z) = x*y + z
    """
    @classmethod
    def eval(cls, x, y, z):
        # Number is the base class of Integer, Rational, and Float
        if all(isinstance(i, Number) for i in [x, y, z]):
           return x*y + z
    def doit(self, deep=True, **hints):
        x, y, z = self.args
        # Recursively call doit() on the args whenever deep=True.
        # Be sure to pass deep=True and **hints through here.
        if deep:
            x = x.doit(deep=deep, **hints)
            y = y.doit(deep=deep, **hints)
            z = z.doit(deep=deep, **hints)
        return x*y + z
    
    


# # Obara-Saika Scheme for Two-electron integrals



#
# We prepare dummy expressions for THE FUNCTIONS TN0000(N,a,b,c,d,RA,RB,RC,RD): TNS[0],TNS[1],...., WHICH SHALL BE USED LATER
#  THEY ARE AUXILLARY INTEGRALS, AND IF N=0 , THEY ARE THE TWO-ELECTRON INTGRALS.
#

TNSTRING=''
for i in range(11):
    TNSTRING+= 'TN'+str(i)+' '
TNS=symbols(TNSTRING)
def TN(N):
    return TNS[N]



#
#  INITIALIZATION OF OBARA-SAIKA
#
#INTEGRALALL=dict()
#for i in range(11):
#    INTEGRALALL[(i,0,0,0,0,0,0,0,0,0,0,0,0)]=TN(i)
    
IG=nx.DiGraph() 
#
#  EACH NUCLEI IS STATIONED IN THE 3D CARTESIAN SPACE:r=(x,y,z)
#  THE COORDINATES ARE REPRESENTED AS RA=(XA,YA,ZB)
#  THE DIFFERENCE ARE REPRESENTED BY XPA=XP-XA, YPQ=YP-YQ, ZAB=ZA-AB, AND SO ON.
#
#  WE START FROM SPHERICAL GAUSSIAN FUNCTIONS : (a,RA)=exp(-a|r-RA|^2),exp(-b|r-RB|^2),exp(-c|r-RC|^2),exp(-d|r-rB|^2).
#
#
#
XPA, XPQ, XCD, XAB=symbols("XPA XPQ XCD XAB")
YPA, YPQ,YCD, YAB=symbols("YPA YPQ YCD YAB")
ZPA, ZPQ, ZCD, ZAB=symbols("ZPA ZPQ ZCD ZAB")
XQC, YQC, ZQC=symbols("XQC YQC ZQC")
#
#
#  [(a,RA;r1),(b,Rb;r1)| 1/|r1-r2| | (c,Rd);r2,(d,RD;r2)] 
#
#
#  We put p=a*b/(a+b);q=c*d/(c+d); alpha=p*q/(p+q).
#  Provisionally p,q are left as they are without substitution.
#  At the end of the computation, we have to replace them.
#
a,b,c,d=symbols("a b c d")
p,q=symbols("p q")
alpha=symbols("alpha")

#
#  We compute the integrals using general Cartesian Gaussian functions:
#   (x-XA)^i (y-YA)^j (z-ZA)^j exp(-a|r-RA|^2)
#
#  VIZ.
#   [(i1,j1,k1,a,RA),(i2,j2,k2,b,RB)|1/|r1-r2||(i3,j3,k3,d,RC),(i4,j4,k4,d,RD)]
#
#  For brevity, we write:
#   i=(i1,i2,i3), j=(j1,j2,j3), k=(k1,k2,k3),
# 
#  To compute the auxillary integdals T(N,i,j,k,l) we lift from T(N,i=0.j=0,k=0,z=0)
#  WHEN N=0 THEY GIVE THE TWO-ELECTRON INTEGRALS; OBARA-SAIKA RECURSION USES THEM WITH N>=1.
#
#  BY VERTICAL RECUSION IN OBARA-SAIKA, WE COMPUTE 
#  (N: i+1 0 0 0 ) from (N: i 0 0 0) , (N+1 i 0 0 0) , (N i-1 0 0 0), (N+1 i 0 0 0)
#
#  WHERE i+1 is the every possible move from (i1,i2,i3) by adding one or zero to each of (i1,i2,i3), 
#  except zero-move. 
#
#  THE FOLLOWING FUNCTION LIFTS UP IN D=X/Y/Z DIRECTIONS AND COMPUTES THE INTGRALS.

def VERTICAL(D,N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3,INFO=0):
    alpha=p*q/(p+q)
    
    if D=="X":
        assert (j1==j2==j3==k1==k2==k3==l1==l2==l3==0),print(D,N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3,"X")
        #print("X")
        key0=(N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        key1=(N+1,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        key2=(N,i1-1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        key3=(N+1,i1-1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        keyn=(N,i1+1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        i_d,k_d=i1,k1
        #print(keyn)
        DPA=XPA
        DPQ=XPQ
    if D=="Y":
        #print("Y")
        assert (j1==j2==j3==k1==k2==k3==l1==l2==l3==0),print(D,N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3,"X")
        key0=(N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        key1=(N+1,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        key2=(N,i1,i2-1,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        key3=(N+1,i1,i2-1,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        keyn=(N,i1,i2+1,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        i_d,k_d=i2,k2
        #print(keyn)
        DPA=YPA
        DPQ=YPQ
    if D=="Z":
        assert (j1==j2==j3==k1==k2==k3==l1==l2==l3==0),print(D,N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3,"X")
        #print("Z")
        key0=(N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        key1=(N+1,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        key2=(N,i1,i2,i3-1,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        key3=(N+1,i1,i2,i3-1,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        keyn=(N,i1,i2,i3+1,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        i_d,k_d=i3,k3
        #print(keyn)
        DPA=ZPA
        DPQ=ZPQ
    if None!=integrals_all.get(keyn):
        return
    
    data0=integrals_all.get(key0)
    data1=integrals_all.get(key1)
    data2=integrals_all.get(key2)
    data3=integrals_all.get(key3)
    #print(key0,key1,key2,key3)
    #print(data0,data1,data2,data3)
    nx.add_star(IG, [keyn,key0, key1, key2, key3])
    if min(list(key2))<0:
        data2=0
        integrals_all[key2]=0
    if min(list(key3))<0:
        data3=0
        integrals_all[key3]=0
    if data0==None:
        #print("data 0 requested")
        requested_keys[key0]=key0
    if data1==None:
        #print("data 1 requested")
        requested_keys[key1]=key1
    if data2==None:
        #print("data 2 requested")
        requested_keys[key2]=key2
    if data3==None:
        #print("data 3 requested")
        requested_keys[key3]=key3
    #print(requested_keys.keys())
    if data0!=None and data1!=None and data2!=None and data3!=None:
        if (INFO==1):
            print(keyn,"WRITTEN")
        integrals_all[keyn]= DPA*data0-alpha/p*DPQ*data1+i_d/2/p*(data2-alpha/p*data3)
    else:
        if (INFO==1):
            print(keyn,"NOT WRITTEN")
            print(key0,key1,key2,key3)
            print(data0,data1,data2,data3)
        else:
            pass
    return
    
#
#  The indices of the integrals which are required, but has not yet been computed.
#
requested_keys=dict()
        
#
#  The dictionary of computed integrals
#
integrals_all=dict()
for i in range(11):
    integrals_all[(i,0,0,0,0,0,0,0,0,0,0,0,0)]=TN(i)        


# In[ ]:


ISAB=S000000(z1,z2,[AX,AY,AZ],[BX,BX,BZ])
IKAB=K000000(z1,z2,[AX,AY,AZ],[BX,BY,BZ])
ISABZZ=ISAB.diff(AZ).diff(BZ)/z1/z2/4
IKABZZ=IKAB.diff(AZ).diff(BZ)/z1/z2/4


# In[ ]:


IKABZZ.subs(AZ,0).subs(BZ,0).subs(z1,1).subs(z2,1)


# In[ ]:


AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
z1,z2,z3,z4=symbols("z1 z2 z3 z4")
IVABC=V000000(z1,z2,[AX,AY,AZ],[BX,BY,BZ],[CX,CY,CZ])
IVABCZZ=IVABC.diff(AZ).diff(BZ)/z1/z2/4


# In[ ]:


ISAB


# In[ ]:


z1,z2,z3,z4=symbols("z1 z2 z3 z4")
T=symbols("T",positive=True)
AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
fssss=T0000(z1,z2,z3,z4,[AX,AY,AZ],[BX,BY,BZ],[CX,CY,CZ],[DX,DY,DZ])
fssssABAB=fssss.subs([(z3,z1),(z4,z2),(CX,AX),(CY,AY),(CZ,AZ),(DX,BX),(DY,BY),(DZ,BZ+T)])
fssssAAAA=fssssABAB.subs([(z2,z1),(BX,AX),(BY,AY),(BZ,AZ+T)])
fzzzz=fssss.diff(AZ).diff(BZ).diff(CZ).diff(DZ)
fzzzzABAB=fzzzz.subs([(z3,z1),(z4,z2),(CX,AX),(CY,AY),(CZ,AZ),(DX,BX),(DY,BY),(DZ,BZ+T)])
fzzzzAAAA=fzzzzABAB.subs([(z2,z1),(BX,AX),(BY,AY),(BZ,AZ+T)])
with open('fzzzAAAA.txt', 'w') as f:
    f.write(str(fzzzzAAAA))
with open('fzzzABAB.txt', 'w') as f:
    f.write(str(fzzzzABAB))


# In[ ]:


#
print("We have the ingtegral formula of [AB|CD] of pz orbitals, placed at centers A, B, C, and D.")
#
z1,z2,z3,z4=symbols("z1 z2 z3 z4")
T=symbols("T",positive=True)
AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")

print("Four centers coincident at the origin:")
print(fzzzz.subs(z1,1).subs(z2,1).subs(z3,1).subs(z4,1).subs(AX,0).subs(AY,0).subs(AZ,0).subs(BX,0).subs(BY,0).subs(BZ,0).\
subs(CX,0).subs(CY,0).subs(CZ,0).subs(DX,0).subs(DY,0).subs(DZ,0))


print("[AB|AB]")
print(fzzzz.subs(z1,1).subs(z2,1).subs(z3,1).subs(z4,1).subs(AX,0).subs(AY,0).subs(AZ,0).subs(BX,1).subs(BY,0).subs(BZ,0).\
subs(CX,0).subs(CY,0).subs(CZ,0).subs(DX,1).subs(DY,0).subs(DZ,0))


print("[AB|CD]")
print(fzzzz.subs(z1,1).subs(z2,1).subs(z3,1).subs(z4,1).subs(AX,0).subs(AY,0).subs(AZ,0).subs(BX,1).subs(BY,0).subs(BZ,0).\
subs(CX,1).subs(CY,1).subs(CZ,0).subs(DX,1).subs(DY,1).subs(DZ,1))

print("We find the false singularity of the integrals in the first two cases!")


# In[ ]:


#
print("We elicit the false singularity of [AB|AB]. In the following we compute [AB|A,B+T] and expand it as the polynomial of T.")
print("With T->0, it should remain finite, and we have well done.")

#
T=symbols("T",positive=True)
AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")

fT=fssss.subs(z1,1).subs(z2,1).subs(z3,1).subs(z4,1).subs(AX,0).subs(AY,0).subs(AZ,0).subs(BX,0).subs(BY,0).subs(BZ,0).\
subs(CX,0).subs(CY,0).subs(CZ,0).subs(DX,0).subs(DY,0).subs(DZ,T)
sympy.series(fT,T)


# In[ ]:


#
print("We elicit the false singularity of [AB|AB]. In the following we compute [AB|A,B+T] and expand it as the polynomial of T.")
print("With T->0, it should remain finite, and we have well done.")

#
#T=symbols("T",positive=True)
#AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")

#fT=fssss.subs(z1,1.).subs(z2,1.).subs(z3,1.).subs(z4,1.).subs(AX,0.5).subs(AY,0.5).subs(AZ,0.5).subs(BX,0.5).subs(BY,0.5).subs(BZ,0.5).\
#subs(CX,0.5).subs(CY,0.5).subs(CZ,0.5).subs(DX,0.5).subs(DY,0.5).subs(DZ,T)
#sympy.series(fT,T)


# In[ ]:


#
print("We elicit the false singularity of [AB|AB]. In the following we compute [AB|A,B+T] and expand it as the polynomial of T.")
print("With T->0, it should remain finite, and we have well done.")

#
fT=fzzzz.subs(z1,1).subs(z2,1).subs(z3,1).subs(z4,1).subs(AX,0).subs(AY,0).subs(AZ,0).subs(BX,1).subs(BY,0).subs(BZ,0).\
subs(CX,0).subs(CY,0).subs(CZ,0).subs(DX,1).subs(DY,0).subs(DZ,T)
sympy.series(fT,T)


# In[ ]:


#
print("We elicit the false singularity of [AB|AB]. In the following we compute [AB|A,B+T] and expand it as the polynomial of T.")
print("With T->0, it diverges. This is because of the numerical errors. The singularity should be removed.")

#
fT=fzzzz.subs(z1,1.).subs(z2,1.).subs(z3,1.).subs(z4,1).subs(AX,0).subs(AY,0).subs(AZ,0).subs(BX,1).subs(BY,0).subs(BZ,0).\
subs(CX,0).subs(CY,0).subs(CZ,0).subs(DX,1).subs(DY,0).subs(DZ,T)
sympy.series(fT,T)


# In[ ]:


import numpy

TTlist=dict()
Integralslist=dict()
IntegralslistABAB=dict()
IntegralslistAAAA=dict()

def CheckLimit(PP):
    tiny=1.0e-5
    def DropCoeff(x):
    #
    #  ROWND DOWN x*N AFTER THE DECIMAL POINT and get AN INTEGER M
    #  THEN X is approximated by M/N. 
    #
    
        if np.abs(x) <tiny:
            return 0
        return x

    def getPowerProduct(args,tm):
        V=1
        for w,p in zip(list(args[1:]),tm):
            V*= w**p
        return V
    
    ENTS3=sympy.poly(PP)
    AT=ENTS3.terms()
    #print(AT)
    getF=0
    for tm in AT:
        #p0,p1,p2,p3,p4,p5,p6,p7=tm[0]
        cf=tm[1]
        #print(p0,p1,p2,cf)
        #getF+=x**p0* y**p1* z**p2* u**p3* v**p4* w**p5* e**p5* f**p7*getINT(cf,10000)
        getF+=getPowerProduct(ENTS3.args,tm[0])*DropCoeff(cf)
    return getF

def indexijkl(i,j,k,l):
    A=[(i,j,k,l),(j,i,k,l),(i,j,l,k),(j,i,l,k)]
    A+=[(k,l,i,j),(l,k,i,j),(k,l,j,i),(l,k,j,i)]
    A.sort()
    return A[0]


R_length=0.9/0.529177
#R=symbols("R")
#R_length=R
R=R_length
atomA=[0,0,0]
atomB=[R_length,0,0]
atomC=[0,R_length,0]
atomD=[R_length,R_length,0]

#atomA=[0,0,1]
#atomB=[R_length,0,1]
#atomC=[2*R_length,0,1]
#atomD=[3*R_length,0,1]
atomE=np.array([0,0,0])
atomA=np.array([1,1,1])*0.5
atomB=np.array([1,-1,-1])*0.5
atomC=np.array([-1,1,-1])*0.5
atomD=np.array([-1,-1,1])*0.5
ZETA1=2.0925
ZETA2=1.24
ZETA1=ZETA2
ZETA3=ZETA2
ZETA4=ZETA3
ZETA5=ZETA4
#
#
#
ZA=2.0
ZB=1.0
ZA=ZB
ZC=ZB
ZD=ZC
ZE=ZD
atoms=[atomA,atomB,atomC,atomD,atomE]
ZETAS=[ZETA1,ZETA2,ZETA3,ZETA4,ZETA5]
ORBITKEYS=[(0,0,0),(0,0,0),(0,0,0),(0,0,0),(1,0,0)]
ZS=[ZA,ZB,ZC,ZD,ZE]
NORBITALS=5
COEF=[[1.0,0.0,0.0],[0.678914,0.430129,0.0],[0.444635,0.535328,0.154329]]
EXPON=[[0.270950,0.0,0.0],[0.151623,0.851819,0.0],[0.109818,0.405771,2.22766]]
DA=[0]*3
CA=[0]*3
DB=[0]*3
CB=[0]*3
DC=[0]*3
CC=[0]*3
DD=[0]*3
CD=[0]*3
PI=numpy.pi

TTlist=dict()

Integralslist[(0,0,0,0,0,0,0,0,0,0,0,0)]=fssss
def GetTwoEIntegral(key1,key2,key3,key4):
    AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
    z1,z2,z3,z4=symbols("z1 z2 z3 z4")
    fml=fssss
    keys1234=key1+key2+key3+key4
    if Integralslist.get(keys1234)!=None:
        return Integralslist.get(keys1234)
    for ifl,var in zip(list(keys1234),[AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ]):
        if ifl==1:
            fml=fml.diff(var)
    Integralslist[keys1234]=fml
    return fml
        
"""
fssssABAB=fssss.subs([(z3,z1),(z4,z2),(CX,AX),(CY,AY),(CZ,AZ),(DX,BX),(DY,BY),(DZ,BZ+T)])
fssssAAAA=fssssABAB.subs([(z2,z1),(BX,AX),(BY,AY),(BZ,AZ+T)])
"""    
def GetTwoEIntegralABAB(key1,key2,key3,key4):
    AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
    z1,z2,z3,z4=symbols("z1 z2 z3 z4")
    T=symbols("T",positive=True)
    keys1234=key1+key2+key3+key4
    if IntegralslistABAB.get(keys1234)!=None:
        return IntegralslistABAB.get(keys1234)
    f=GetTwoEIntegral(key1,key2,key3,key4)
    fABAB=f.subs([(z3,z1),(z4,z2),(CX,AX),(CY,AY),(CZ,AZ),(DX,BX),(DY,BY),(DZ,BZ+T)])
    IntegralslistABAB[keys1234]=fABAB
    return fABAB

def GetTwoEIntegralAAAA(key1,key2,key3,key4):
    AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
    z1,z2,z3,z4=symbols("z1 z2 z3 z4")
    T=symbols("T",positive=True)
    keys1234=key1+key2+key3+key4
    if IntegralslistAAAA.get(keys1234)!=None:
        return IntegralslistAAAA.get(keys1234)
    fABAB=GetTwoEIntegralABAB(key1,key2,key3,key4)
    fAAAA=fABAB.subs([(z2,z1),(BX,AX),(BY,AY),(BZ,AZ+T)])
    IntegralslistAAAA[keys1234]=fAAAA
    return fAAAA

GetTwoEIntegralAAAA((0,0,0),(0,0,0),(0,0,0),(0,0,0))


# In[ ]:


fssss


# In[ ]:


def TTZZZZ(ca,cb,cc,cd,RA,RB,RC,RD):
    AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
    z1,z2,z3,z4=symbols("z1 z2 z3 z4")
    T=symbols("T", positive=True)
    #print(RA,RB,RC,RD)
    #print(ca,cb,cc,cd)
    tiny=0
    #V=fzzzz.subs([(z1,ca),(z2,cb+tiny),(z3,cc-tiny),(z4,cd+2*tiny)])
 
    V=fzzzz
    V=V.subs([(z1,ca),(z2,cb),(z3,cc),(z4,cd)])
    V=V.subs([(AX,RA[0]),(AY,RA[1]),(AZ,RA[2])])
    #print(V)
    V=V.subs([(BX,RB[0]),(BY,RB[1]),(BZ,RB[2])])
    V=V.subs([(CX,RC[0]),(CY,RC[1]),(CZ,RC[2])])
    V=V.subs([(DX,RD[0]),(DY,RD[1])])
    VO=V

    #V=V.subs([(DZ, RD[2])])
    V=sympy.limit(V,DZ,RD[2])
    #return V
    if V==sympy.oo or V==-sympy.oo or np.abs(sympy.N(V))>1.0e10 :
        print(RA,RB,RC,RD)
        #print(VO,V)
        VS=sympy.series(VO,DZ,RD[2]).removeO()
        V=CheckLimit(VS)
        print(V)
        V=V.subs(DZ,RD[2])
        print(V)
    return V

def TTZZZZE(expr,ca,cb,cc,cd,RA,RB,RC,RD,IOP=0):
    AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
    z1,z2,z3,z4=symbols("z1 z2 z3 z4")
    T=symbols("T",positive=True)
    print(RA,RB,RC,RD)
    print(ca,cb,cc,cd)
    tiny=0
    #V=fzzzz.subs([(z1,ca),(z2,cb+tiny),(z3,cc-tiny),(z4,cd+2*tiny)])
    symbolslist=[z1,z2,z3,z4,AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ]
    valueslist=[ca,cb,cc,cd]
    for RS in [RA,RB,RC,RD]:
        valueslist.extend(RS)
 
    V=expr
    index=0
    F=[]
    for s,v in zip(symbolslist,valueslist):
        #print(s,v)
        VO=V
        V=V.subs(s,v)
        if V==sympy.nan and IOP==1:
            V=VO
            print("Fail in substitution...")
            F.append((s,v))
        index+=1
    return V
        
    V=V.subs([(z1,ca),(z2,cb),(z3,cc),(z4,cd)])
    V=V.subs([(AX,RA[0]),(AY,RA[1]),(AZ,RA[2])])
    #print(V)
    V=V.subs([(BX,RB[0]),(BY,RB[1]),(BZ,RB[2])])
    V=V.subs([(CX,RC[0]),(CY,RC[1]),(CZ,RC[2])])
    #print(V)
    V=V.subs([(DX,RD[0])])
    #print(V)
    V=V.subs([(DY,RD[1])])
    VO=V
    #print(V)
    V=V.subs([(DZ, RD[2])])
    #V=sympy.limit(V,DZ,RD[2])
    return V
    if V==sympy.oo or V==-sympy.oo or np.abs(sympy.N(V))>1.0e10 :
        print(RA,RB,RC,RD)
        #print(VO,V)
        VS=sympy.series(VO,DZ,RD[2]).removeO()
        V=CheckLimit(VS)
        print(V)
        V=V.subs(DZ,RD[2])
        print(V)
    return V

print('\
\n#  We compute two-center integrals of pz orbitals.\
\n#  \
\n#  The integral [AB|AB] becomes falsely simgular.\
\n#  So we compute [AB|A+T, B+T] (with the shift of the coordinates of the last two centers) to get it as the polynomial of T.\
\n#  After this loop, we elicit the correct value of that type of the integrals.\
\n#  \
')
AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
z1,z2,z3,z4=symbols("z1 z2 z3 z4")
T=symbols("T", positive=True)
TTlist=dict()
Flist=dict()
TT=[[[[0 for _ in range(3)] for _  in range(3)] for _ in range(3)] for _ in range(3)]
for ID in range(NORBITALS):
    for JD in range(NORBITALS):
        for KD in range(NORBITALS):
            for LD in range(NORBITALS):
                RA=[AX,AY,AZ]=atoms[ID]
                RB=[BX,BY,BZ]=atoms[JD]
                RC=[CX,CY,CZ]=atoms[KD]
                RD=[DX,DY,DZ]=atoms[LD]
                key1=ORBITKEYS[ID]
                key2=ORBITKEYS[JD]
                key3=ORBITKEYS[KD]
                key4=ORBITKEYS[LD]
                print(key1,key2,key3,key4)
                if ID!=KD or JD!=LD:
                    formula=GetTwoEIntegral(key1,key2,key3,key4)
                if ID==KD and JD==LD and ID!=JD:
                    formula=GetTwoEIntegralABAB(key1,key2,key3,key4)       
                if ID==KD and JD==LD and ID==JD:
                    formula=GetTwoEIntegralAAAA(key1,key2,key3,key4)   
                Flist[(ID,JD,KD,LD)]=formula
                
                if TTlist.get(indexijkl(ID,JD,KD,LD))==None:
                    #print(ID,JD,KD,LD)
                    N=1
                    for i in range(N):
                        #print(i,EXPON[N-1][i],COEF[N-1][i],ZETA1,ZETA2)
                        CA[i]=EXPON[N-1][i]*ZETAS[ID]**2
                        DA[i]=COEF[N-1][i]*((2.0*CA[i]/PI)**0.75)
                        CB[i]=EXPON[N-1][i]*(ZETAS[JD]**2)
                        DB[i]=COEF[N-1][i]*((2.0*CB[i]/PI)**0.75)
                        CC[i]=EXPON[N-1][i]*(ZETAS[KD]**2)
                        DC[i]=COEF[N-1][i]*((2.0*CC[i]/PI)**0.75)
                        CD[i]=EXPON[N-1][i]*(ZETAS[LD]**2)
                        DD[i]=COEF[N-1][i]*((2.0*CD[i]/PI)**0.75)

                    N=1
                    V=0
                    VW=0
                    for I in range(N):            
                        for J in range(N):
                            for K in range(N):
                                for L in range(N):
                                    ca=CA[I]
                                    cb=CB[J]
                                    cc=CC[K]
                                    cd=CD[L]
                                    #print(ca,cb,cc,cd,sympy.N(T0000(ca,cb,cc,cd,RA,RB,RC,RD)))

                                    #V=V+T0000(ca,cb,cc,cd,RA,RB,RC,RD)*DA[I]*DB[J]*DC[K]*DD[L]
                                    if ID!=KD or JD!=LD:
                                        print("1")
                                        #VW=VW+sympy.N(T0000(ca,cb,cc,cd,RA,RB,RC,RD))*DA[I]*DB[J]*DC[K]*DD[L]/ca/cb/cc/cd/16
                                        V=V+TTZZZZE(formula,ca,cb,cc,cd,RA,RB,RC,RD)*DA[I]*DB[J]*DC[K]*DD[L]/ca/cb/cc/cd/16
                                    #if [ID,JD,KD,LD]==[0,3,1,2]:
                                    #print(V)
                                    else:
                                        print("2",formula)
                                        #Provisionally compute [A,B|A+T,B+T] with the symbol T.
                                        #VW=VW+T0000(ca,cb,cc,cd,RA,RB,RC,RD)*DA[I]*DB[J]*DC[K]*DD[L]/ca/cb/cc/cd/16
                                        #print( TTZZZZE(formula,ca,cb,cc,cd,RA,RB,RC,RD))

                                        V=V+TTZZZZE(formula,ca,cb,cc,cd,RA,RB,RC,RD)*DA[I]*DB[J]*DC[K]*DD[L]/ca/cb/cc/cd/16
                    
                    #if [ID,JD,KD,LD]==[0,3,1,2]:
                        #print(sympy.N(T0000(ca,cb,cc,cd,atoms[ID],atoms[JD],atoms[KD],atoms[LD]))/ca/cb/cc/cd/8)

                    V=sympy.N(V)
                    VW=sympy.N(VW)
                    print(RA,RB,RC,RD,V,VW)
                    TTlist[indexijkl(ID,JD,KD,LD)]=V


# In[ ]:


GetTwoEIntegralAAAA((0,0,0),(0,0,0),(0,0,0),(0,0,0)).subs(z1,ca)


# In[ ]:


import copy
TTlist_save=copy.deepcopy(TTlist)

def Pruning(expr):
#
#
#   To get the constant term from a power series which include negative exponent terms.
#   They might be wrong expressons when the coefficients of negative terms are very small. 
#
#
    def getCoeff(x):
        tiny=1.0e-10
        if numpy.abs(x)<tiny:
            return 0
        else:
            return x

    def getPowerProduct(args,tm):
        V=1
        for w,p in zip(list(args[1:]),tm):
            V*= w**p
        return V
    ENTS3=sympy.poly(expr)
    AT=ENTS3.terms()
    getF=0
    for tm in AT:
        #p0,p1,p2,p3,p4,p5,p6,p7=tm[0]
        cf=tm[1]
        if 0==max(list(tm[0])):
            getF+=getPowerProduct(ENTS3.args,tm[0])*getCoeff(cf)
    return getF


#
#   Some integrals remain as formulas, because the uncautious numerical substitutions 
#   shall cause the false simgularity in them.
#   We elicit and remove the false singular part from those integrals to evaluate them properly.
#
for vk in TTlist.keys():
    v=TTlist.get(vk)
    if v.is_Number==False:
        w=sympy.series(v,T,0)
        w=sympy.N(w)
        w=w.removeO()
        print("\n",vk,w)
        Pw=Pruning(w)
        print("-->",Pw)
        TTlist[vk]=Pw


# In[ ]:


TTlist


# In[ ]:


Flist[(2,3,2,3)].subst(z3,1).


# In[ ]:


#
print("We still found another absurde divergence! at([03|12])")
#
print(TTlist)
for vk in TTlist.keys():
    w=TTlist.get(vk)
    if np.abs(w) > 1.e+30:
        print("\n",vk,w,"\n")


# In[ ]:


print(ca,cb,cc,cd)
AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
z1,z2,z3,z4=symbols("z1 z2 z3 z4")
R=symbols("R")

#sympy.N(T0000(ca,cb,cc,cd,atoms[0],atoms[3],atoms[1],atoms[2]))
PC=[atoms[2][0],atoms[2][1],atoms[2][2]]
print(sympy.N(TTZZZZE(fzzzz,ca,cb,cc,cd,atoms[0],atoms[3],atoms[1],PC)))
print(sympy.N(TTZZZZE(fzzzz,z1,z1,z1,z2,atoms[0],atoms[3],atoms[1],PC)))
print(sympy.N(TTZZZZE(fzzzz,z1,z1,z1,z1+T,[0,0,0],[R,R,0],[R,0,0],[0,R,0])))
print(sympy.N(TTZZZZE(fzzzz,ca*0.999,cb,cc,cd,atoms[0],atoms[3],atoms[1],atoms[2])))


# In[ ]:


print("The standing points of orbitals of [03|12]:")
print([atoms[0],atoms[3],atoms[1],atoms[2]])


# In[ ]:


print("Prepare a special formula for [03|12]")
AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
z1,z2,z3,z4=symbols("z1 z2 z3 z4")
R=symbols("R")
spform=sympy.series(TTZZZZE(fzzzz,z1,z1,z1,z1,[0,0,0],[BX,BX,0],[BX,0,0],[0,BX,T]),T,0).subs(T,0)
spform


# In[ ]:


print("Replace the anomalous integral.")
ID,JD,KD,LD=(0,3,1,2)
N=1
for i in range(N):
    #print(i,EXPON[N-1][i],COEF[N-1][i],ZETA1,ZETA2)
    CA[i]=EXPON[N-1][i]*ZETAS[ID]**2
    DA[i]=COEF[N-1][i]*((2.0*CA[i]/PI)**0.75)
    CB[i]=EXPON[N-1][i]*(ZETAS[JD]**2)
    DB[i]=COEF[N-1][i]*((2.0*CB[i]/PI)**0.75)
    CC[i]=EXPON[N-1][i]*(ZETAS[KD]**2)
    DC[i]=COEF[N-1][i]*((2.0*CC[i]/PI)**0.75)
    CD[i]=EXPON[N-1][i]*(ZETAS[LD]**2)
    DD[i]=COEF[N-1][i]*((2.0*CD[i]/PI)**0.75)
V=0
for I in range(N):            
    for J in range(N):
        for K in range(N):
            for L in range(N):
                ca=CA[I]
                cb=CB[J]
                cc=CC[K]
                cd=CD[L]
                if ca==cb==cc==cd:
                    V=V+TTZZZZE(spform,ca,cb,cc,cd,RA,RB,RC,RD)*DA[I]*DB[J]*DC[K]*DD[L]/ca/cb/cc/cd/8
V=sympy.N(V)
print(RA,RB,RC,RD,V)
TTlist[indexijkl(ID,JD,KD,LD)]=V


# In[ ]:


print(TTlist)
for vk in TTlist.keys():
    w=TTlist.get(vk)
    if np.abs(w) > 1.e+30:
        print("\n",vk,w,"\n")


# In[ ]:


#
print("Check the three center integral when three centers coincide. Does the substitution give back the proper numerical value?")
#
AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ") 
expr=IVABCZZ.subs(BX,AX).subs(BY,AY).subs(BZ,AZ).subs(CX,AX).subs(CY,AY).subs(CZ,AZ)
expr.subs(z1,1).subs(z2,2).subs(AX,1).subs(AY,0).subs(AZ,1)


# In[ ]:


#
# Prepare the three-cencer integral (A|1/|r-C||B) for the case A,B,C coinside at a point and the orbitals A and B are equivalent. 
#
AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ") 
z1,z2=symbols("z1 z2")
T=symbols("T", positive=True)
IVz1z1ABC=V000000(z1,z1,[AX,AY,AZ],[BX,BY,BZ],[CX,CY,CZ])
IVz1z1ABCZZ=IVz1z1ABC.diff(AZ).diff(BZ)/z1/z1/4
expr=(IVz1z1ABC.expand().diff(AZ).diff(BZ)/z1/z1/4).subs(BX,AX).subs(BY,AY).subs(BZ,AZ).subs(CX,AX).subs(CY,AY).subs(CZ,AZ+T)
exprIVAAAZZ=sympy.series(expr.expand(),T,0).removeO().subs(T,0)
exprIVAAAZZ


# In[ ]:


"""
            RAP=A2[J]*R/(A1[I]+A2[J])
            RAP2=RAP**2
            RBP2=(R-RAP)**2
            S12+=getS(A1[I],A2[J],R2)*D1[I]*D2[J]
            T11=T11+getT(A1[I],A1[J],0.0)*D1[I]*D1[J]
            #print(I,J,getT(A1[I],A1[J],0.0),D1[I],D1[J],T11)
            T12=T12+getT(A1[I],A2[J],R2)*D1[I]*D2[J]
            T22=T22+getT(A2[I],A2[J],0.0)*D2[I]*D2[J]
            V11A=V11A+V(A1[I],A1[J],0.0,0.0,ZA)*D1[I]*D1[J]
            V12A=V12A+V(A1[I],A2[J],R2,RAP2,ZA)*D1[I]*D2[J]
            V22A=V22A+V(A2[I],A2[J],0.0,R2,ZA)*D2[I]*D2[J]
            V11B=V11B+V(A1[I],A1[J],0.0,R2,ZB)*D1[I]*D1[J]
            V12B=V12B+V(A1[I],A2[J],R2,RBP2,ZB)*D1[I]*D2[J]
            V22B=V22B+V(A2[I],A2[J],0.0,0.0,ZB)*D2[I]*D2[J]
"""
def S000000(A,B,RA,RB):
#
#   OVERLAP INTEGRAL between exp(-A*|r-RA|^2) and exp(-B*|r-RB|^2)
#
    RAB2=sum([(c1-c2)**2 for c1,c2 in zip(RA,RB)])
    PI=np.pi
    #print(A,B,RA,RB,RAB2,(PI/(A+B))**1.5)
    return (PI/(A+B))**1.5*sympy.exp(-A*B*RAB2/(A+B))
def GetRP(a,b,RA,RB):
    AX,AY,AZ=RA
    BX,BY,BZ=RB
    PX=(a*AX+b*BX)/(a+b)
    PY=(a*AY+b*BY)/(a+b)
    PZ=(a*AZ+b*BZ)/(a+b)
    return [PX,PY,PZ]
AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ") 
z1,z2=symbols("z1 z2")

ISAB=S000000(z1,z2,[AX,AY,AZ],[BX,BY,BZ])
IKAB=K000000(z1,z2,[AX,AY,AZ],[BX,BY,BZ])
ISABZZ=ISAB.diff(AZ).diff(BZ)/z1/z2/4
IKABZZ=IKAB.diff(AZ).diff(BZ)/z1/z2/4

VAB_C=[[[0 for _ in range(NORBITALS)] for _  in range(NORBITALS)] for _ in range(NORBITALS)]
for ID in range(NORBITALS):
    for JD in range(NORBITALS):
        for KD in range(NORBITALS):
                RA=[AX,AY,AZ]=atoms[ID]
                RB=[BX,BY,BZ]=atoms[JD]
                RC=[CX,CY,CZ]=atoms[KD]
                ZC=ZS[KD]
                N=1
                for i in range(N):
                    #print(i,EXPON[N-1][i],COEF[N-1][i],ZETA1,ZETA2)
                    CA[i]=EXPON[N-1][i]*ZETAS[ID]**2
                    DA[i]=COEF[N-1][i]*((2.0*CA[i]/PI)**0.75)
                    CB[i]=EXPON[N-1][i]*(ZETAS[JD]**2)
                    DB[i]=COEF[N-1][i]*((2.0*CB[i]/PI)**0.75)
                    CC[i]=EXPON[N-1][i]*(ZETAS[KD]**2)
                    DC[i]=COEF[N-1][i]*((2.0*CC[i]/PI)**0.75)
                    CD[i]=EXPON[N-1][i]*(ZETAS[LD]**2)
                    DD[i]=COEF[N-1][i]*((2.0*CD[i]/PI)**0.75)
                    
                N=1
                V=0
                for I in range(N):            
                    for J in range(N):
                            ca=CA[I]
                            cb=CB[J]
                            #V000000(a,b,RA,RB,RC)
                            V=V+V000000(ca,cb,RA,RB,RC)*DA[I]*DB[J]*(-ZC)
                            #V+=TTZZZZE(IVABCZZ,ca,cb,cc,cd,RA,RB,RC,RD)*DA[I]*DB[J]*(-ZC)
                if ID==JD==KD and ca==cb:
                    V=0
                    for I in range(N):            
                        for J in range(N):
                            ca=CA[I]
                            cb=CB[J]
                            #V000000(a,b,RA,RB,RC)
                            V=V+V000000(ca,cb,RA,RB,RC)*DA[I]*DB[J]*(-ZC)
                            #V+=TTZZZZE(exprIVAAAZZ,ca,cb,cc,cd,RA,RB,RC,RD)*DA[I]*DB[J]*(-ZC)
                    
                V=sympy.N(V)
                #print(RA,RB,RC,RD,V)
                VAB_C[ID][JD][KD]=V

SAB=[[0 for _ in range(NORBITALS)] for _  in range(NORBITALS)]              
for ID in range(NORBITALS):
    for JD in range(NORBITALS):
        RA=[AX,AY,AZ]=atoms[ID]
        RB=[BX,BY,BZ]=atoms[JD]
        for i in range(N):
            #print(i,EXPON[N-1][i],COEF[N-1][i],ZETA1,ZETA2)
            CA[i]=EXPON[N-1][i]*ZETAS[ID]**2
            DA[i]=COEF[N-1][i]*((2.0*CA[i]/PI)**0.75)
            CB[i]=EXPON[N-1][i]*(ZETAS[JD]**2)
            DB[i]=COEF[N-1][i]*((2.0*CB[i]/PI)**0.75)

        N=1
        V=0
        for I in range(N):            
            for J in range(N):
                        ca=CA[I]
                        cb=CB[J]
                        #V000000(a,b,RA,RB,RC)
                        V=V+S000000(ca,cb,RA,RB)*DA[I]*DB[J]
                        #V=V+TTZZZZE(ISABZZ,ca,cb,0,0,RA,RB,[0,0,0],[0,0,0])*DA[I]*DB[J]
        V=sympy.N(V)
        print(RA,RB,V)
        SAB[ID][JD]=V
        
KAB=[[0 for _ in range(NORBITALS)] for _  in range(NORBITALS)]              
for ID in range(NORBITALS):
    for JD in range(NORBITALS):
        RA=[AX,AY,AZ]=atoms[ID]
        RB=[BX,BY,BZ]=atoms[JD]
        for i in range(N):
            #print(i,EXPON[N-1][i],COEF[N-1][i],ZETA1,ZETA2)
            CA[i]=EXPON[N-1][i]*ZETAS[ID]**2
            DA[i]=COEF[N-1][i]*((2.0*CA[i]/PI)**0.75)
            CB[i]=EXPON[N-1][i]*(ZETAS[JD]**2)
            DB[i]=COEF[N-1][i]*((2.0*CB[i]/PI)**0.75)

        N=1
        V=0
        for I in range(N):            
            for J in range(N):
                        ca=CA[I]
                        cb=CB[J]
                        #V000000(a,b,RA,RB,RC)
                        V=V+K000000(ca,cb,RA,RB)*DA[I]*DB[J]
                        #V=V+TTZZZZE(IKABZZ,ca,cb,cc,cd,RA,RB,RC,RD)*DA[I]*DB[J]
        V=sympy.N(V)
        #print(RA,RB,V)
        KAB[ID][JD]=V


# In[ ]:


VAB_C


# In[ ]:


x,y,z,u,v,w=symbols("x y z u v w")
p,q,r,s,t=symbols("p,q,r,s,t")
lcao=[x,y,u,v,w]
lcao2=[p,q,r,s,t]
def SCF_SIMBOL3(IOP,N,R,ZETA1,ZETA2,ZA,ZB):
#
# PREPARES THE ANALYTIC FORMULA OF THE TOTAL ENERGY.
#


    P=[[0 for _ in range(NORBITALS)] for _ in range(NORBITALS)]
    P2=[[0 for _ in range(NORBITALS)] for _ in range(NORBITALS)]
    G=[[0 for _ in range(NORBITALS)] for _ in range(NORBITALS)]
    G2=[[0 for _ in range(NORBITALS)] for _ in range(NORBITALS)]
    F=[[0 for _ in range(NORBITALS)] for _ in range(NORBITALS)]
    F2=[[0 for _ in range(NORBITALS)] for _ in range(NORBITALS)]
    H=[[0 for _ in range(NORBITALS)] for _ in range(NORBITALS)]
    x,y,z,u,v,w=symbols("x y z u v w")
    p,q,r,s,t=symbols("p,q,r,s,t")
    lcao=[x,y,u,v,w]
    lcao2=[p,q,r,s,t]
    PI=np.pi
    CRIT=1.0e-4
    MAXIT=25
    ITER=0
    for I in range(NORBITALS):
        for J in range(NORBITALS):
            P[I][J]=lcao[I]*lcao[J]*2+lcao2[I]*lcao2[J]*2

    for I in range(NORBITALS):
        for J in range(NORBITALS):
            G[I][J]=0
            G2[I][J]=0
            for K in range(NORBITALS):
                for L in range(NORBITALS):
                    #print(I,J,K,L,P[K][L],TT[I][J][K][L],TT[I][L][J][K],G[I][J])
                    #print(I,J,K,L)
                    G[I][J]+=P[K][L]*TTlist.get(indexijkl(I,J,K,L))-0.5*P[K][L]*TTlist.get(indexijkl(I,L,J,K))
    H=[[0 for _ in range(NORBITALS)] for _ in range(NORBITALS)]
    for I in range(NORBITALS):
        for J in range(NORBITALS):
            H[I][J]=KAB[I][J]
            for K in range(NORBITALS):
                H[I][J]+=VAB_C[I][J][K]

    for i in range(NORBITALS):
        for j in range(NORBITALS):
            F[i][j]=H[i][j]+G[i][j]

    EN=0
    for i in range(NORBITALS):
        for j in range(NORBITALS):
            EN+=0.5*P[i][j]*(H[i][j]+F[i][j])
    ENT=EN
    for i in range(NORBITALS):
        for j in range(i+1,4):
            RPQ=[(c1-c2)**2 for c1,c2 in zip(atoms[i],atoms[j])]
            RPQ2=sum(RPQ)
            #print(RPQ2)
            ENT+=ZS[i]*ZS[j]/np.sqrt(RPQ2)
            


    return EN,ENT,F,F2,H,P
x,y,z,u,v,w=symbols("x y z u v w")
EN,ENT,FM,FM2,HM,PM=SCF_SIMBOL3(IOP,N,R,ZETA1,ZETA2,ZA,ZB)


# In[ ]:


TTlist.values()


# In[ ]:


ENV=0
for i in range(3):
    for j in range(i+1,3):
        RPQ=[(c1-c2)**2 for c1,c2 in zip(atoms[i],atoms[j])]
        RPQ2=sum(RPQ)
        #print(RPQ2)
        ENV+=ZS[i]*ZS[j]/R
ENV


# In[ ]:


from sympy import series


# In[ ]:


ENT=ENT.expand()


# In[ ]:


ENS=series(ENT,R,1.7)


# In[ ]:


def GetNormS(vec,SAB):
    V=0
    for i in range(len(vec)):
        for j in range(len(vec)):
            V+=vec[i]*SAB[i][j]*vec[j]
    return V

def vSw(vec,SAB,wvec):
    V=0
    for i in range(len(vec)):
        for j in range(len(wvec)):
            V+=vec[i]*SAB[i][j]*wvec[j]
    return V

e,f,g=symbols("e f g")
#OBJ=ENT-2*e*(GetNormS([x,y,z],SAB)-1)


# In[ ]:


EPART=ENS.removeO().expand()


# In[ ]:


SPART=series(GetNormS(lcao,SAB)-1,R,1.7).removeO().expand()
SPART2=series(GetNormS(lcao2,SAB)-1,R,1.7).removeO().expand()
SPART3=series(vSw(lcao,SAB,lcao2),R,1.7).removeO().expand()


# In[ ]:





# In[ ]:


EPART


# In[ ]:


OBJE=sympy.N(EPART-2*e*SPART-2*f*SPART2-2*g*SPART3)
OBJ=OBJE.expand()
SPART3


# In[ ]:


EPART


# In[ ]:


import copy
OBJ0=copy.deepcopy(OBJ)
#One can try various electronic configurations:
# LCAFO alpha(x,x,x,x); LCAO beta (p,p,-p,-p)
OBJ=OBJ0.subs([(y,x),(u,x),(w,x),(q,p),(r,-p),(s,-p)])

#One can try various electronic configurations:
# LCAFO alpha(x,y,u,w); LCAO beta (0,0,0,0)
OBJ=OBJ0.subs([(p,0),(q,0),(r,0),(s,0),(t,0),(f,0)])


# In[ ]:


sympy.poly(sympy.N(OBJ))


# In[ ]:


def getINT(x,N):
#
#  ROWND DOWN x*N AFTER THE DECIMAL POINT and get AN INTEGER M
#  THEN X is approximated by M/N. 
#
    return (int(np.floor(x*N)))

def getPowerProduct(args,tm):
    V=1
    for w,p in zip(list(args[1:]),tm):
        V*= w**p
    return V
OBJ=OBJ.expand()
ENTS3=sympy.poly(sympy.N(OBJ))
AT=ENTS3.terms()
getF=0
for tm in AT:
    #p0,p1,p2,p3,p4,p5,p6,p7=tm[0]
    cf=tm[1]
    #print(p0,p1,p2,cf)
    #getF+=x**p0* y**p1* z**p2* u**p3* v**p4* w**p5* e**p5* f**p7*getINT(cf,10000)
    getF+=getPowerProduct(ENTS3.args,tm[0])*getINT(cf,10000)


# In[ ]:


str(ENTS3.args[1:])


# In[ ]:


getF
Fargs=str(ENTS3.args[1:])
stringQ='option(noredefine);LIB "solve.lib";option(redSB);\n'
stringQ+='ring R=0,'+Fargs+',dp;\n'+'poly OBJ='+str(getF)+';\n'
stringQ+='list diffs;\n'
stringQ+='for(int i=1;i<=nvars(R); i=i+1){diffs=insert(diffs,diff(OBJ,var(i)));}\n'
stringQ+='ideal I=0;\n'
stringQ+='for(int i=1;i<=nvars(R); i=i+1){I=I+diff(OBJ,var(i));}\n'
stringQ+='print(I);'
stringQ+='ideal SI=std(I);\n'
stringQ+='print(SI);'
stringQ+='ring s=0,'+Fargs+',lp;\n'
stringQ+='setring s;\n'
stringQ+='ideal j=fglm(R,SI);\n'
stringQ+='dim(j);\n'
stringQ+='def S=triang_solve(j,80);\n'
stringQ+='setring S;rlist;quit;'
stringQ+='poly OBJ=fetch(R,OBJ);\
ideal I=fetch(R,I);\
I;\
OBJ;\
for (i=1;i<=size(rlist);i=i+1)\
{\
list substv;\
\
poly OBJ2=OBJ;\
for (int k=1; k<=nvars(R); k=k+1)\
{\
 OBJ2=subst(OBJ2,var(k), rlist[i][k]);\
}\
substv=insert(substv,OBJ2);\
\
for (int l=1;l<size(I);l=l+1)\
{\
poly OBJ2=I[l];\
 for (int k=1; k<=nvars(R); k=k+1)\
 {\
  OBJ2=subst(OBJ2,var(k), rlist[i][k]);\
 }\
 substv=insert(substv,OBJ2);\
}\
print(substv);}'

text_file = open("SCRIPT.txt", "w")
on = text_file.write(stringQ)
 
#close file
text_file.close()


# In[ ]:


import subprocess
cp = subprocess.run("wsl Singular<SCRIPT.txt", shell=True,capture_output=True,text=True)
print("stdout:", cp.stdout)
print("stderr:", cp.stderr)

