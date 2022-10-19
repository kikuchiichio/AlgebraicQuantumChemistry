#!/usr/bin/env python
# coding: utf-8

# # Two-Electron Integrals through Obara-Saika recursion method.
# 
# ## We compute functions of the form $$
# 
# $
# \phi_A(r)=(x-Ax)^{i1}(y-Ay)^{i2}(z-Az)^{i2}\exp(-a|r-A|^2)
# $
# 
# $
# \phi_B(r)=(x-Bx)^{i1}(y-By)^{i2}(z-Vz)^{i2}\exp(-b|r-B|^2)
# $
# 
# $
# \phi_C(r)=(x-Cx)^{i1}(y-Cy)^{i2}(z-Cz)^{i2}\exp(-b|r-C|^2)
# $
# 
# $
# \phi_C(r)=(x-Dx)^{i1}(y-Dy)^{i2}(z-Dz)^{i2}\exp(-b|r-D|^2)
# $
# 
# $
# [AB|CD]=\int dr_1 dr_2 \phi_A(r_1)\phi_B(r_2)\frac{1}{|r_1-r_2|}\phi_C(r_1)\phi_D(r_2)
# $
# $
# I(0;i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)=[AB|CD]
# $
# 
#  Read this notion in this way:
# 
# $(i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)\rightarrow((i1,i2,i3),(j1,j2,j3),(k1,k2,k3),(l1,l2,l3))$
# 
# 
# We also use the symbolic differentiation to compute the integrals. The keys $(i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)$ shall indicate the differentiation.
# 
# $(i1,i2,i3)$: The insruction to the differention with respect to the first orbital at (Ax,Ay,Az). 
# 
# (0,1,0) : d/dAy, (1,1,0): d^2/dAx/dAy, and so on.
#     
# Likewise
# 
# $(j1,j2,j3)$: ... with respect to the secicond orbital.
# 
# $(k1,k2,k3)$: ... with respect to the third orbital.
# 
# $(l1,l2,l3)$: ... with respect to the fourth orbital.
# 
# ## We use several functions to conduct recursive computations.
# 
# In the folllowing, we omit the subscript index (1,2,3) which represents (x,y,z). 
# 
# (i+1) means the shift from i with the increasement to one of the coordinate by 1; the direction of the increasment shall be clear in the contexts. 
# 
# ## <Vertical> computes in this way:  
# 
# (N; i+1 0 0 0 ) <== (N; i 0 0 0) , (N+1; i 0 0 0) , (N; i-1 0 0 0), (N+1; i-1 0 0 0)
# 
# 
# 
# ## Horizontal computes in this way:
# 
# (N; i 0 k+1 0) <== (N; i 0 k 0), (N; i-1 0 k 0), (N; i, 0, k-1, 0), (N; i+1 0 k 0)
# 
# 
# ## TRANSFER 1 and 2. these functions compute in this way:
# 
# 1:(N; i j+1 k l ) <== (N;i+1 j k l), (N; i j k l)
# 2:(N; i j k l+1 ) <== (N;j k+1 l), (N; i j k l)
# 
# 
# <1> we prepare I(N;0,0,0,0,0,0,0,0,0,0,0,0) for N=0,...,Nmax
# 
# <2> For any I(N;i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3), we try the above formulas to see if the integral is compted from the other integrals which have already been computed. 
# 
# <3> If all necessary ingredients are prepared, we compute the integral.
# 
# <4> If some of the necessary ingredients are lacking, we request their computations in the following computations.
# 
# <5> Then we check and compute the requested integrals, by repeating <2>--<4>; we might request more of not-yet-computed integrals. 
# 
# <C1> In the computation, we write and rewrite the list of requested integrals and that of computed ones.
# 
# <C2> The recusive process finally touches its bottom, where it shall find {I(N;0,0,0,0,0,0,0,0,0,0,0,0),N=1,...Nmax}. Hence it terminates after a finite numer of computations.
# 
# <C4> My implementation simply applies every formulas to check the computability of the integrals. It would cause a troube.
# 
# For example, let us compute I(0, 0,0,0, 0,0,0, 0,0,0, 0,0,1).
#  
# (P1) It is computed by Transfer2 along z-direction, from I(0, 0,0,0, 0,0,0, 0,0,1, 0,0,0) and I(0, 0,0,0, 0,0,0, 0,0,0, 0,0,0).
# 
# (P2) II(0, 0,0,0, 0,0,0, 0,0,0, 0,0,0) is already prepared at <1>. We request the computation of I(0, 0,0,0, 0,0,0, 0,0,1, 0,0,0).
# 
# (P3) We compute I(0, 0,0,0, 0,0,0, 0,0,1, 0,0,0) by Horizontal along z-direction,
#   from (A) I(0, 0,0,0, 0,0,0, 0,0,0, 0,0,0), (B) I(0, 0,0,-1, 0,0,0, 0,0,0, 0,0,0), (C) I(0, 0,0,0, 0,0,0, 0,0,-1, 0,0,0), 
#        and (D) I(0, 0,0,1, 0,0,0, 0,0,0)
#   (A) is already prepared. (B) and (C) are the integrals not defined, and we safely regard them as zero.
#   We request (D) I(0, 0,0,1, 0,0,0, 0,0,0).
#   
# (P4) I(0, 0,0,1, 0,0,0, 0,0,0) is computable by <vertical> along z-direction, 
#      from I(0, 0,0,0, 0,0,0, 0,0,1, 0,0,0) and I(1, 0,0,0, 0,0,0, 0,0,1, 0,0,0).
#      
# Nevertheless, we might assume another path.
# 
# (P3') We try to compute I(0, 0,0,0, 0,0,0, 0,0,1, 0,0,0) by <vertical> along x-direction.
#       If we uncarefully apply the formula,
#       it requests I(0, -1,0,0, 0,0,0, 0,0,1, 0,0,0), 
#                   I(1, -1,0,0, 0,0,0, 0,0,1, 0,0,0),
#                   I(0, -2,0,0, 0,0,0, 0,0,1, 0,0,0), and
#                   I(1, -2,0,0, 0,0,0, 0,0,1, 0,0,0).
#       All of those integrals have negative values in the keys, and we should regard them as zero.
#       Then we get the integral I(0, 0,0,0, 0,0,0, 0,0,1, 0,0,0)=0, which is absurd.
# 
# We must avoid such absurd application of formulas. 
# 
# However, in the following, we do computations without this caution.
# 
# The integrals are numerically evaluated in two ways: 
#    (W1) By Obara-Saika recurison mehod
#    (W2) By formulas made by symbolic differentiation of I(0, 0,0,0, 0,0,0, 0,0,0, 0,0,0)
#    
# We should be confindent of (W2)! 
# 

# In[1]:


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
#                 z1,z2,z3,z4=[random.random() for _ in range(4)]
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


# In[2]:


str(f)


# In[3]:


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



# In[4]:


import sympy as sympy
from sympy import symbols, Function,expand,core,sqrt
import numpy as np

from sympy import symbols, Function,expand,sqrt
import numpy as np
import copy

R_length=0.9/0.129177
atomA=[0,0,0]
atomB=[R_length,0,0]
atomC=[R_length*1/2,R_length*np.sqrt(3)/2,0]
atoms=[atomA,atomB,atomC]

S12,T11,T12,T22,V11A,V12A,V22A,V11B,V12B,V22B,V1111,V2111,V2121,V2211,V2221,V2222=[0 for _ in range(16)]

S=[[0 for _ in range(3)] for _ in range(3)]
X=[[0 for _ in range(3)] for _  in range(3)]
NM=[[0 for _ in range(3)] for _  in range(3)]
XT=[[0 for _ in range(3)] for _ in  range(3)]
H=[[0 for _ in range(3)] for _ in  range(3)]
F=[[0 for _ in range(3)] for _ in  range(3)]
G=[[0 for _ in range(3)] for _ in  range(3)]
C=[[0 for _ in range(3)] for _ in  range(3)]
FPRIME=[[0 for _ in range(3)] for _  in range(3)]
CPRIME=[[0 for _ in range(3)] for _  in range(3)]
P=[[0 for _ in range(3)] for _  in range(3)]
ODLPP=[[0 for _ in range(3)] for _  in range(3)]
E=[[0 for _ in range(3)] for _  in range(3)]
TT=[[[[0 for _ in range(3)] for _  in range(3)] for _ in range(3)] for _ in range(3)]


IOP=2
#
# CHOOSE STO-NG
#
N=3
R=1.4632
#
# Gaussian exponents:
# He:ZETA1; H;ZETA2
#
ZETA1=2.0925
ZETA2=1.24
ZETA1=ZETA2
ZETA3=ZETA2
#
#
#
ZA=2.0
ZB=1.0
ZA=ZB
ZC=ZB
#
# ATOMIC DISTANCE 
#
R=symbols("R",positive=True)
# WAVEFUNCTION (x,y)
x, y, z= symbols("x y z")


def F0(ARG):
#
# F0 FUNCTION
#
    PI = np.pi
    if  type(ARG)==float and ARG < 1.0e-6:
        return 1 -ARG/3.
    if  type(ARG)==sympy.core.numbers.Zero and ARG < 1.0e-6:
        return 1 -ARG/3.
    else:
        #print("F0:ARG",ARG)
        return sqrt(PI/ARG)*sympy.erf(sqrt(ARG))/2
    
def getS(A,B,RAB2):
#
# Calculates the overlap matrix
#   
    PI=np.pi
    #print(A,B,RAB2)
    return (PI/(A+B))**1.5*sympy.exp(-A*B*RAB2/(A+B))

def getT(A,B,RAB2):
#
# Calculates the kinetic energy.
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
    PI=np.pi
    return 2.0*(PI**2.5)/((A+B)*(C+D)*np.sqrt(A+B+C+D))*F0((A+B)*(C+D)*RPQ2/(A+B+C+D))*sympy.exp(-A*B*RAB2/(A+B)-C*D*RCD2/(C+D))



# In[5]:


#for a in integrals_all.keys():
#    N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=a
#    if N==0 and min(list(a))>=0:
#        print(a)
#
#  THERE ARE KEYS REQUESTED TO COMPUTE INTEGRALS. BECAUSE THEY WERE NOT PREPARED IN ADVANCE.
#

print(len(requested_keys.keys()))
for keyr in integrals_all.keys():
    getV=requested_keys.get(keyr)
    if getV!=None:
        requested_keys.pop(keyr)
print(len(requested_keys.keys()))
#for kr in requested_keys.keys():
#    N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=list(kr)
#    if i1<=2 and i2<=2 and i3<=2:
#        print(kr)

#
#  NEXT WE DO HORIZONTAL RECUSIONS.
#
# HORIZONRAL
# (N; i 0 k+1 0) <= (N; i 0 k 0), (N; i-1 0 k 0), (N; i, 0, k-1, 0), (N; i+1 0 k 0)
#
#  SIMILARLY WE USE THE FUNCTION WHICH MAKES THE HORIZONTAL SHIFT ALONG X/Y/Z DIRECTIONS
#

def HORIZONTAL(D,N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3,INFO=0):
    if D=="X":
        key0=(N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        key1=(N,i1-1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        key2=(N,i1,i2,i3,j1,j2,j3,k1-1,k2,k3,l1,l2,l3)
        key3=(N,i1+1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        keyn=(N,i1,i2,i3,j1,j2,j3,k1+1,k2,k3,l1,l2,l3)
        i_d,k_d=i1,k1
        #print(D,keyn)
        DAB=XAB
        DCD=XCD
    if D=="Y":
        key0=(N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        key1=(N,i1,i2-1,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        key2=(N,i1,i2,i3,j1,j2,j3,k1,k2-1,k3,l1,l2,l3)
        key3=(N,i1,i2+1,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        keyn=(N,i1,i2,i3,j1,j2,j3,k1,k2+1,k3,l1,l2,l3)
        i_d,k_d=i2,k2
        DAB=YAB
        DCD=YCD
        #print(D,keyn)
    if D=="Z":
        key0=(N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        key1=(N,i1,i2,i3-1,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        key2=(N,i1,i2,i3,j1,j2,j3,k1,k2,k3-1,l1,l2,l3)
        key3=(N,i1,i2,i3+1,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        keyn=(N,i1,i2,i3,j1,j2,j3,k1,k2,k3+1,l1,l2,l3)
        i_d,k_d=i3,k3
        #print(D,keyn)
        DAB=ZAB
        DCD=ZCD
    data0=integrals_all.get(key0)
    data1=integrals_all.get(key1)
    data2=integrals_all.get(key2)
    data3=integrals_all.get(key3)
    nx.add_star(IG, [keyn,key0, key1, key2, key3])

    if min(list(key0))<0:
        data0=0
        integrals_all[key0]=0
    if min(list(key1))<0:
        data1=0
        integrals_all[key1]=0
    if min(list(key2))<0:
        data2=0
        integrals_all[key2]=0
    if min(list(key3))<0:
        data3=0
        integrals_all[key3]=0
 
    if data0==None:
        requested_keys[key0]=key0
    if data1==None:
        requested_keys[key1]=key1
    if data2==None:
        requested_keys[key2]=key2
    if data3==None:
        requested_keys[key3]=key3
    #if data0==None:
    #    integrals_all[key0]="REQUIRED"
    #if data1==None:
    #    integrals_all[key1]="REQUIRED"
    #if data2==None:
    #    integrals_all[key2]="REQUIRED"
    #if data3==None:
    #    integrals_all[key3]="REQUIRED"
    if min(list(key0))<0 and min(list(key1))<0 and min(list(key2))<0 and min(list(key3))<0:
        #
        # Maybe in this case, the formula cannot be applied.
        #
        if (INFO==1):
            print(key0,key1,key2,key3,D)
        return
    if data0!=None and data1!=None and data2!=None and data3!=None:
        #print(data0,data1,data2,data3)
        #print(b,d,DAB,DCD,i1,k1,p,q)
        integrals_all[keyn]= -(b*DAB+d*DCD)/q*data0 + i_d/2/q*data1 +k_d/2/q*data2 -p/q*data3
    #else:
        #print(keyn,"NOT WRITTEN")
        #print(key0,key1,key2,"key3=",key3)
        #print(data0,data1,data2,data3)
        
def HORIZONTAL0(D,N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3,INFO=0):
    if D=="X":
        key0=(N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        key1=(N+1,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        key2=(N+1,i1-1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        key3=(N,i1,i2,i3,j1,j2,j3,k1-1,k2,k3,l1,l2,l3)
        key4=(N+1,i1,i2,i3,j1,j2,j3,k1-1,k2,k3,l1,l2,l3)
        keyn=(N,i1,i2,i3,j1,j2,j3,k1+1,k2,k3,l1,l2,l3)
        i_d,k_d=i1,k1
        DQC=XQC
        DPQ=XPQ
    if D=="Y":
        key0=(N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        key1=(N+1,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        key2=(N+1,i1,i2-1,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        key3=(N,i1,i2,i3,j1,j2,j3,k1,k2-1,k3,l1,l2,l3)
        key4=(N+1,i1,i2,i3,j1,j2,j3,k1,k2-1,k3,l1,l2,l3)
        keyn=(N,i1,i2,i3,j1,j2,j3,k1,k2+1,k3,l1,l2,l3)
        i_d,k_d=i2,k2
        DQC=YQC
        DPQ=YPQ
        #print(D,keyn)
    if D=="Z":
        key0=(N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        key1=(N+1,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        key2=(N+1,i1,i2,i3-1,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        key3=(N,i1,i2,i3,j1,j2,j3,k1,k2,k3-1,l1,l2,l3)
        key4=(N+1,i1,i2,i3,j1,j2,j3,k1,k2,k3-1,l1,l2,l3)
        keyn=(N,i1,i2,i3,j1,j2,j3,k1,k2,k3+1,l1,l2,l3)
        i_d,k_d=i3,k3
        #print(D,keyn)
        DQC=ZQC
        DPQ=ZPQ
    data0=integrals_all.get(key0)
    data1=integrals_all.get(key1)
    data2=integrals_all.get(key2)
    data3=integrals_all.get(key3)
    data4=integrals_all.get(key4)
    nx.add_star(IG, [keyn,key0, key1, key2, key3, key4])

    if min(list(key0))<0:
        data0=0
        integrals_all[key0]=0
    if min(list(key1))<0:
        data1=0
        integrals_all[key1]=0
    if min(list(key2))<0:
        data2=0
        integrals_all[key2]=0
    if min(list(key3))<0:
        data3=0
        integrals_all[key3]=0
    if min(list(key4))<0:
        data4=0
        integrals_all[key4]=0
    if data0==None:
        requested_keys[key0]=key0
    if data1==None:
        requested_keys[key1]=key1
    if data2==None:
        requested_keys[key2]=key2
    if data3==None:
        requested_keys[key3]=key3
    if data4==None:
        requested_keys[key4]=key4
    #if data0==None:
    #    integrals_all[key0]="REQUIRED"
    #if data1==None:
    #    integrals_all[key1]="REQUIRED"
    #if data2==None:
    #    integrals_all[key2]="REQUIRED"
    #if data3==None:
    #    integrals_all[key3]="REQUIRED"
    if min(list(key0))<0 and min(list(key1))<0 and min(list(key2))<0 and min(list(key3))<0 and min(list(key4))<0:
        #
        # Maybe in this case, the formula cannot be applied.
        #
        if (INFO==1):
            print(key0,key1,key2,key3,key4,D)
        return
    if data0!=None and data1!=None and data2!=None and data3!=None and data4!=None:
        #print(data0,data1,data2,data3)
        #print(b,d,DAB,DCD,i1,k1,p,q)
        integrals_all[keyn]= DQC*data0+alpha/q*data1+i_d/2/(p+q)*data2\
        +k_d/2/q*(data3-alpha/q*data4)
    #else:
        #print(keyn,"NOT WRITTEN")
        #print(key0,key1,key2,"key3=",key3)
        #print(data0,data1,data2,data3)
        


# In[6]:


#
#  WE TRANSFER THE INTEGRALS TO COVER THE BLANK DATA.
#
# TRANSFER 1 and 2: THESE FUNCTION COMPUTES
#  (N; i j+1 k l ) <= (N;i+1 j k l), (N; i j k l)
#  (N; i j k l+1 ) <= (N;j k+1 l), (N; i j k l)
#
#  MAYBE SOMETING NECESSARY ARE STILL MISSING; IF WE DETECT THEM, WE TRY TO COMPUTE THEM BY "HORIZONTAL" ON THE FLY!
#
def try_to_compute(kr,INFO=0):
    if INFO==1:
        print(">>>TRY TO COMPUTE AT",kr)
    N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=kr
    HORIZONTAL("X",N,i1,i2,i3,j1,j2,j3,k1-1,k2,k3,l1,l2,l3)
    HORIZONTAL("Y",N,i1,i2,i3,j1,j2,j3,k1,k2-1,k3,l1,l2,l3)
    HORIZONTAL("Z",N,i1,i2,i3,j1,j2,j3,k1,k2,k3-1,l1,l2,l3)
    if integrals_all.get(kr)!=None:
        if INFO==1:
            print(">>>>SUCCESS!")
        return 1
    else:
        if INFO==1:
            print(">>>>FAIL!")
        return 0

    
def TRANSFER1(D,N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3,INFO=0):
    if D=="X":
        key0=(N,i1+1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        key1=(N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        keyn=(N,i1,i2,i3,j1+1,j2,j3,k1,k2,k3,l1,l2,l3)
        DAB=XAB
    if D=="Y":
        key0=(N,i1,i2+1,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        key1=(N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        keyn=(N,i1,i2,i3,j1,j2+1,j3,k1,k2,k3,l1,l2,l3)
        DAB=YAB
    if D=="Z":
        key0=(N,i1,i2,i3+1,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        key1=(N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        keyn=(N,i1,i2,i3,j1,j2,j3+1,k1,k2,k3,l1,l2,l3)
        DAB=ZAB
        
    data0=integrals_all.get(key0)
    data1=integrals_all.get(key1)
    nx.add_star(IG, [keyn,key0, key1])
    if data0==None:
        requested_keys[key0]=key0
    if data1==None:
        requested_keys[key1]=key1
    if data0!=None and data1!=None:
        #print(key1,"TRANSFERED",keyn)
        #print(data0,data1,data2,data3)
        integrals_all[keyn]= data0+DAB*data1
        return 0
    else:
        if (INFO==1):
            print("TRANSFER(", key1,"->",keyn, ") \n   !FAILS AT KEY0=",key0)
        return key0
    
def TRANSFER2(D,N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3,INFO=0):
    if D=="X":
        key0=(N,i1,i2,i3,j1,j2,j3,k1+1,k2,k3,l1,l2,l3)
        key1=(N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        keyn=(N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1+1,l2,l3)
        DCD=XCD
    if D=="Y":
        key0=(N,i1,i2,i3,j1,j2,j3,k1,k2+1,k3,l1,l2,l3)
        key1=(N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        keyn=(N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2+1,l3)
        DCD=YCD
    if D=="Z":
        key0=(N,i1,i2,i3,j1,j2,j3,k1,k2,k3+1,l1,l2,l3)
        key1=(N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        keyn=(N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3+1)
        DCD=ZCD
    data0=integrals_all.get(key0)
    data1=integrals_all.get(key1)
    nx.add_star(IG, [keyn,key0, key1])
    if data0==None:
        requested_keys[key0]=key0
    if data1==None:
        requested_keys[key1]=key1
    if data0!=None and data1!=None:
        #print(key1,"TRANSFERED",keyn)
        #print(data0,data1)
        integrals_all[keyn]= data0+DCD*data1
        return 0
    else:
        if (INFO==1):
            print("TRANSFER(", key1,"->",keyn, "\n    FAILS AT KEY0=",key0)
        return key0


# In[7]:


a,b,c,d, p, q=symbols("a b c d p q")
XA,YA,ZA=symbols("XA YA ZA")
XB,YB,ZB=symbols("XB YB ZB")
XC,YC,ZC=symbols("XC YC ZC")
XD,YD,ZD=symbols("XD YD ZD")
XP,YP,ZP=symbols("XP YP ZP")
XQ,YQ,ZQ=symbols("XQ YQ ZQ")
XAB=symbols("XAB")
XPA=symbols("XPA")
TN0,TN1=symbols("TN0,TN1")

p=a+b

X_P=(a*XA+b*XB)/p
Y_P=(a*YA+b*YB)/p
Z_P=(a*ZA+b*ZB)/p
q=c+d

X_Q=(c*XC+d*XD)/q
Y_Q=(c*YC+d*YD)/q
Z_Q=(c*ZC+d*ZD)/q
X_AB=XA-XB
X_PA=XP-XA
X_PQ=XP-XQ
X_CD=XC-XD
F1=sympy.sympify(str(integrals_all.get((0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))))
F2=sympy.sympify(str(integrals_all.get((0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0))))
F3=sympy.sympify("XAB")


# In[8]:


requested_keys=dict()
integrals_all=dict()
NMAX=11
for i in range(NMAX):
    integrals_all[(i,0,0,0,0,0,0,0,0,0,0,0,0)]=TN(i)    

def late_evaluate(ckeys):
#
# THIS FUNCTION COMPUTES THE TWO-ELECTRON INTEGRAL AT CKEYS THROUGH RECUSION,
#   USING THE DATA WHICH HAVE ALREADY BEEN COMPUTED,
#   AND WRITING THE RESULTS WHICH HAVE JUST BEEN COMPUTED
#
    N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=ckeys
    if integrals_all.get(ckeys)!=None:
        print("FOUND")
        return 1
    # VERTICAL:
    if j1==j2==j3==k1==k2==k3==l1==l2==l3==0:
        #print("VERTICAL")
        VERTICAL("X",N,i1-1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        VERTICAL("Y",N,i1,i2-1,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        VERTICAL("Z",N,i1,i2,i3-1,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        return
    if j1==j2==j3==l1==l2==l3==0:
        HORIZONTAL("X",N,i1,i2,i3,j1,j2,j3,k1-1,k2,k3,l1,l2,l3)
        HORIZONTAL("Y",N,i1,i2,i3,j1,j2,j3,k1,k2-1,k3,l1,l2,l3)
        HORIZONTAL("Z",N,i1,i2,i3,j1,j2,j3,k1,k2,k3-1,l1,l2,l3)
        return
    if j1-1>=0:
        TRANSFER1("X",N,i1,i2,i3,j1-1,j2,j3,k1,k2,k3,l1,l2,l3)
    if j2-1>=0:
        TRANSFER1("Y",N,i1,i2,i3,j1,j2-1,j3,k1,k2,k3,l1,l2,l3)
    if j3-1>=0:
        TRANSFER1("Z",N,i1,i2,i3,j1,j2,j3-1,k1,k2,k3,l1,l2,l3)
    if l1-1>=0:
        TRANSFER2("X",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1-1,l2,l3)
    if l2-1>=0:
        TRANSFER2("Y",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2-1,l3)
    if l3-1>=0:
        TRANSFER2("Z",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3-1)
    #print(requested_keys.keys())

ckeys=(0,1,0,0,0,0,0,0,0,0,0,0,0)    
late_evaluate(ckeys)    
#requested_keys.keys()

CurrentN=list(ckeys)[0]

import copy
IsFound=None
while(IsFound==None and CurrentN<NMAX):
    requested_keys_copy=copy.deepcopy(requested_keys)
    for ckey in requested_keys_copy.keys():
        if min(list(ckey))>=0:
            F=late_evaluate(ckey)
            if F==1:
                print("\n\nCOMPUTED",ckey,integrals_all.get(ckey))
                requested_keys.pop(ckey)

    late_evaluate(ckeys)
    IsFound=integrals_all.get(ckeys)
    if IsFound!=None:
        print(ckeys, "is computed")
print(IsFound)        

ckeys=(0,2,0,0,0,0,0,0,0,0,0,0,0)    
late_evaluate(ckeys)    
#requested_keys.keys()

CurrentN=list(ckeys)[0]

import copy
IsFound=None
while(IsFound==None and CurrentN<NMAX):
    requested_keys_copy=copy.deepcopy(requested_keys)
    for ckey in requested_keys_copy.keys():
        if min(list(ckey))>=0:
            F=late_evaluate(ckey)
            if F==1:
                print("\n\nCOMPUTED",ckey,integrals_all.get(ckey))
                requested_keys.pop(ckey)

    late_evaluate(ckeys)
    IsFound=integrals_all.get(ckeys)
    if IsFound!=None:
        print(ckeys, "is computed")
print(IsFound)        


ckeys=(0,0,0,0,0,0,0,2,0,0,0,0,0)    
late_evaluate(ckeys)    
#requested_keys.keys()

CurrentN=list(ckeys)[0]

import copy
IsFound=None
while(IsFound==None and CurrentN<NMAX):
    requested_keys_copy=copy.deepcopy(requested_keys)
    for ckey in requested_keys_copy.keys():
        if min(list(ckey))>=0:
            F=late_evaluate(ckey)
            if F==1:
                print("\n\nCOMPUTED",ckey,integrals_all.get(ckey))
                requested_keys.pop(ckey)

    late_evaluate(ckeys)
    IsFound=integrals_all.get(ckeys)
    if IsFound!=None:
        print(ckeys, "is computed")
print(IsFound)        


# In[9]:


AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
z1,z2,z3,z4=symbols("z1 z2 z3 z4")
f=T0000(1,1,1,1,[AX,0,0],[0,0,0],[CX,0,0],[1,1,1])
print(f)


# In[10]:


#
# We have computed integrals at various keys by the recustion.
#
ckeys=(0,0,0,0,0,0,0,0,0,0,0,0,0)
v0000=integrals_all.get(ckeys)
ckeys=(0,1,0,0,0,0,0,0,0,0,0,0,0)
v1000=integrals_all.get(ckeys)
ckeys=(0,2,0,0,0,0,0,0,0,0,0,0,0)
v2000=integrals_all.get(ckeys)
ckeys=(0,1,0,0,1,0,0,1,0,0,1,0,0)
late_evaluate(ckeys)    
ckeys=(0,0,0,0,0,0,0,2,0,0,0,0,0)
late_evaluate(ckeys) 
v0020=integrals_all.get(ckeys)
ckeys=(0,1,0,0,0,0,0,1,0,0,0,0,0)
late_evaluate(ckeys) 
v1010=integrals_all.get(ckeys)


# In[11]:


NMAX


# In[12]:


ckeys=(0,1,0,0,1,0,0,1,0,0,1,0,0)
late_evaluate(ckeys)    
#requested_keys.keys()

CurrentN=list(ckeys)[0]

import copy
IsFound=None
while(IsFound==None and CurrentN<NMAX):
    requested_keys_copy=copy.deepcopy(requested_keys)
    for ckey in requested_keys_copy.keys():
        if min(list(ckey))>=0:
            F=late_evaluate(ckey)
            if F==1:
                print("\n\nCOMPUTED",ckey,integrals_all.get(ckey))
                requested_keys.pop(ckey)

    late_evaluate(ckeys)
    IsFound=integrals_all.get(ckeys)
    if IsFound!=None:
        print(ckeys, "is computed")
print(IsFound)  
v1111=integrals_all.get(ckeys)


# In[13]:


#       |x    |x    |  y  |    z|   
ckeys=(0,1,0,0,1,0,0,0,1,0,0,0,1)
late_evaluate(ckeys) 
CurrentN=list(ckeys)[0]

import copy
IsFound=None
while(IsFound==None and CurrentN<NMAX):
    requested_keys_copy=copy.deepcopy(requested_keys)
    for ckey in requested_keys_copy.keys():
        if min(list(ckey))>=0:
            F=late_evaluate(ckey)
            if F==1:
                print("\n\nCOMPUTED",ckey,integrals_all.get(ckey))
                requested_keys.pop(ckey)

    late_evaluate(ckeys)
    IsFound=integrals_all.get(ckeys)
    if IsFound!=None:
        print(ckeys, "is computed")
print(IsFound)  
if (None!=integrals_all.get(ckeys)):
    print("You have goten it!")
    
vxxyz=integrals_all.get(ckeys)

#       |x    |  y  |z   |    z|   
ckeys=(0,1,0,0,0,1,0,0,0,1,0,0,1)
late_evaluate(ckeys) 

late_evaluate(ckeys) 
CurrentN=list(ckeys)[0]

import copy
IsFound=None
while(IsFound==None and CurrentN<NMAX):
    requested_keys_copy=copy.deepcopy(requested_keys)
    for ckey in requested_keys_copy.keys():
        if min(list(ckey))>=0:
            F=late_evaluate(ckey)
            if F==1:
                print("\n\nCOMPUTED",ckey,integrals_all.get(ckey))
                requested_keys.pop(ckey)

    late_evaluate(ckeys)
    IsFound=integrals_all.get(ckeys)
    if IsFound!=None:
        print(ckeys, "is computed")
print(IsFound)  
if (None!=integrals_all.get(ckeys)):
    print("You have goten it!")
    
vxyzz=integrals_all.get(ckeys)


# In[14]:


str(v1111)


# # Check the validity of the formulas
# A reference shows the formula:
# 
# $\theta^N_{i,0,k+1,0} = -\frac{bX_{AB}+dX_{CD}}{q}\theta^N_{i,0,k,0}  -\frac{i}{2q}\theta^N_{i-1,0,k,0}+\frac{k}{2p}\theta^N_{i,0,k-1,0}-\frac{p}{q}\theta^N_{i+1,0,k,0}$
# 
# However another uses different one:
# 
# $\theta^N_{i,0,k+1,0} = -\frac{bX_{AB}+dX_{CD}}{q}\theta^N_{i,0,k,0} +\frac{i}{2q}\theta^N_{i-1,0,k,0}+\frac{k}{2p}\theta^N_{i,0,k-1,0}-\frac{p}{q}\theta^N_{i+1,0,k,0}$
# 

# In[15]:


AX


# In[16]:


#  A TEST OF THE PROGRAM
# 
#  Compute various integrals [[AX,0,0],[0,0,0] | [CX,0,0],[1,0,0] ]
#
#
#  We use the following rule: 
#   The expressions such as X_A, X_CD, Y_B, Z_D... (with understores '_' ) shall represent temporary valiables on X,Y,and Z coordinates,
#     which are often numbers.
#
#   The expressions such as AX,BY,CZ,... (with understores '_' ) always represent symbols on X,Y,and Z coordinates.
#
# 
a,b,c,d, p, q=symbols("a b c d p q")
XA,YA,ZA=symbols("XA YA ZA")
XB,YB,ZB=symbols("XB YB ZB")
XC,YC,ZC=symbols("XC YC ZC")
XD,YD,ZD=symbols("XD YD ZD")
XP,YP,ZP=symbols("XP YP ZP")
XQ,YQ,ZQ=symbols("XQ YQ ZQ")

AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
z1,z2,z3,z4=symbols("z1 z2 z3 z4")

[X_A,Y_A,Z_A]=[AX,0,0]
[X_B,Y_B,Z_B]=[0,0,0]
[X_C,Y_C,Z_C]=[CX,0,0]
[X_D,Y_D,Z_D]=[1,0,0]


[X_AB,Y_AB,Z_AB]=[X_A-X_B,Y_A-Y_B,Z_A-Z_B]
[X_CD,Y_CD,Z_CD]=[X_C-X_D,Y_C-Y_D,Z_C-Z_D]

aa=1.45
bb=1.54
cc=1.3
dd=1.2

X_P=(aa*X_A+bb*X_B)/(aa+bb)
Y_P=(aa*Y_A+bb*Y_B)/(aa+bb)
Z_P=(aa*Z_A+bb*Z_B)/(aa+bb)

pp=aa+bb
qq=cc+dd

X_Q=(cc*X_C+dd*X_D)/(cc+dd)
Y_Q=(cc*Y_C+dd*Y_D)/(cc+dd)
Z_Q=(cc*Z_C+dd*Z_D)/(cc+dd)


X_PQ=X_P-X_Q
Y_PQ=Y_P-Y_Q
Z_PQ=Z_P-Z_Q


X_PA=X_P-X_A
Y_PA=Y_P-Y_A
Z_PA=Z_P-Z_A
PQ2=X_PQ**2+Y_PQ**2+Z_PQ**2



alpha=pp*qq/(pp+qq)



alpha=pp*qq/(pp+qq)
f=T0000(aa,bb,cc,dd,[AX,0,0],[0,0,0],[CX,0,0],[1,0,0])
f4=T0000(aa,bb,cc,dd,[AX,0,0],[BX,0,0],[CX,0,0],[DX,0,0])
# th0000 [ss|ss] at [AB|CD] 
th0000=f
# th1010 [ps|ps] at [AB|CD]
th1010=f.diff(AX).diff(CX)/4/aa/cc
# th1111 [pxpx|pxpx] at [AB|CD]
th1111=f4.diff(AX).diff(BX).diff(CX).diff(DX)/16/aa/bb/cc/dd
th0010=f.diff(CX)/2/cc
th1000=f.diff(AX)/2/aa
th2000=(f.diff(AX).diff(AX)+th0000*2*aa)/4/aa/aa
th0020=(f.diff(CX).diff(CX)+th0000*2*cc)/4/cc/cc


# In[17]:


v0000


# In[18]:


v1000


# In[ ]:





# In[19]:


v2000


# #
# # TEST OF COMPUTATION
# #

# In[20]:


#
# EVALUATED THETA(N;000 000 000 000) at ([AX,0,0],[0,0,0],[CX,0,0],[DX,0,0])
#
V_T_0_0000=2*sympy.pi**2.5/pp/qq/sympy.sqrt(pp+qq)*KXYZAB(aa,bb,[X_A,Y_A,Z_A],[X_B,Y_B,Z_B])*KXYZAB(cc,dd,[X_C,Y_C,Z_C],[X_D,Y_D,Z_D])*BOYS(0,alpha*PQ2)
V_T_1_0000=2*sympy.pi**2.5/pp/qq/sympy.sqrt(pp+qq)*KXYZAB(aa,bb,[X_A,Y_A,Z_A],[X_B,Y_B,Z_B])*KXYZAB(cc,dd,[X_C,Y_C,Z_C],[X_D,Y_D,Z_D])*BOYS(1,alpha*PQ2)
V_T_2_0000=2*sympy.pi**2.5/pp/qq/sympy.sqrt(pp+qq)*KXYZAB(aa,bb,[X_A,Y_A,Z_A],[X_B,Y_B,Z_B])*KXYZAB(cc,dd,[X_C,Y_C,Z_C],[X_D,Y_D,Z_D])*BOYS(2,alpha*PQ2)
V_T_3_0000=2*sympy.pi**2.5/pp/qq/sympy.sqrt(pp+qq)*KXYZAB(aa,bb,[X_A,Y_A,Z_A],[X_B,Y_B,Z_B])*KXYZAB(cc,dd,[X_C,Y_C,Z_C],[X_D,Y_D,Z_D])*BOYS(3,alpha*PQ2)
V_T_4_0000=2*sympy.pi**2.5/pp/qq/sympy.sqrt(pp+qq)*KXYZAB(aa,bb,[X_A,Y_A,Z_A],[X_B,Y_B,Z_B])*KXYZAB(cc,dd,[X_C,Y_C,Z_C],[X_D,Y_D,Z_D])*BOYS(4,alpha*PQ2)


# In[21]:


sympy.N(v0000.subs([(q,qq),(p,pp),(b,bb),(d,dd),(XPQ,X_PQ),(XAB,X_AB),(XCD,X_CD),(XPA,X_PA),(TNS[0],V_T_0_0000),(TNS[1],V_T_1_0000),(AX,1),(CX,2)])).expand()


# In[22]:


sympy.N(th0000.subs([(q,qq),(p,pp),(b,bb),(d,dd),(XPQ,X_PQ),(XAB,X_AB),(XCD,X_CD),(XPA,X_PA),(AX,1),(CX,2)]))


# In[23]:


V_T_0_0000


# In[24]:


th0000


# In[25]:


(v1000.subs([(q,qq),(p,pp),(a,aa),(b,bb),(c,cc),(d,dd),(XPQ,X_PQ),(XAB,X_AB),(XCD,X_CD),(XPA,X_PA),(TNS[0],V_T_0_0000),(TNS[1],V_T_1_0000),(AX,1),(CX,2)])).expand()


# In[26]:


ww=v1000.subs([(q,qq),(p,pp),(b,bb),(d,dd),(XPQ,X_PQ),(XAB,AX),(XCD,CX-1),(XPA,XP-AX),(TNS[0],V_T_0_0000),(TNS[1],V_T_1_0000)])
#ww


# In[27]:


sympy.N(v1000.subs([(q,qq),(p,pp),(a,aa),(b,bb),(c,cc),(d,dd),(XPQ,X_PQ),(XAB,X_AB),(XCD,X_CD),(XPA,X_PA),(TNS[0],V_T_0_0000),(TNS[1],V_T_1_0000),(AX,1),(CX,2)])).expand()


# In[28]:


sympy.N(th1000.subs([(q,qq),(p,pp),(b,bb),(d,dd),(XPQ,X_PQ),(XAB,X_AB),(XCD,X_CD),(XPA,X_PA),(AX,1),(CX,2)]))


# In[29]:


sympy.N(v2000.subs([(q,qq),(p,pp),(a,aa),(b,bb),(c,cc),(d,dd),(XPQ,X_PQ),(XAB,X_AB),(XCD,X_CD),(XPA,X_PA),(TNS[0],V_T_0_0000),(TNS[1],V_T_1_0000),(TNS[2],V_T_2_0000)]).expand()).subs([(AX,1),(CX,2)])


# In[30]:


sympy.N(th2000.expand()).subs([(AX,1),(CX,2)])


# In[31]:


sympy.N(v2000.subs([(q,qq),(p,pp),(a,aa),(b,bb),(c,cc),(d,dd),(XPQ,X_PQ),(XAB,AX),(XCD,(CX-1)),(XPA,(X_P-AX)),(TNS[0],V_T_0_0000),(TNS[1],V_T_1_0000),(TNS[2],V_T_2_0000),(AX,1),(CX,2)])).expand()


# In[32]:


sympy.N(th2000.subs([(q,qq),(p,pp),(b,bb),(d,dd),(XPQ,X_PQ),(XAB,X_AB),(XCD,X_CD),(XPA,X_PA),(AX,1),(CX,2)])).expand()


# In[33]:


sympy.N(v1111.subs([(q,qq),(p,pp),(a,aa),(b,bb),(c,cc),(d,dd),(XPQ,X_PQ),(XAB,X_AB),(XCD,X_CD),(XPA,X_PA),\
             (TNS[0],V_T_0_0000),(TNS[1],V_T_1_0000),(TNS[2],V_T_2_0000),\
             (TNS[3],V_T_3_0000),(TNS[4],V_T_4_0000),\
             (AX,1),(CX,2)])).expand()


# In[34]:


sympy.N(th1111.subs([(q,qq),(p,pp),(b,bb),(d,dd),(XPQ,X_PQ),(XAB,AX),(XCD,(CX-1)),(XPA,(X_P-AX)),(BX,0),(DX,1),(TNS[0],V_T_0_0000),(TNS[1],V_T_1_0000),(TNS[2],V_T_2_0000),(AX,1),(CX,2)])).expand()


# In[35]:


pp=aa+bb
qq=cc+dd
V1010=-(bb*X_AB+dd*X_CD)/qq*th1000+1/2/qq*th0000-pp/qq*th2000
V0010=-(bb*X_AB+dd*X_CD)/qq*th0000-pp/qq*th1000
V0020=-(bb*X_AB+dd*X_CD)/qq*th0010+1/2/qq*th0000-pp/qq*th1010


# In[36]:


[aa,bb,cc,dd,pp,qq]


# In[37]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[38]:


v1010


# In[39]:


[a,b,c,d]


# In[40]:


[(V0010.subs([(AX,1),(CX,2)])),(th0010.subs([(AX,1),(CX,2)]))]


# In[41]:


sympy.N(v1010.subs([(q,qq),(p,pp),(b,bb),(a,aa),(c,cc),(d,dd),(XPQ,X_PQ),(XAB,X_AB),(XCD,X_CD),(XPA,X_PA),(BX,0),(DX,1),(TNS[0],V_T_0_0000),(TNS[1],V_T_1_0000),(TNS[2],V_T_2_0000),(AX,1),(CX,2)])).expand()


# In[42]:


[sympy.N(V1010.subs([(AX,1),(CX,2)])),sympy.N(th1010.subs([(AX,1),(CX,2)]))]


# In[43]:


[(V0020.subs([(AX,1),(CX,2)])),(th0020.subs([(AX,1),(CX,2)]))]


# In[44]:


[sympy.N(V0020.subs([(AX,1),(CX,2)])),sympy.N(th0020.subs([(AX,1),(CX,2)]))]


# In[45]:


sympy.N(v0020.subs([(q,qq),(p,pp),(a,aa),(c,cc),(b,bb),(d,dd),(XPQ,X_PQ),(XAB,X_AB),(XCD,X_CD),(XPA,X_PA),(TNS[0],V_T_0_0000),(TNS[1],V_T_1_0000),(TNS[2],V_T_2_0000),(AX,1),(CX,2)])).expand()


# In[51]:


#  A TEST OF THE PROGRAM
# 
#  Compute various integrals [[AX,0,0],[0,0,0] | [CX,0,0],[1,0,0] ]
#
#
#  We use the following rule: 
#   The expressions such as X_A, X_CD, Y_B, Z_D... (with understores '_' ) shall represent temporary valiables on X,Y,and Z coordinates,
#     which are often numbers.
#
#   The expressions such as AX,BY,CZ,... (with understores '_' ) always represent symbols on X,Y,and Z coordinates.
#
# 
a,b,c,d, p, q=symbols("a b c d p q")
XA,YA,ZA=symbols("XA YA ZA")
XB,YB,ZB=symbols("XB YB ZB")
XC,YC,ZC=symbols("XC YC ZC")
XD,YD,ZD=symbols("XD YD ZD")
XP,YP,ZP=symbols("XP YP ZP")
XQ,YQ,ZQ=symbols("XQ YQ ZQ")

AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
z1,z2,z3,z4=symbols("z1 z2 z3 z4")

[X_A,Y_A,Z_A]=[0,1.4,0]
[X_B,Y_B,Z_B]=[0.98,1.5,0]
[X_C,Y_C,Z_C]=[0,0,1.93]
[X_D,Y_D,Z_D]=[1.205,1.02,0]


[X_AB,Y_AB,Z_AB]=[X_A-X_B,Y_A-Y_B,Z_A-Z_B]
[X_CD,Y_CD,Z_CD]=[X_C-X_D,Y_C-Y_D,Z_C-Z_D]

aa=1
bb=1
cc=1
dd=1
aa=1.45
bb=1.54
cc=1.3
dd=1.2

X_P=(aa*X_A+bb*X_B)/(aa+bb)
Y_P=(aa*Y_A+bb*Y_B)/(aa+bb)
Z_P=(aa*Z_A+bb*Z_B)/(aa+bb)

pp=aa+bb
qq=cc+dd

X_Q=(cc*X_C+dd*X_D)/(cc+dd)
Y_Q=(cc*Y_C+dd*Y_D)/(cc+dd)
Z_Q=(cc*Z_C+dd*Z_D)/(cc+dd)


X_PQ=X_P-X_Q
Y_PQ=Y_P-Y_Q
Z_PQ=Z_P-Z_Q
PQ2=X_PQ**2+Y_PQ**2+Z_PQ**2

X_PA=X_P-X_A
Y_PA=Y_P-Y_A
Z_PA=Z_P-Z_A


alpha=pp*qq/(pp+qq)

V_T_0_0000=2*sympy.pi**2.5/pp/qq/sympy.sqrt(pp+qq)*KXYZAB(aa,bb,[X_A,Y_A,Z_A],[X_B,Y_B,Z_B])*KXYZAB(cc,dd,[X_C,Y_C,Z_C],[X_D,Y_D,Z_D])*BOYS(0,alpha*PQ2)
V_T_1_0000=2*sympy.pi**2.5/pp/qq/sympy.sqrt(pp+qq)*KXYZAB(aa,bb,[X_A,Y_A,Z_A],[X_B,Y_B,Z_B])*KXYZAB(cc,dd,[X_C,Y_C,Z_C],[X_D,Y_D,Z_D])*BOYS(1,alpha*PQ2)
V_T_2_0000=2*sympy.pi**2.5/pp/qq/sympy.sqrt(pp+qq)*KXYZAB(aa,bb,[X_A,Y_A,Z_A],[X_B,Y_B,Z_B])*KXYZAB(cc,dd,[X_C,Y_C,Z_C],[X_D,Y_D,Z_D])*BOYS(2,alpha*PQ2)
V_T_3_0000=2*sympy.pi**2.5/pp/qq/sympy.sqrt(pp+qq)*KXYZAB(aa,bb,[X_A,Y_A,Z_A],[X_B,Y_B,Z_B])*KXYZAB(cc,dd,[X_C,Y_C,Z_C],[X_D,Y_D,Z_D])*BOYS(3,alpha*PQ2)
V_T_4_0000=2*sympy.pi**2.5/pp/qq/sympy.sqrt(pp+qq)*KXYZAB(aa,bb,[X_A,Y_A,Z_A],[X_B,Y_B,Z_B])*KXYZAB(cc,dd,[X_C,Y_C,Z_C],[X_D,Y_D,Z_D])*BOYS(4,alpha*PQ2)


# In[52]:


sympy.N(vxxyz.subs([(p,pp),(q,qq),(a,aa),(c,cc),(b,bb),(d,dd),
                    (XPQ,X_PQ),(XAB,X_AB),(XCD,X_CD),\
                    (YPQ,Y_PQ),(YAB,Y_AB),(YCD,Y_CD),\
                    (ZPQ,Z_PQ),(ZAB,Z_AB),(ZCD,Z_CD),\
                    (XPA,X_PA),(YPA,Y_PA),(ZPA,Z_PA),\
                    (YAB,Y_AB),(YAB,Y_AB),(ZAB,Z_AB),\
                    (TNS[0],V_T_0_0000),(TNS[1],V_T_1_0000),(TNS[2],V_T_2_0000),(TNS[3],V_T_3_0000),(TNS[4],V_T_4_0000)])).expand()


# In[53]:


sympy.N(vxyzz.subs([(q,qq),(p,pp),(a,aa),(c,cc),(b,bb),(d,dd),
                    (XPQ,X_PQ),(XAB,X_AB),(XCD,X_CD),\
                    (YPQ,Y_PQ),(YAB,Y_AB),(YCD,Y_CD),\
                    (ZPQ,Z_PQ),(ZAB,Z_AB),(ZCD,Z_CD),\
                    (XPA,X_PA),(YPA,Y_PA),(ZPA,Z_PA),\
                    (YAB,Y_AB),(YAB,Y_AB),(ZAB,Z_AB),\
                    (TNS[0],V_T_0_0000),(TNS[1],V_T_1_0000),(TNS[2],V_T_2_0000),(TNS[3],V_T_3_0000),(TNS[4],V_T_4_0000)])).expand()


# In[54]:


# f:=[ss|ss] -> (Differentiation) -> [xx|yz]
AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
z1,z2,z3,z4=symbols("z1 z2 z3 z4")
f=T0000(z1,z2,z3,z4,[AX,AY,AZ],[BX,BY,BZ],[CX,CY,CZ],[DX,DY,DZ])
#print(f)
sympy.N(f.subs([(z1,aa),(z2,bb),(z3,cc),(z4,dd)])\
                    .diff(AX).diff(BX).diff(CY).diff(DZ)\
                    .subs([
                    (AX,X_A),(AY,Y_A),(AZ,Z_A),\
                    (BX,X_B),(BY,Y_B),(BZ,Z_B),\
                    (CX,X_C),(CY,Y_C),(CZ,Z_C),\
                    (DX,X_D),(DY,Y_D),(DZ,Z_D)])/2/aa/2/bb/2/cc/2/dd)


# In[55]:


# f:=[ss|ss] -> (Differentiation) -> [xy|zz]
AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
z1,z2,z3,z4=symbols("z1 z2 z3 z4")
f=T0000(z1,z2,z3,z4,[AX,AY,AZ],[BX,BY,BZ],[CX,CY,CZ],[DX,DY,DZ])
#print(f)
sympy.N(f.subs([(z1,aa),(z2,bb),(z3,cc),(z4,dd)])\
                    .diff(AX).diff(BY).diff(CZ).diff(DZ)\
                    .subs([
                    (AX,X_A),(AY,Y_A),(AZ,Z_A),\
                    (BX,X_B),(BY,Y_B),(BZ,Z_B),\
                    (CX,X_C),(CY,Y_C),(CZ,Z_C),\
                    (DX,X_D),(DY,Y_D),(DZ,Z_D)])/2/aa/2/bb/2/cc/2/dd)

