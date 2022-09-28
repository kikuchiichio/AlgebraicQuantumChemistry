#!/usr/bin/env python
# coding: utf-8

# These Python programs compute the electronic structure of H3+ in the equilateral triangle. 
# 
# The atoms are indexed by A, B, and C, or a, b, and c.
# The STO-3G are determined by the parameters ZETA, COEFF, and EXPON. 
# 
# (1) SZ-SIM-H3P-RHF
#       For RHF computation.  
#       The variables (x,y,z) : the LCAO coefficients of 1s orbitals on the vertices.
#       The variable e : the orbital energy
#       The variable R: the length of the edges, fixed at a positive number.
# 
# (2) SZ-SIM-H3P-UHF
#       For UHF computation.  
#       The variables (x,y,z) : the LCAO coefficients of 1s orbitals on the vertices, for spin alpha
#       The variables e : the orbital energy, for spin alpha
#       The variables (u,v,w) : the LCAO coefficients of 1s orbitals on the vertices, for spin beta
#       The variables f : the orbital energy, for spin beta
#       The variable R: the length of the edges, fixed at a positive number.
# 
# (3) SZ-SIM-H3P-RHF-DeterminR, SZ-SIM-H3P-RHF-DeterminR-Version2
#       The variables (x,x,x) : the LCAO coefficients of 1s orbitals on the vertices.
#       The variable e : the orbital energy
#       The variable R: the length of the edges, to be optimized as well as other variables.
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
R=symbols("R",positive=True)
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


fzzzz=f.diff(AZ).diff(BZ).diff(CZ).diff(DZ)


# In[3]:


R_length=0.9/0.529177


# In[4]:


R_length


# In[5]:


V1111=0.0
V2111=0.0
V2121=0.0
V2211=0.0
V2221=0.0
V2222=0.0
N=3
COEF=[[1.0,0.0,0.0],[0.678914,0.430129,0.0],[0.444635,0.535328,0.154329]]
EXPON=[[0.270950,0.0,0.0],[0.151623,0.851819,0.0],[0.109818,0.405771,2.22766]]
AtomA=[0,0,0]
R_length=0.9/0.529177
AtomB=[R_length,0,0]
ZETA1=2.0925
ZETA2=1.24
ZETA1=ZETA2
A1=[0]*N
D1=[0]*N
A2=[0]*N
D2=[0]*N
PI=np.pi
for i in range(N):
    #print(i,EXPON[N-1][i],COEF[N-1][i],ZETA1,ZETA2)
    A1[i]=EXPON[N-1][i]*ZETA1**2
    D1[i]=COEF[N-1][i]*((2.0*A1[i]/PI)**0.75)
    A2[i]=EXPON[N-1][i]*(ZETA2**2)
    D2[i]=COEF[N-1][i]*((2.0*A2[i]/PI)**0.75)
for I in range(N):
    for J in range(N):
        for K in range(N):
            for L in range(N):
                RAP=A2[I]*R/(A2[I]+A1[J])
                RBP=R-RAP
                RAQ=A2[K]*R/(A2[K]+A1[L])
                RBQ=R-RAQ
                RPQ=RAP-RAQ
                RAP2=RAP*RAP
                RBP2=RBP*RBP
                RAQ2=RAQ*RAQ
                RBQ2=RBQ*RBQ
                RPQ2=RPQ*RPQ
                """
                V1111=V1111+TWOE(A1[I],A1[J],A1[K],A1[L],0.0,0.0,0.0)*D1[I]*D1[J]*D1[K]*D1[L]
                V2111=V2111+TWOE(A2[I],A1[J],A1[K],A1[L],R2,0.0,RAP2)*D2[I]*D1[J]*D1[K]*D1[L]
                V2121=V2121+TWOE(A2[I],A1[J],A2[K],A1[L],R2,R2,RPQ2)*D2[I]*D1[J]*D2[K]*D1[L]
                V2211=V2211+TWOE(A2[I],A2[J],A1[K],A1[L],0.0,0.0,R2)*D2[I]*D2[J]*D1[K]*D1[L]
                V2221=V2221+TWOE(A2[I],A2[J],A2[K],A1[L],0.0,R2,RBQ2)*D2[I]*D2[J]*D2[K]*D1[L]
                V2222=V2222+TWOE(A2[I],A2[J],A2[K],A2[L],0.0,0.0,0.0)*D2[I]*D2[J]*D2[K]*D2[L]
                """
                ca=A1[I]
                cb=A1[J]
                cc=A1[K]
                cd=A1[L]
                """
                V1111=V1111+T0000(A1[I],A1[J],A1[K],A1[L],AtomA,AtomA,AtomA,AtomA)*D1[I]*D1[J]*D1[K]*D1[L]
                V2111=V2111+T0000(A2[I],A1[J],A1[K],A1[L],AtomB,AtomA,AtomA,AtomA)*D2[I]*D1[J]*D1[K]*D1[L]
                V2121=V2121+T0000(A2[I],A1[J],A2[K],A1[L],AtomB,AtomA,AtomB,AtomA)*D2[I]*D1[J]*D2[K]*D1[L]
                V2211=V2211+T0000(A2[I],A2[J],A1[K],A1[L],AtomB,AtomB,AtomA,AtomA)*D2[I]*D2[J]*D1[K]*D1[L]
                V2221=V2221+T0000(A2[I],A2[J],A2[K],A1[L],AtomB,AtomB,AtomB,AtomA)*D2[I]*D2[J]*D2[K]*D1[L]
                V2222=V2222+T0000(A2[I],A2[J],A2[K],A2[L],AtomB,AtomB,AtomB,AtomB)*D2[I]*D2[J]*D2[K]*D2[L]
                """
                V1111=V1111+T0000(ca,cb,cc,cd,AtomA,AtomA,AtomA,AtomA)*D1[I]*D1[J]*D1[K]*D1[L]
                V2111=V2111+T0000(ca,cb,cc,cd,AtomB,AtomA,AtomA,AtomA)*D2[I]*D1[J]*D1[K]*D1[L]
                V2121=V2121+T0000(ca,cb,cc,cd,AtomB,AtomA,AtomB,AtomA)*D2[I]*D1[J]*D2[K]*D1[L]
                V2211=V2211+T0000(ca,cb,cc,cd,AtomB,AtomB,AtomA,AtomA)*D2[I]*D2[J]*D1[K]*D1[L]
                V2221=V2221+T0000(ca,cb,cc,cd,AtomB,AtomB,AtomB,AtomA)*D2[I]*D2[J]*D2[K]*D1[L]
                V2222=V2222+T0000(ca,cb,cc,cd,AtomB,AtomB,AtomB,AtomB)*D2[I]*D2[J]*D2[K]*D2[L]
print(sympy.N(V1111),sympy.N(V2111),sympy.N(V2121),sympy.N(V2211),sympy.N(V2221),sympy.N(V2222))


# In[25]:


import numpy

TTlist=dict()

def indexijkl(i,j,k,l):
    A=[(i,j,k,l),(j,i,k,l),(i,j,l,k),(j,i,l,k)]
    A+=[(k,l,i,j),(l,k,i,j),(k,l,j,i),(l,k,j,i)]
    A.sort()
    return A[0]


R_length=0.9/0.529177
R=symbols("R")
R_length=R
atomA=[0,0,0]
atomB=[R_length,0,0]
atomC=[R_length*1/2,R_length*sympy.sqrt(3)/2,0]
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
atoms=[atomA,atomB,atomC]
ZETAS=[ZETA1,ZETA2,ZETA3]
ZS=[ZA,ZB,ZC]
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

TT=[[[[0 for _ in range(3)] for _  in range(3)] for _ in range(3)] for _ in range(3)]
for ID in range(3):
    for JD in range(3):
        for KD in range(3):
            for LD in range(3):
                RA=[AX,AY,AZ]=atoms[ID]
                RB=[BX,BY,BZ]=atoms[JD]
                RC=[CX,CY,CZ]=atoms[KD]
                RD=[DX,DY,DZ]=atoms[LD]
                if TTlist.get(indexijkl(ID,JD,KD,LD))==None:
                    N=3
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

                    N=3
                    V=0
                    for I in range(N):            
                        for J in range(N):
                            for K in range(N):
                                for L in range(N):
                                    ca=CA[I]
                                    cb=CB[J]
                                    cc=CC[K]
                                    cd=CD[L]
                                    #print(ca,cb,cc,cd,sympy.N(T0000(ca,cb,cc,cd,RA,RB,RC,RD)))

                                    V=V+T0000(ca,cb,cc,cd,RA,RB,RC,RD)*DA[I]*DB[J]*DC[K]*DD[L]
                    V=sympy.N(V)
                    print(RA,RB,RC,RD,V)
                    TTlist[indexijkl(ID,JD,KD,LD)]=V




# In[7]:


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

VAB_C=[[[0 for _ in range(3)] for _  in range(3)] for _ in range(3)]
for ID in range(3):
    for JD in range(3):
        for KD in range(3):
                RA=[AX,AY,AZ]=atoms[ID]
                RB=[BX,BY,BZ]=atoms[JD]
                RC=[CX,CY,CZ]=atoms[KD]
                ZC=ZS[KD]
                N=3
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
                    
                N=3
                V=0
                for I in range(N):            
                    for J in range(N):
                            ca=CA[I]
                            cb=CB[J]
                            #V000000(a,b,RA,RB,RC)
                            V=V+V000000(ca,cb,RA,RB,RC)*DA[I]*DB[J]*(-ZC)
                V=sympy.N(V)
                #print(RA,RB,RC,RD,V)
                VAB_C[ID][JD][KD]=V

SAB=[[0 for _ in range(3)] for _  in range(3)]              
for ID in range(3):
    for JD in range(3):
        RA=[AX,AY,AZ]=atoms[ID]
        RB=[BX,BY,BZ]=atoms[JD]
        for i in range(N):
            #print(i,EXPON[N-1][i],COEF[N-1][i],ZETA1,ZETA2)
            CA[i]=EXPON[N-1][i]*ZETAS[ID]**2
            DA[i]=COEF[N-1][i]*((2.0*CA[i]/PI)**0.75)
            CB[i]=EXPON[N-1][i]*(ZETAS[JD]**2)
            DB[i]=COEF[N-1][i]*((2.0*CB[i]/PI)**0.75)

        N=3
        V=0
        for I in range(N):            
            for J in range(N):
                        ca=CA[I]
                        cb=CB[J]
                        #V000000(a,b,RA,RB,RC)
                        V=V+S000000(ca,cb,RA,RB)*DA[I]*DB[J]
        V=sympy.N(V)
        #print(RA,RB,V)
        SAB[ID][JD]=V
        
KAB=[[0 for _ in range(3)] for _  in range(3)]              
for ID in range(3):
    for JD in range(3):
        RA=[AX,AY,AZ]=atoms[ID]
        RB=[BX,BY,BZ]=atoms[JD]
        for i in range(N):
            #print(i,EXPON[N-1][i],COEF[N-1][i],ZETA1,ZETA2)
            CA[i]=EXPON[N-1][i]*ZETAS[ID]**2
            DA[i]=COEF[N-1][i]*((2.0*CA[i]/PI)**0.75)
            CB[i]=EXPON[N-1][i]*(ZETAS[JD]**2)
            DB[i]=COEF[N-1][i]*((2.0*CB[i]/PI)**0.75)

        N=3
        V=0
        for I in range(N):            
            for J in range(N):
                        ca=CA[I]
                        cb=CB[J]
                        #V000000(a,b,RA,RB,RC)
                        V=V+K000000(ca,cb,RA,RB)*DA[I]*DB[J]
        V=sympy.N(V)
        #print(RA,RB,V)
        KAB[ID][JD]=V


# In[26]:


def SCF_SIMBOL3(IOP,N,R,ZETA1,ZETA2,ZA,ZB):
#
# PREPARES THE ANALYTIC FORMULA OF THE TOTAL ENERGY.
#


    P=[[0 for _ in range(3)] for _ in range(3)]
    P2=[[0 for _ in range(3)] for _ in range(3)]
    G=[[0 for _ in range(3)] for _ in range(3)]
    G2=[[0 for _ in range(3)] for _ in range(3)]
    F=[[0 for _ in range(3)] for _ in range(3)]
    F2=[[0 for _ in range(3)] for _ in range(3)]
    H=[[0 for _ in range(3)] for _ in range(3)]
    x,y,z,u,v,w=symbols("x y z u v w")

    PI=np.pi
    CRIT=1.0e-4
    MAXIT=25
    ITER=0
    for I in range(3):
        for J in range(3):
            P[I][J]=0.
    P[0][0]=x*x*2
    P[0][1]=x*x*2
    P[0][2]=x*x*2
    P[1][0]=x*x*2
    P[1][1]=x*x*2
    P[1][2]=x*x*2
    P[2][0]=x*x*2
    P[2][1]=x*x*2
    P[2][2]=x*x*2

    for I in range(3):
        for J in range(3):
            G[I][J]=0
            G2[I][J]=0
            for K in range(3):
                for L in range(3):
                    #print(I,J,K,L,P[K][L],TT[I][J][K][L],TT[I][L][J][K],G[I][J])
                    #print(I,J,K,L)
                    G[I][J]+=P[K][L]*TTlist.get(indexijkl(I,J,K,L))-0.5*P[K][L]*TTlist.get(indexijkl(I,L,J,K))
    H=[[0 for _ in range(3)] for _ in range(3)]
    for I in range(3):
        for J in range(3):
            H[I][J]=KAB[I][J]
            for K in range(3):
                H[I][J]+=VAB_C[I][J][K]

    for i in range(3):
        for j in range(3):
            F[i][j]=H[i][j]+G[i][j]

    EN=0
    for i in range(3):
        for j in range(3):
            EN+=0.5*P[i][j]*(H[i][j]+F[i][j])
    ENT=EN
    for i in range(3):
        for j in range(i+1,3):
            RPQ=[(c1-c2)**2 for c1,c2 in zip(atoms[i],atoms[j])]
            RPQ2=sum(RPQ)
            #print(RPQ2)
            ENT+=ZS[i]*ZS[j]/R
            


    return EN,ENT,F,F2,H,P
x,y,z,u,v,w=symbols("x y z u v w")
EN,ENT,FM,FM2,HM,PM=SCF_SIMBOL3(IOP,N,R,ZETA1,ZETA2,ZA,ZB)


# In[27]:


ENV=0
for i in range(3):
    for j in range(i+1,3):
        RPQ=[(c1-c2)**2 for c1,c2 in zip(atoms[i],atoms[j])]
        RPQ2=sum(RPQ)
        #print(RPQ2)
        ENV+=ZS[i]*ZS[j]/R
ENV


# In[28]:


from sympy import series


# In[29]:


ENT=ENT.expand()


# In[30]:


ENS=series(ENT,R,1.7)


# In[31]:


def GetNormS(vec,SAB):
    V=0
    for i in range(len(vec)):
        for j in range(len(vec)):
            V+=vec[i]*SAB[i][j]*vec[j]
    return V

e,f=symbols("e f")
#OBJ=ENT-2*e*(GetNormS([x,y,z],SAB)-1)


# In[32]:


EPART=ENS.removeO().expand()


# In[33]:


SPART=series(GetNormS([x,x,x],SAB)-1,R,1.7).removeO().expand()


# In[34]:


OBJE=sympy.N(EPART-2*e*SPART)
OBJ=OBJE.expand()


# In[35]:


sympy.poly(sympy.N(OBJ))


# In[36]:


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


# In[37]:


getF


# In[38]:


str(ENTS3.args[1:])


# In[39]:


getF


# In[40]:


getF
Fargs=str(ENTS3.args[1:])
stringQ='option(noredefine);LIB "solve.lib";option(redSB);\n'
stringQ+='ring r=0,'+Fargs+',dp;\n'+'poly OBJ='+str(getF)+';\n'
stringQ+='list diffs;\n'
stringQ+='for(int i=1;i<=nvars(r); i=i+1){diffs=insert(diffs,diff(OBJ,var(i)));}\n'
stringQ+='ideal I=0;\n'
stringQ+='for(int i=1;i<=nvars(r); i=i+1){I=I+diff(OBJ,var(i));}\n'
stringQ+='print(I);'
stringQ+='ideal SI=std(I);\n'
stringQ+='print(SI);'
stringQ+='ring s=0,'+Fargs+',lp;\n'
stringQ+='setring s;\n'
stringQ+='ideal j=fglm(r,SI);\n'
stringQ+='def R=triang_solve(j,50);\n'
stringQ+='setring R;rlist;'
stringQ+='poly OBJ=fetch(r,OBJ);ideal I=fetch(r,I);OBJ;for (i=1;i<=size(rlist);i=i+1){list substv;poly OBJ2=OBJ;for (int k=1; k<=nvars(R); k=k+1){ OBJ2=subst(OBJ2,var(k), rlist[i][k]);}substv=insert(substv,OBJ2);for (int l=1;l<size(I);l=l+1){poly OBJ2=I[l]; for (int k=1; k<=nvars(R); k=k+1) {  OBJ2=subst(OBJ2,var(k), rlist[i][k]); } substv=insert(substv,OBJ2);}print(substv);}'

text_file = open("SCRIPT.txt", "w")
on = text_file.write(stringQ)
 
#close file
text_file.close()


# In[41]:


import subprocess
cp = subprocess.run("wsl Singular<SCRIPT.txt", shell=True,capture_output=True,text=True)
print("stdout:", cp.stdout)
print("stderr:", cp.stderr)

