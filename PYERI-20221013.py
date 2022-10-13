#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


fzzzz=f.diff(AZ).diff(BZ).diff(CZ).diff(DZ)


# In[ ]:


R_length=0.9/0.129177


# In[ ]:


R


# In[ ]:


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


# In[ ]:


import numpy
R_length=0.9/0.529177
R_length=1.46
atomA=[0,0,0]
atomB=[R_length,0,0]
atomC=[R_length*1/2,R_length*np.sqrt(3)/2,0]
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
TT=[[[[0 for _ in range(3)] for _  in range(3)] for _ in range(3)] for _ in range(3)]
for ID in range(3):
    for JD in range(3):
        for KD in range(3):
            for LD in range(3):
                RA=[AX,AY,AZ]=atoms[ID]
                RB=[BX,BY,BZ]=atoms[JD]
                RC=[CX,CY,CZ]=atoms[KD]
                RD=[DX,DY,DZ]=atoms[LD]
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
                #print(RA,RB,RC,RD,V)
                TT[ID][JD][KD][LD]=V




# In[ ]:


TT[2][2][2][2]


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
                        for K in range(N):
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


# In[ ]:


def FORMG():
#
# Calculates G matrix from density matrix and two-electron integrals
#
    for I in range(3):
        for J in range(3):
            G[I][J]=0
            for K in range(3):
                for L in range(3):
                    #print(I,J,K,L,P[K][L],TT[I][J][K][L],TT[I][L][J][K],G[I][J])
                    G[I][J]+=P[K][L]*(TT[I][J][K][L]-0.5*TT[I][L][J][K])
    #print("G",G)
    return 



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
    P[0][0]=x*x
    P[0][1]=x*y
    P[0][2]=x*z
    P[1][0]=y*x
    P[1][1]=y*y
    P[1][2]=y*z
    P[2][0]=z*x
    P[2][1]=z*y
    P[2][2]=z*z
    P2[0][0]=u*u
    P2[0][1]=u*v
    P2[0][2]=u*w
    P2[1][0]=v*u
    P2[1][1]=v*v
    P2[1][2]=v*w
    P2[2][0]=w*u
    P2[2][1]=w*v
    P2[2][2]=w*w
    for I in range(3):
        for J in range(3):
            G[I][J]=0
            G2[I][J]=0
            for K in range(3):
                for L in range(3):
                    #print(I,J,K,L,P[K][L],TT[I][J][K][L],TT[I][L][J][K],G[I][J])
                    #print(I,J,K,L)
                    G[I][J]+=(P[K][L]+P2[K][L])*TT[I][J][K][L]-P[K][L]*TT[I][L][J][K]
                    G2[I][J]+=(P[K][L]+P2[K][L])*TT[I][J][K][L]-P2[K][L]*TT[I][L][J][K]
    H=[[0 for _ in range(3)] for _ in range(3)]
    for I in range(3):
        for J in range(3):
            H[I][J]=KAB[I][J]
            for K in range(3):
                H[I][J]+=VAB_C[I][J][K]

    for i in range(3):
        for j in range(3):
            F[i][j]=H[i][j]+G[i][j]
            F2[i][j]=H[i][j]+G2[i][j]
    EN=0
    for i in range(3):
        for j in range(3):
            EN+=0.5*P[i][j]*(H[i][j]+F[i][j])
            EN+=0.5*P2[i][j]*(H[i][j]+F2[i][j])

    ENT=EN
    for i in range(3):
        for j in range(i+1,3):
            RPQ=[(c1-c2)**2 for c1,c2 in zip(atoms[i],atoms[j])]
            RPQ2=sum(RPQ)
            #print(RPQ2)
            ENT+=ZS[i]*ZS[j]/np.sqrt(RPQ2)
    return EN,ENT,F,F2,H,P
x,y,z,u,v,w=symbols("x y z u v w")
EN,ENT,FM,FM2,HM,PM=SCF_SIMBOL3(IOP,N,R,ZETA1,ZETA2,ZA,ZB)


# In[ ]:


ENT


# In[ ]:


H=[[0 for _ in range(3)] for _ in range(3)]
for I in range(3):
    for J in range(3):
        H[I][J]=KAB[I][J]
        for K in range(3):
            H[I][J]+=VAB_C[I][J][K]
            
H


# In[ ]:


PMV=[[0,0,0],[0,0,0],[0,0,0]]
for i in range(3):
    for j in range(3):
        PMV[i][j]=PM[i][j].subs([(x,1.1851214044319374728682484869323596084452768449552),(y,-1.1662974420807114976749839300596884444219146118261),      (z,-0.033386336086772973894315489691480653627247197124287)])
        
PMV


# In[ ]:


HMV=[[0,0,0],[0,0,0],[0,0,0]]
for i in range(3):
    for j in range(3):
        HMV[i][j]=HM[i][j].subs([(x,1.1851214044319374728682484869323596084452768449552),(y,-1.1662974420807114976749839300596884444219146118261),      (z,-0.033386336086772973894315489691480653627247197124287)])
        
HMV


# In[ ]:


FMV=[[0,0,0],[0,0,0],[0,0,0]]
for i in range(3):
    for j in range(3):
        FMV[i][j]=FM[i][j].subs([(x,1.1851214044319374728682484869323596084452768449552),(y,-1.1662974420807114976749839300596884444219146118261),      (z,-0.033386336086772973894315489691480653627247197124287)])
        
FMV


# In[ ]:


def GetNormS(vec,SAB):
    V=0
    for i in range(len(vec)):
        for j in range(len(vec)):
            V+=vec[i]*SAB[i][j]*vec[j]
    return V

e,f=symbols("e f")
OBJ=EN-e*(GetNormS([x,y,z],SAB)-1)-f*(GetNormS([u,v,w],SAB)-1)


# In[ ]:


FM


# In[ ]:


def GetMdotV(M,S,V,e):
    R=[0]*len(V)
    for i in range(len(V)):
        for j in range(len(V)):
            R[i]+=M[i][j]*V[j]-e*S[i][j]*V[j]
    return R

FMdotxy=GetMdotV(FM,SAB,[x,y,z],e)
FM2dotxy=GetMdotV(FM2,SAB,[u,v,w],f)
2*FMdotxy[0].expand()


# In[ ]:


from scipy.linalg import eigh
def fvector(vector):
    V=[[0 for i in range(len(vector))] for j in range(len(vector))]

    
    for i in range(len(vector)):
        for j in range(len(vector[i])):
                       V[i][j] = float(vector[i][j])
    return np.array(V)
HM=fvector(np.array(H))
SM=fvector(np.array(SAB))

eigvals, eigvecs = eigh(HM,SM, eigvals_only=False)
#eigvals, eigvecs = eigh(np.array([[0,1],[1,0]]))
eigvals,eigvecs


# In[ ]:


AL=list(set(H[0]+H[1]+H[2]))


# In[ ]:


EN.subs([(x,1.1851214044319374728682484869323596084452768449552),(y,-1.1662974420807114976749839300596884444219146118261),      (z,-0.033386336086772973894315489691480653627247197124287)])


# In[ ]:


sympy.poly(sympy.N(OBJ))


# In[ ]:


def getINT(x,N):
#
#  ROWND DOWN x*N AFTER THE DECIMAL POINT and get AN INTEGER M
#  THEN X is approximated by M/N. 
#
    return (int(np.floor(x*N)))
OBJ=OBJ.expand()
ENTS3=sympy.poly(sympy.N(OBJ))
AT=ENTS3.terms()
getF=0
for tm in AT:
    p0,p1,p2,p3,p4,p5,p6,p7=tm[0]
    cf=tm[1]
    #print(p0,p1,p2,cf)
    getF+=x**p0* y**p1* z**p2* u**p3* v**p4* w**p5* e**p5* f**p7*getINT(cf,10000)


# In[ ]:


getF


# In[ ]:


getF

stringQ='LIB "solve.lib";option(redSB);\n'
stringQ+='ring r=0,(x,y,z,u,v,w,e,f),dp;\n'+'poly OBJ='+str(getF)+';\n'
stringQ+='list diffs;\n'
stringQ+='for(int i=1;i<=nvars(r); i=i+1){diffs=insert(diffs,diff(OBJ,var(i)));}\n'
stringQ+='ideal I=0;\n'
stringQ+='for(int i=1;i<=nvars(r); i=i+1){I=I+diff(OBJ,var(i));}\n'
stringQ+='print(I);'
stringQ+='ideal SI=std(I);\n'
stringQ+='print(SI);'
stringQ+='ring s=0,(x,y,z,u,v,w,e,f),lp;\n'
stringQ+='setring s;\n'
stringQ+='ideal j=fglm(r,SI);\n'
stringQ+='def R=triang_solve(j,50);\n'
stringQ+='setring R;rlist;'

text_file = open("SCRIPT.txt", "w")
on = text_file.write(stringQ)
 
#close file
text_file.close()


# In[ ]:


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

def INTGRL_SIMBOL(IOP,N,R,ZETA1,ZETA2,ZA,ZB):
#
#  CALCULATES ALL THE BASIC INTEGRALS NEEDED FOR SCF CALCULATION
#
    global S12,T11,T12,T22,V11A,V12A,V22A,V11B,V12B,V22B,V1111,V2111,V2121,V2211,V2221,V2222
    COEF=[[1.0,0.0,0.0],[0.678914,0.430129,0.0],[0.444635,0.535328,0.154329]]
    EXPON=[[0.270950,0.0,0.0],[0.151623,0.851819,0.0],[0.109818,0.405771,2.22766]]
    R2=R*R
    N=3
    A1=[0 for I in range(N)]
    D1=[0 for I in range(N)]
    A2=[0 for I in range(N)]
    D2=[0 for I in range(N)]
    PI=np.pi

    for i in range(N):
        #print(i,EXPON[N-1][i],COEF[N-1][i],ZETA1,ZETA2)
        A1[i]=EXPON[N-1][i]*ZETA1**2
        D1[i]=COEF[N-1][i]*((2.0*A1[i]/PI)**0.75)
        A2[i]=EXPON[N-1][i]*(ZETA2**2)
        D2[i]=COEF[N-1][i]*((2.0*A2[i]/PI)**0.75)

    #print("A1",A1,"D1",D1,"A2",A2,"D2",D2)
        
    S12=0.0
    T11=0.0
    T12=0.0
    T22=0.0
    V11A=0.0
    V12A=0.0
    V22A=0.0
    V11B=0.0
    V12B=0.0
    V22B=0.0
    V1111=0.0
    V2111=0.0
    V2121=0.0
    V2211=0.0
    V2221=0.0
    V2222=0.0
    
    for I in range(N):
        for J in range(N):
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
                    V1111=V1111+TWOE(A1[I],A1[J],A1[K],A1[L],0.0,0.0,0.0)*D1[I]*D1[J]*D1[K]*D1[L]
                    V2111=V2111+TWOE(A2[I],A1[J],A1[K],A1[L],R2,0.0,RAP2)*D2[I]*D1[J]*D1[K]*D1[L]
                    V2121=V2121+TWOE(A2[I],A1[J],A2[K],A1[L],R2,R2,RPQ2)*D2[I]*D1[J]*D2[K]*D1[L]
                    V2211=V2211+TWOE(A2[I],A2[J],A1[K],A1[L],0.0,0.0,R2)*D2[I]*D2[J]*D1[K]*D1[L]
                    V2221=V2221+TWOE(A2[I],A2[J],A2[K],A1[L],0.0,R2,RBQ2)*D2[I]*D2[J]*D2[K]*D1[L]
                    V2222=V2222+TWOE(A2[I],A2[J],A2[K],A2[L],0.0,0.0,0.0)*D2[I]*D2[J]*D2[K]*D2[L]

    

def COLLECT(IOP,N,R,ZETA1,ZETA2,ZA,ZB):
#
# THIS TAKES THE BASIC INTEGRALS FROM COMMON AND ASSEMBLES THE
# RELEVENT MATRICES, THAT IS S,H,X,XT, AND TWO-ELECTRON INTEGRALS
#
    global H,S,X,TT,XT
    H[0][0]=T11+V11A+V11B
    H[0][1]=T12+V12A+V12B
    H[1][0]=H[0][1]
    H[1][1]=T22+V22A+V22B

    S[0][0]=1.
    S[0][1]=S12
    S[1][0]=S[0][1]
    S[1][1]=1.


    TT[0][0][0][0]=V1111
    TT[1][0][0][0]=V2111
    TT[0][1][0][0]=V2111
    TT[0][0][1][0]=V2111
    TT[0][0][0][1]=V2111
    TT[1][0][1][0]=V2121
    TT[0][1][1][0]=V2121
    TT[1][0][0][1]=V2121
    TT[0][1][0][1]=V2121
    TT[1][1][0][0]=V2211
    TT[0][0][1][1]=V2211
    TT[1][1][1][0]=V2221
    TT[1][1][0][1]=V2221
    TT[1][0][1][1]=V2221
    TT[0][1][1][1]=V2221
    TT[1][1][1][1]=V2222


    return

def MATOUT(A,IM,IN,M,N,LABEL):
    print(LABEL)
    print(A)
    return

def MULT(A,B,C,IM,M):
    for i in range(2):
        for j in range(2):
            C[i][j]=0
            for k in range(2):
                C[i][j]+=A[i][k]*B[k][j]
    return 

def FORMG():
#
# Calculates G matrix from density matrix and two-electron integrals
#
    global G,P,TT
    for I in range(2):
        for J in range(2):
            G[I][J]=0
            for K in range(2):
                for L in range(2):
                    #print(I,J,K,L,P[K][L],TT[I][J][K][L],TT[I][L][J][K],G[I][J])
                    G[I][J]+=P[K][L]*(TT[I][J][K][L]-0.5*TT[I][L][J][K])
    #print("G",G)
    return 



def SCF_SIMBOL(IOP,N,R,ZETA1,ZETA2,ZA,ZB):
#
# PREPARES THE ANALYTIC FORMULA OF THE TOTAL ENERGY.
#
    global G,C,FPRIME,CPRIME

    PI=np.pi
    CRIT=1.0e-4
    MAXIT=25
    ITER=0
    for I in range(2):
        for J in range(2):
            P[I][J]=0.
    P[0][0]=x*x*2
    P[0][1]=x*y*2
    P[1][0]=x*y*2
    P[1][1]=y*y*2
    
    MAXIT=1
    for ITER in range(MAXIT):
        
        FORMG()
        for i in range(2):
            for j in range(2):
                F[i][j]=H[i][j]+G[i][j]
        EN=0
        for i in range(2):
            for j in range(2):
                EN+=0.5*P[i][j]*(H[i][j]+F[i][j])

    ENT=EN+ZA*ZB/R

    return EN,ENT

#
# COMPUTES THE ELECTRON ENERGY (EN) and THE TOTAL ENERGY(ENT)
#
INTGRL_SIMBOL(IOP,N,R,ZETA1,ZETA2,ZA,ZB)
COLLECT(IOP,N,R,ZETA1,ZETA2,ZA,ZB)
EN,ENT=SCF_SIMBOL(IOP,N,R,ZETA1,ZETA2,ZA,ZB)

from sympy import Symbol, cos, series

#
print("CHECK THE TOTAL ENERGY: BY THIS SUBSTITUTION WE'LL GET THE VALUE CLOSE TO -2.86...")
#
print(ENT.subs([(x,-0.80191751),(y,-0.33680049),(R,1.463)]))

# TAYLOR EXPANSION OF THE TOTAL ENERGY AROUNR R0=1.5
ENTS=series(expand(ENT),R,1.5)
ENTS2=expand(ENTS.removeO())
print("CHECK THE TOTAL ENERGY: BY THIS SUBSTITUTION WE'LL GET THE VALUE CLOSE TO -2.86...")
print(sympy.N(ENTS2.subs([(x,-0.80191751),(y,-0.33680049),(R,1.463)])))

#
# THE NORMALIZARION CONDITION OF THE WAVE FUNCTION
#
W=[0,0]
W[0]=S[0][0]*x+S[0][1]*y
W[1]=S[1][0]*x+S[1][1]*y
NORMC=expand(x*W[0]+y*W[1])

#
# THE NORMALIZATION CONDITION AFTER TAYLOR EXPANSION
#
NORMCS=expand(series(NORMC,R,1.5).removeO())
print("CHECK THE NOTMALIZATION: BY THIS SUBSTITUTION WE'LL GET THE VALUE CLOSE TO 1...")
print(NORMCS.subs([(x,-0.80191751),(y,-0.33680049),(R,1.463)]))

#
# TURN THE POLYNOMIAL ENERGY FUNCTIONAL INTO A SYMPY-POLYNOMIAL
#
ENTS3=sympy.poly(sympy.N(ENTS2))

#
# GET THE OBJECTIVE FUNTION WITH INTEGER COEFFICIENTS
#

def getINT(x,N):
#
#  ROWND DOWN x*N AFTER THE DECIMAL POINT and get AN INTEGER M
#  THEN X is approximated by M/N. 
#
    return (int(np.floor(x*N)))

AT=ENTS3.terms()
getF=0
for tm in AT:
    p0,p1,p2=tm[0]
    cf=tm[1]
    #print(p0,p1,p2,cf)
    getF+=x**p0*y**p1*R**p2*getINT(cf,10000)

AT=sympy.poly(NORMCS-1).terms()
getNS=0
for tm in AT:
    p0,p1,p2=tm[0]
    cf=tm[1]
    #print(p0,p1,p2,cf)
    getNS+=x**p0*y**p1*R**p2*getINT(cf,10000)

print((NORMCS-1).subs([(x,-0.80191751),(y,-0.33680049),(R,1.463)]))
print(ENTS3.subs([(x,-0.80191751),(y,-0.33680049),(R,1.463)]))
print(getNS.subs([(x,-0.80191751),(y,-0.33680049),(R,1.463)]))

e=symbols("e")

#
# GET THE OBJECTIVE FUNCTION
#
OF=getF-2*e*getNS

#
# PREPARES A Singular SCRIPT. 
#
#  If you are a user of Singular, you simply do as follows.
#
#  >> Singular < SCRIPT.txt
#
stringQ='LIB "solve.lib";option(redSB);\n'
stringQ+='ring r=0,(x,y,e,R),dp;\n'+'poly OBJ='+str(OF)+';\n'
stringQ+='list diffs;\n'
stringQ+='for(int i=1;i<=nvars(r); i=i+1){diffs=insert(diffs,diff(OBJ,var(i)));}\n'
stringQ+='poly fR=100*R-146;\n'
stringQ+='ideal I=fR;\n'
stringQ+='for(int i=1;i<=nvars(r)-1; i=i+1){I=I+diff(OBJ,var(i));}\n'
stringQ+='ideal SI=std(I);\n'
stringQ+='ring s=0,(x,y,e,R),lp;\n'
stringQ+='setring s;\n'
stringQ+='ideal j=fglm(r,SI);\n'
stringQ+='def R=triang_solve(j,10);\n'
stringQ+='setring R;rlist;'

text_file = open("SCRIPT.txt", "w")
 
#write string to file
on = text_file.write(stringQ)
 
#close file
text_file.close()

stringQ

#
# GET THE DIFFERENTIALS OF THE OBJECTIVE FUNCTION.
#
DS=[]
for t in [x,y,e,R]:
    DS.append(sympy.diff(OF,t))
index=0
for i in DS:
    print("poly f"+str(index)+"=",i,";")
    index+=1


# In[ ]:



#
#  sizeI : size of "requested_keys" BEFORE VERTICAL RECURSION 
#  sizeE : size of "requested_keys" AFTER  VERTICAL RECURSION
#  
#  If we do not limit the range of the indices correctly,
#  always the requested but not computed integrals remain, 
#  
#  We try to compute the requested integrals in a succesive repetition.
#  if sizeI == sizeE, the computation is satulated and we termintate it
#
print("VERTICAL")
sizeI=0
sizeE=1
LOOP=0
while (sizeI!=sizeE):
for N in range(6):
    for i1 in range(4):
        for i2 in range(4):
            for i3 in range(4):
                #print(i1,i2,i3)
                j1=j2=j3=k1=k2=k3=l1=l2=l3=0
                VERTICAL("X",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
                j2=j3=k1=k2=k3=l1=l2=l3=0
                VERTICAL("Y",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
                j2=j3=k1=k2=k3=l1=l2=l3=0
                VERTICAL("Z",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)

#for N in range(6):
#    for i1 in range(4):
#        for i2 in range(4):
#            for i3 in range(4):
#                #print(i1,i2,i3)
#                j1=j2=j3=k1=k2=k3=l1=l2=l3=0
#                VERTICAL("X",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
#                j2=j3=k1=k2=k3=l1=l2=l3=0
#                VERTICAL("Y",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
#                j2=j3=k1=k2=k3=l1=l2=l3=0
#                VERTICAL("Z",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
j1=j2=j3=k1=k2=k3=l1=l2=l3=0
i1=0
i2=i3=1
for N in range(6):
    VERTICAL("X",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
j1=j2=j3=k1=k2=k3=l1=l2=l3=0
i2=0
i1=i3=1
for N in range(6):
    VERTICAL("Y",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
j1=j2=j3=k1=k2=k3=l1=l2=l3=0
i3=0
i1=i2=1
for N in range(6):
    VERTICAL("Z",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)


sizeI=len(requested_keys.keys())
for keyr in integrals_all.keys():
    getV=requested_keys.get(keyr)
    if getV!=None:
        requested_keys.pop(keyr)
sizeE=len(requested_keys.keys())

print("LOOP:",LOOP,sizeI,sizeE)
LOOP+=1


# In[ ]:


integrals_all.get((0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0))


# In[ ]:


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
        integrals_all[keyn]= DQC*data0+alpha/q*data1+i_d/2/(p+q)*data2        +k_d/2/q*(data3-alpha/q*data4)
    #else:
        #print(keyn,"NOT WRITTEN")
        #print(key0,key1,key2,"key3=",key3)
        #print(data0,data1,data2,data3)
        


# In[ ]:



requested_keys=dict()
sizeI=0
sizeE=1
#
#  sizeI : size of "requested_keys" BEFORE VERTICAL RECURSION 
#  sizeE : size of "requested_keys" AFTER  VERTICAL RECURSION
#  
#  If we do not limit the range of the indices correctly,
#  always the requested but not computed integrals remain, 
#  
#  We try to compute the requested integrals in a succesive repetition.
#  if sizeI == sizeE, the computation is satulated and we termintate it
#
print("HORIZONTAL")
LOOP=0
while (sizeI!=sizeE):
    for N in range(5):
j1=j2=j3=k1=k2=k3=l1=l2=l3=0
itr_i123=list(product(range(3), repeat=3))
itr_k123=list(product(range(3), repeat=3))
for i123 in itr_i123:
    j1=j2=j3=l1=l2=l3=0
    i1,i2,i3=i123
    for k123 in itr_k123:
        k1,k2,k3=k123
        HORIZONTAL("X",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        HORIZONTAL("Y",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        HORIZONTAL("Z",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
removable=dict()
for a in requested_keys.keys():
    if integrals_all.get(a)!=None:
        removable[a]=a
sizeI=len(requested_keys.keys())
for keyr in integrals_all.keys():
    getV=requested_keys.get(keyr)
    if getV!=None:
        requested_keys.pop(keyr)
sizeE=len(requested_keys.keys())
print(LOOP,sizeI,sizeE)
LOOP+=1





#for a in requested_keys.keys():
#    if integrals_all.get(a)==None:
#        print(a,"None")
#    else:
#        print(a)





#
#  CHECK THE REQUESTED KEYS. IF WE MISS SOMETHING IN THE NEEDED RANGE, WE TRY TO COMPUTE THEM BY "HORIZONTAL AGAIN".
#  IF THERE IS NO OUTPUT HERE, WE HAVE GATHERED ALL REQUIRED INTEGRALS.
#
keyslist=[kr for kr in requested_keys.keys()]

for kr in keyslist:
    N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=list(kr) 
    if max([i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3])<=2:
#print(kr)
N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=kr
integrals_all.get(kr)
#HORIZONTAL("X",N,i1,i2,i3,j1,j2,j3,k1-1,k2,k3,l1,l2,l3)
#HORIZONTAL("Y",N,i1,i2,i3,j1,j2,j3,k1,k2-1,k3,l1,l2,l3)
#HORIZONTAL("Z",N,i1,i2,i3,j1,j2,j3,k1,k2,k3-1,l1,l2,l3)

for kr in keyslist:
    N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=list(kr) 
    if max([i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3])<=2:
N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=kr
integrals_all.get(kr)
#        HORIZONTAL("X",N,i1,i2,i3,j1,j2,j3,k1-1,k2,k3,l1,l2,l3)
#        HORIZONTAL("Y",N,i1,i2,i3,j1,j2,j3,k1,k2-1,k3,l1,l2,l3)
#        HORIZONTAL("Z",N,i1,i2,i3,j1,j2,j3,k1,k2,k3-1,l1,l2,l3)




print(len(requested_keys.keys()))
for keyr in integrals_all.keys():
    getV=requested_keys.get(keyr)
    if getV!=None:
requested_keys.pop(keyr)
print(len(requested_keys.keys()))
keyslist=[kr for kr in requested_keys.keys()]
#for kr in keyslist:
#    N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=list(kr) 
#    if max([i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3])<=1 and min([i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3])>=0:
#        print(kr)
#        N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=kr





requested_keys_save=copy.deepcopy(requested_keys)


# In[ ]:


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


# In[ ]:





# In[ ]:


print("TRANSFER1 & 2")
requested_keys=dict()
sizeI=0
sizeE=1
counter=0
print(len(requested_keys.keys()))
while (sizeI!=sizeE):
    for N in range(1):
        ijkl_iter=list(product(range(3), repeat=12))
        for ijkl in ijkl_iter:
            i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,i3=ijkl
            if (i1+i2+i3)<=2 and (j1+j2+j3)<=2 and (k1+k2+k3)<=2 and (l1+l2+l3)<=2:
                counter+=1
                V=TRANSFER1("X",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
                if V!=0: #V is a key to a missing integral
                    W=try_to_compute(V)
                    if W==1:
                        V=TRANSFER1("X",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)   
                V=TRANSFER1("Y",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
                if V!=0:
                    W=try_to_compute(V)
                    if W==1:
                        V=TRANSFER1("Y",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3) 
                V=TRANSFER1("Z",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
                if V!=0:
                    W=try_to_compute(V)
                    if W==1:
                        V=TRANSFER1("Z",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3) 
        sizeI=len(requested_keys.keys())
        for keyr in integrals_all.keys():
            getV=requested_keys.get(keyr)
            if getV!=None:
                requested_keys.pop(keyr)
        sizeE=len(requested_keys.keys())
        #print(sizeI,sizeE)
        
    for N in range(1):
        ijkl_iter=list(product(range(3), repeat=12))
        for ijkl in ijkl_iter:
            i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,i3=ijkl
            if (i1+i2+i3)<=2 and (j1+j2+j3)<=2 and (k1+k2+k3)<=2 and (l1+l2+l3)<=2:
                V=TRANSFER2("X",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
                if V!=0:
                    W=try_to_compute(V)
                    if W==1:
                        V=TRANSFER2("X",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)   
                V=TRANSFER2("Y",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
                if V!=0:
                    W=try_to_compute(V)
                    if W==1:
                        V=TRANSFER2("Y",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3) 
                V=TRANSFER2("Z",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
                if V!=0:
                    W=try_to_compute(V)
                    if W==1:
                        V=TRANSFER2("Z",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3) 
        sizeI=len(requested_keys.keys())
        for keyr in integrals_all.keys():
            getV=requested_keys.get(keyr)
            if getV!=None:
                requested_keys.pop(keyr)
        sizeE=len(requested_keys.keys())
        print(sizeI,sizeE)


# In[ ]:


counter


# In[ ]:


# We have "transferred" the two-electron integrals: Are them complete? Are tere no missing one?
#
# If something is missing,
# we might retrieve it using the symmetric property of the integrals:
# [AB|CD]=[BA|CD]=[AB|DC]=[BA|CD] and so and on...
#





def interchanged(kr):
    Z=[]
    N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=kr
    R=[[i1,i2,i3],[j1,j2,j3],[k1,k2,k3],[l1,l2,l3]]
    RS=[N]+R[0]+R[1]+R[2]+R[3]
    Z.append(RS)
    RS=[N]+R[1]+R[0]+R[2]+R[3]
    Z.append(RS)
    RS=[N]+R[0]+R[1]+R[3]+R[2]
    Z.append(RS)
    RS=[N]+R[1]+R[0]+R[3]+R[2]
    Z.append(RS)
    R=[[k1,k2,k3],[l1,l2,l3],[i1,i2,i3],[j1,j2,j3]]
    RS=[N]+R[0]+R[1]+R[2]+R[3]
    Z.append(RS)
    RS=[N]+R[1]+R[0]+R[2]+R[3]
    Z.append(RS)
    RS=[N]+R[0]+R[1]+R[3]+R[2]
    Z.append(RS)
    RS=[N]+R[1]+R[0]+R[3]+R[2]
    Z.append(RS)
    ZZ=[tuple(i) for i in Z]
    return ZZ    

#for ijkl in requested_keys.keys():
#    print(ijkl)
#    for N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3 in interchanged(ijkl):
#        getW=INTEGRALALL.get((N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3))
#        if getW!=None:
#            print(ijkl,"INDEX INTERCHANGED, AND FOUND" )
#            break
    
itr_ijkl=list(product(range(2), repeat=12))
index=1
INTERCHANGED=0
ptlost=[]
formulas=[]
for ijkl in itr_ijkl:
    pt=tuple([0]+list(ijkl))
    N,ii1,ii2,ii3,ij1,ij2,ij3,ik1,ik2,ik3,il1,il2,il3=pt

    if (ii1+ii2+ii3)==1 and (ij1+ij2+ij3)==1 and (ik1+ik2+ik3)==1 and (il1+il2+il3)==1:
        #print(pt)
        #print(N,ii1,ii2,ii3,ij1,ij2,ij3,ik1,ik2,ik3,il1,il2,il3)
        #print("Checks ", pt[0],pt[1:],pt,type(pt))
        F=0
        #print(type((N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)))
        getV=integrals_all.get((N,ii1,ii2,ii3,ij1,ij2,ij3,ik1,ik2,ik3,il1,il2,il3))
        #print((N,ii1,ii2,ii3,ij1,ij2,ij3,ik1,ik2,ik3,il1,il2,il3),getV)
        print("\n\n\n\n")
        if getV!=None:
            F=1
            print(index,pt,"FOUND")
            print(getV)
            formulas.append([pt,str(getV)])
            index+=1
        for ptn in interchanged(pt):
            N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=ptn
            #print(ptn,type(ptn),(N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)==ptn)
            getW=integrals_all.get(ptn)
            #if getW!=None and getV!=None:
                #print(pt,"INDEX INTERCHANGED, AND FOUND" ,ptn, "interchange:",not pt==ptn)
                #print(getW)
                #if pt!=ptn:
                #    INTERCHANGED+=1
                    #ptlost.append(pt)
        #        F=1
        #        index+=1
        #        break
        if F==0:
            print("\n\n",pt,"Missing!\n\n")
            ptlost.append(pt)
            
#
#  IF THERE IS NO MESSAGE <MISSING!> AND THE <INTERCHANGED BELOW> IS ZERO, WE HAVE COMPLETED THE LIST OF INTEGRALS.
#
print(INTERCHANGED)            
print(ptlost)              


# In[ ]:


import pickle
 
with open('2eri.pickle', mode='wb') as f:
    pickle.dump(formulas, f)        
print(integrals_all.get((0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
print(integrals_all.get((0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)))


# In[ ]:


for tn in ["TN0", "TN1","TN2","TN3","TN4","TN5","TN6"]:
    FOUND=False
    for f in formulas:
        if f[1].find(tn)>=0:
            FOUND=True
            break
    print(tn,FOUND)
    


# In[ ]:


sympy.sympify(str(integrals_all.get((0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))))


# In[ ]:


formulas


# In[ ]:


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
#b=a
p=a+b

X_P=(a*XA+b*XB)/p
Y_P=(a*YA+b*YB)/p
Z_P=(a*ZA+b*ZB)/p
#d=c
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


# In[ ]:


for ky in integrals_all.keys():
    if (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) ==ky:
        print(ky)
        
integrals_all.get((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1))


# In[ ]:


itr_ijkl=list(product(range(2), repeat=12))
for ijkl in itr_ijkl:
    pt=tuple([0]+list(ijkl))
    N,ii1,ii2,ii3,ij1,ij2,ij3,ik1,ik2,ik3,il1,il2,il3=pt
    if (ii1+ii2+ii3)==1 and (ij1+ij2+ij3)==1 and (ik1+ik2+ik3)==1 and (il1+il2+il3)==1:
        print(pt,integrals_all.get((N,ii1,ii2,ii3,ij1,ij2,ij3,ik1,ik2,ik3,il1,il2,il3)))


# In[ ]:


def try_to_compute2(kr):
    print(">>>TRY TO COMPUTE AT",kr)
    N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=kr
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
    if integrals_all.get(kr)!=None:
        print(">>>>SUCCESS!")
        return 1
    else:
        print(">>>>FAIL!")
        return 0
for x in ptlost:
    print(x)
    try_to_compute2(x)
    
ptlost=[]
itr_ijkl=list(product(range(2), repeat=12))
for ijkl in itr_ijkl:
    pt=tuple([0]+list(ijkl))
    N,ii1,ii2,ii3,ij1,ij2,ij3,ik1,ik2,ik3,il1,il2,il3=pt
    if (ii1+ii2+ii3)==1 and (ij1+ij2+ij3)==1 and (ik1+ik2+ik3)==1 and (il1+il2+il3)==1:
        getV=integrals_all.get((N,ii1,ii2,ii3,ij1,ij2,ij3,ik1,ik2,ik3,il1,il2,il3))
        if getV==None:
            ptlost.append(pt)
        


# In[ ]:


for x in ptlost:
    print(x)
    N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=pt
    pt1=N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3
    pt2=N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3


# In[ ]:


REQUIRED=[]
for y in ptlost:
    for x in IG.out_edges(y):
        getV=integrals_all.get(x[1])
        print(x[1],getV)
        if getV==None:
            REQUIRED.append(x[1])
print(REQUIRED)
    


# In[ ]:


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


# In[ ]:


AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
z1,z2,z3,z4=symbols("z1 z2 z3 z4")
f=T0000(1,1,1,1,[AX,0,0],[0,0,0],[CX,0,0],[1,1,1])
print(f)


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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

# In[ ]:


#  A TEST OF THE PROGRAM
# 
#  Compute various integrals [[AX,0,0],[0,0,0] | [CX,0,0],[1,0,0] ]
#
#
#
# 
a,b,c,d, p, q=symbols("a b c d p q")
XA,YA,ZA=symbols("XA YA ZA")
XB,YB,ZB=symbols("XB YB ZB")
XC,YC,ZC=symbols("XC YC ZC")
XD,YD,ZD=symbols("XD YD ZD")
XP,YP,ZP=symbols("XP YP ZP")
XQ,YQ,ZQ=symbols("XQ YQ ZQ")

X_AB=AX
X_CD=CX-1

aa=1.45
bb=1.54
cc=1.3
dd=1.2
X_P=aa*AX/(aa+bb)
pp=aa+bb
qq=cc+dd
X_Q=(cc*CX+dd*1)/(cc+dd)
X_PQ=X_P-X_Q
X_PA=X_P-AX
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


# In[ ]:


v0000


# In[ ]:


v1000


# In[ ]:


v2000


# #
# # TEST OF COMPUTATION
# #

# In[ ]:


#
# EVALUATED THETA(N;000 000 000 000) at ([AX,0,0],[0,0,0],[CX,0,0],[DX,0,0])
#
V_T_0_0000=2*sympy.pi**2.5/pp/qq/sympy.sqrt(pp+qq)*KXYZAB(aa,bb,[AX,0,0],[0,0,0])*KXYZAB(cc,dd,[CX,0,0],[1,0,0])*BOYS(0,alpha*X_PQ*X_PQ)
V_T_1_0000=2*sympy.pi**2.5/pp/qq/sympy.sqrt(pp+qq)*KXYZAB(aa,bb,[AX,0,0],[0,0,0])*KXYZAB(cc,dd,[CX,0,0],[1,0,0])*BOYS(1,alpha*X_PQ*X_PQ)
V_T_2_0000=2*sympy.pi**2.5/pp/qq/sympy.sqrt(pp+qq)*KXYZAB(aa,bb,[AX,0,0],[0,0,0])*KXYZAB(cc,dd,[CX,0,0],[1,0,0])*BOYS(2,alpha*X_PQ*X_PQ)
V_T_3_0000=2*sympy.pi**2.5/pp/qq/sympy.sqrt(pp+qq)*KXYZAB(aa,bb,[AX,0,0],[0,0,0])*KXYZAB(cc,dd,[CX,0,0],[1,0,0])*BOYS(3,alpha*X_PQ*X_PQ)
V_T_4_0000=2*sympy.pi**2.5/pp/qq/sympy.sqrt(pp+qq)*KXYZAB(aa,bb,[AX,0,0],[0,0,0])*KXYZAB(cc,dd,[CX,0,0],[1,0,0])*BOYS(4,alpha*X_PQ*X_PQ)


# In[ ]:


sympy.N(v0000.subs([(q,qq),(p,pp),(b,bb),(d,dd),(XPQ,X_PQ),(XAB,AX),(XCD,CX-1),(XPA,X_P-AX),(TNS[0],V_T_0_0000),(TNS[1],V_T_1_0000),(AX,1),(CX,2)])).expand()


# In[ ]:


sympy.N(th0000.subs([(q,qq),(p,pp),(b,bb),(d,dd),(XPQ,X_PQ),(XAB,AX),(XCD,CX-1),(XPA,X_P-AX),(AX,1),(CX,2)]))


# In[ ]:


V_T_0_0000


# In[ ]:


th0000


# In[ ]:


(v1000.subs([(q,qq),(p,pp),(b,bb),(d,dd),(XPQ,X_PQ),(XAB,AX),(XCD,CX-1),(XPA,X_P-AX),(TNS[0],V_T_0_0000),(TNS[1],V_T_1_0000),(AX,1),(CX,2)])).expand()


# In[ ]:


ww=v1000.subs([(q,qq),(p,pp),(b,bb),(d,dd),(XPQ,X_PQ),(XAB,AX),(XCD,CX-1),(XPA,XP-AX),(TNS[0],V_T_0_0000),(TNS[1],V_T_1_0000)])
#ww


# In[ ]:


sympy.N(v1000.subs([(q,qq),(p,pp),(b,bb),(d,dd),(XPQ,X_PQ),(XAB,AX),(XCD,(CX-1)),(XPA,(X_P-AX)),(TNS[0],V_T_0_0000),(TNS[1],V_T_1_0000),(AX,1),(CX,2)])).expand()


# In[ ]:


sympy.N(th1000.subs([(q,qq),(p,pp),(b,bb),(d,dd),(XPQ,X_PQ),(XAB,AX),(XCD,CX-1),(XPA,X_P-AX),(AX,1),(CX,2)]))


# In[ ]:


sympy.N(v2000.subs([(q,qq),(p,pp),(b,bb),(d,dd),(XPQ,X_PQ),(XAB,AX),(XCD,(CX-1)),(XPA,(X_P-AX)),(TNS[0],V_T_0_0000),(TNS[1],V_T_1_0000),(TNS[2],V_T_2_0000)]).expand())


# In[ ]:


sympy.N(th2000.expand())


# In[ ]:


sympy.N(v2000.subs([(q,qq),(p,pp),(b,bb),(d,dd),(XPQ,X_PQ),(XAB,AX),(XCD,(CX-1)),(XPA,(X_P-AX)),(TNS[0],V_T_0_0000),(TNS[1],V_T_1_0000),(TNS[2],V_T_2_0000),(AX,1),(CX,2)])).expand()


# In[ ]:


sympy.N(th2000.subs([(q,qq),(p,pp),(b,bb),(d,dd),(XPQ,X_PQ),(XAB,AX),(XCD,(CX-1)),(XPA,(X_P-AX)),(AX,1),(CX,2)])).expand()


# In[ ]:


sympy.N(v1111.subs([(q,qq),(p,pp),(b,bb),(d,dd),(XPQ,X_PQ),(XAB,AX),(XCD,(CX-1)),(XPA,(X_P-AX)),             (TNS[0],V_T_0_0000),(TNS[1],V_T_1_0000),(TNS[2],V_T_2_0000),             (TNS[3],V_T_3_0000),(TNS[4],V_T_4_0000),             (AX,1),(CX,2)])).expand()


# In[ ]:


sympy.N(th1111.subs([(q,qq),(p,pp),(b,bb),(d,dd),(XPQ,X_PQ),(XAB,AX),(XCD,(CX-1)),(XPA,(X_P-AX)),(BX,0),(DX,1),(TNS[0],V_T_0_0000),(TNS[1],V_T_1_0000),(TNS[2],V_T_2_0000),(AX,1),(CX,2)])).expand()


# In[ ]:


pp=aa+bb
qq=cc+dd
V1010=-(bb*X_AB+dd*X_CD)/qq*th1000+1/2/qq*th0000-pp/qq*th2000
V0010=-(bb*X_AB+dd*X_CD)/qq*th0000-pp/qq*th1000
V0020=-(bb*X_AB+dd*X_CD)/qq*th0010+1/2/qq*th0000-pp/qq*th1010


# In[ ]:


[aa,bb,cc,dd,pp,qq]


# In[ ]:


v1010


# In[ ]:


[a,b,c,d]


# In[ ]:


[(V0010.subs([(AX,1),(CX,2)])),(th0010.subs([(AX,1),(CX,2)]))]


# In[ ]:


sympy.N(v1010.subs([(q,qq),(p,pp),(b,bb),(a,aa),(c,cc),(d,dd),(XPQ,X_PQ),(XAB,AX),(XCD,(CX-1)),(XPA,(X_P-AX)),(BX,0),(DX,1),(TNS[0],V_T_0_0000),(TNS[1],V_T_1_0000),(TNS[2],V_T_2_0000),(AX,1),(CX,2)])).expand()


# In[ ]:


[sympy.N(V1010.subs([(AX,1),(CX,2)])),sympy.N(th1010.subs([(AX,1),(CX,2)]))]


# In[ ]:


[(V0020.subs([(AX,1),(CX,2)])),(th0020.subs([(AX,1),(CX,2)]))]


# In[ ]:


[sympy.N(V0020.subs([(AX,1),(CX,2)])),sympy.N(th0020.subs([(AX,1),(CX,2)]))]


# In[ ]:


sympy.N(v0020.subs([(q,qq),(p,pp),(a,aa),(c,cc),(b,bb),(d,dd),(XPQ,X_PQ),(XAB,AX),(XCD,(CX-1)),(XPA,(X_P-AX)),(TNS[0],V_T_0_0000),(TNS[1],V_T_1_0000),(TNS[2],V_T_2_0000),(AX,1),(CX,2)])).expand()

