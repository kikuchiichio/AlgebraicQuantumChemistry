#!/usr/bin/env python
# coding: utf-8

# In[133]:


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
    PI = np.pi
    if  type(ARG)==float and ARG < 1.0e-6:
        return 1 -ARG/3.
    if  type(ARG)==sympy.core.numbers.Zero and ARG < 1.0e-6:
        return 1 -ARG/3.
    else:
        #print("F0:ARG",ARG)
        return sympy.sqrt(PI/ARG)*sympy.erf(sqrt(ARG))/2
    
def S000000(A,B,RA,RB):
#
#   OVERLAP INTEGRAL between exp(-A*|r-RA|^2) and exp(-B*|r-RB|^2)
#
    RAB2=sum([(c1-c2)**2 for c1,c2 in zip(RA,RB)])
    PI=np.pi
    return (PI/(A+B))**1.5*sympy.exp(-A*B*RAB2/(A+B))

def getS(A,B,RAB2):
#
# Calculates the overlap matrix between exp(-A*|r-RA|^2) and exp(-A*|B-RB|^2)
# WHEN RAB2=|RA-RB|^2
#
    PI=np.pi
    #print(A,B,RAB2)
    return (PI/(A+B))**1.5*sympy.exp(-A*B*RAB2/(A+B))

        
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
            f=f.diff(Z)
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
            return sympy.sqrt(PI/ARG)*sympy.erf(sqrt(ARG))/2
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
    nx.add_star(IG, [keyn,key0, key1, key2, key3])
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
        integrals_all[keyn]= -(b*DAB+d*DCD)/q*data0 -i_d/2/q*data1 +k_d/2/q*data2 -p/q*data3
    #else:
        #print(keyn,"NOT WRITTEN")
        #print(key0,key1,key2,"key3=",key3)
        #print(data0,data1,data2,data3)
        
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
    for N in range(1):
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




#
#  WE TRANSFER THE INTEGRALS TO COVER THE BLANK DATA.
#
# TRANSFER 1 and 2: THESE FUNCTION COMPUTES
#  (N; i j+1 k l ) <= (N;i+1 j k l), (N; i j k l)
#  (N; i j k l+1 ) <= (N;j k+1 l), (N; i j k l)
#
#  MAYBE SOMETHING NECESSARY ARE STILL MISSING; 
#  IF WE DETECT THEM, WE TRY TO COMPUTE THEM BY "HORIZONTAL" ON THE FLY!
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



# In[134]:


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
for ijkl in itr_ijkl:
    pt=tuple([0]+list(ijkl))
    N,ii1,ii2,ii3,ij1,ij2,ij3,ik1,ik2,ik3,il1,il2,il3=pt

    if (ii1+ii2+ii3)<=1 and (ij1+ij2+ij3)<=1 and (ik1+ik2+ik3)<=1 and (il1+il2+il3)<=1:
        #print(pt)
        #print(N,ii1,ii2,ii3,ij1,ij2,ij3,ik1,ik2,ik3,il1,il2,il3)
        #print("Checks ", pt[0],pt[1:],pt,type(pt))
        F=0
        #print(type((N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)))
        getV=integrals_all.get((N,ii1,ii2,ii3,ij1,ij2,ij3,ik1,ik2,ik3,il1,il2,il3))
        #print((N,ii1,ii2,ii3,ij1,ij2,ij3,ik1,ik2,ik3,il1,il2,il3),getV)
        if getV!=None:
            F=1
            print(index,pt,"FOUND",getV)
            index+=1
        #for ptn in interchanged(pt):
        #    N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=ptn
        #    #print(ptn,type(ptn),(N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)==ptn)
        #    getW=integrals_all.get((N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3))
        #    if getW!=None and getV==None:
        #        print(index,pt,"INDEX INTERCHANGED, AND FOUND" ,ptn, "interchange:",not pt==ptn)
        #        print(getW)
        #        if pt!=ptn:
        #            INTERCHANGED+=1
        #            #ptlost.append(pt)
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


# In[136]:


for ky in integrals_all.keys():
    if (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) ==ky:
        print(ky)
        
integrals_all.get((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1))


# In[141]:


itr_ijkl=list(product(range(2), repeat=12))
for ijkl in itr_ijkl:
    pt=tuple([0]+list(ijkl))
    N,ii1,ii2,ii3,ij1,ij2,ij3,ik1,ik2,ik3,il1,il2,il3=pt
    if (ii1+ii2+ii3)==1 and (ij1+ij2+ij3)==1 and (ik1+ik2+ik3)==1 and (il1+il2+il3)==1:
        print(pt,integrals_all.get((N,ii1,ii2,ii3,ij1,ij2,ij3,ik1,ik2,ik3,il1,il2,il3)))


# In[142]:


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
        


# In[143]:


for x in ptlost:
    print(x)
    N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=pt
    pt1=N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3
    pt2=N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3


# In[144]:


REQUIRED=[]
for y in ptlost:
    for x in IG.out_edges(y):
        getV=integrals_all.get(x[1])
        print(x[1],getV)
        if getV==None:
            REQUIRED.append(x[1])
print(REQUIRED)
    

