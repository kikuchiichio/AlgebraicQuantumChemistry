#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


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
    PI=np.p
    RPQ=[(c1-c2)**2 for c1,c2 in zip(RP,RQ)]
    RPQ2=sum(RPQ)
    return 2*PI**(2.5)/p/q/np.sqrt(p+q)*KXYZAB(a,b,RA,RB)*KXYZAB(c,d,RC,RD)*BOYS(N,alpha*RPQ2)


# In[ ]:


AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
z1,z2,z3,z4=symbols("z1 z2 z3 z4")
#z1,z2,z3,z4=[1,2,3,4]
f=T0000(z1,z2,z3,z4,[AX,AY,AZ],[BX,BY,BZ],[CX,CY,CZ],[DX,DY,DZ])
print(f)


# In[ ]:


((S000000(z1,z2,[AX,AY,AZ],[BX,BY,BZ]).diff(AX)).diff(AX)).subs([(BX,AX),(BY,AY),(BZ,AZ)])


# In[ ]:


def CGAUSS(x,y,z,i,j,k,A,RA):
    RAX,RAY,RAZ=RA
    R2=(x-RAX)**2+(y-RAY)**2+(z-RAZ)**2
    return (x-RAX)**i*(y-RAY)**j*(z-RAZ)**k*sympy.exp(-A*R2) 


# #
# # PERFORMANCE TEST CONCERNING DIFFERENTIATION (TO GENERATE TWO-ELECTRON INTERGRALS OF P-ORBITALS)
# #
# import time
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

# In[ ]:


x,y,z=symbols("x y z")
RAX,RAY,RAX=symbols("RAX RAY RAZ")
alpha=symbols("alpha")
f=2+CGAUSS(x,y,z,1,1,1,alpha,[RAX,RAY,RAX])


# In[ ]:


expand(f).as_ordered_terms()


# In[ ]:


def NLIST(N):
    if N==0:
        return ["UNDEF" for _ in range(4)]
    else:
        return [NLIST(N-1) for _ in range(4)]

A=NLIST(4)
from sympy import Function, Symbol
TN=Function("TN0000")
UNDEF=Function("UNDEF")


# In[ ]:


def TREEINDEX(i1,i2,i3):
    N=3
    return i1+N*i2+N*N*i3

def TWELVEINDEX(i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3):
    M=3*3*3
    ti=TREEINDEX(i1,i2,i3)
    tj=TREEINDEX(j1,j2,j3)
    tk=TREEINDEX(k1,k2,k3)
    tl=TREEINDEX(l1,l2,l3)
    return ti+M*tj+M*M*tk+M*M*M*tl


# In[ ]:


INTEGRALALL=dict()
for i in range(11):
    INTEGRALALL[(i,0,0,0,0,0,0,0,0,0,0,0,0)]=TN(i)


# In[ ]:


i=0
while(i<5):
    print(i)
    i+=1


# # Obara-Saika Scheme for Two-electron integrals

# In[ ]:


XPA, XPQ, XCD, XAB=symbols("XPA XPQ XCD XAB")
YPA, YPQ,YCD, YAB=symbols("YPA YPQ YCD YAB")
ZPA, ZPQ, ZCD, ZAB=symbols("ZPA ZPQ ZCD ZAB")

alpha=symbols("alpha")
p,q=symbols("p q")
b,d=symbols("b d")

#
# VERTICAL
#  (N: i+1 0 0 0 ) <- (N: i 0 0 0) , (N+1 i 0 0 0) , (N i-1 0 0 0) (N+1 i 0 0 0)
#

def VERTICAL(D,N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3,INFO=0):
    
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
    if None!=INTEGRALALL.get(keyn):
        return
    
    data0=INTEGRALALL.get(key0)
    data1=INTEGRALALL.get(key1)
    data2=INTEGRALALL.get(key2)
    data3=INTEGRALALL.get(key3)
    if min(list(key2))<0:
        data2=0
        INTEGRALALL[key2]=0
    if min(list(key3))<0:
        data3=0
        INTEGRALALL[key3]=0
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
        INTEGRALALL[keyn]= DPA*data0-alpha/p*DPQ*data1+i_d/2/p*(data2-alpha/p*data3)
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
INTEGRALALL=dict()
for i in range(11):
    INTEGRALALL[(i,0,0,0,0,0,0,0,0,0,0,0,0)]=TN(i)        

    
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
    for keyr in INTEGRALALL.keys():
        getV=requested_keys.get(keyr)
        if getV!=None:
            requested_keys.pop(keyr)
    sizeE=len(requested_keys.keys())
    print("LOOP:",LOOP)
    LOOP+=1


# In[ ]:


print(len(requested_keys.keys()))
for keyr in INTEGRALALL.keys():
    getV=requested_keys.get(keyr)
    if getV!=None:
        requested_keys.pop(keyr)
print(len(requested_keys.keys()))


# In[ ]:


for kr in requested_keys.keys():
    N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=list(kr)
    if i1<=2 and i2<=2 and i3<=2:
        print(kr)


# In[ ]:


klist=[]
for keys,data in zip(INTEGRALALL.keys(),INTEGRALALL.values()):
    N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=list(keys)
    if N==3 and min(list(keys))>=0 and sum(list(keys))-N==2:
        klist.append(keys)
        print(keys)
            
for keys,data in zip(INTEGRALALL.keys(),INTEGRALALL.values()):
    N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=list(keys)            
    if N==1 and min(list(keys))>=0:
        klist.append(keys)
        #print(keys)


# In[ ]:


#
# HORIZONRAL
# (N; i 0 k+1 0) <= (N; i 0 k 0), (N; i-1 0 k 0), (N; i, 0, k-1, 0), (N; i+1 0 k 0)
#

def HORIZONTAL(D,N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3):
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
    data0=INTEGRALALL.get(key0)
    data1=INTEGRALALL.get(key1)
    data2=INTEGRALALL.get(key2)
    data3=INTEGRALALL.get(key3)
    if min(list(key0))<0:
        data0=0
        INTEGRALALL[key0]=0
    if min(list(key1))<0:
        data1=0
        INTEGRALALL[key1]=0
    if min(list(key2))<0:
        data2=0
        INTEGRALALL[key2]=0
    if min(list(key3))<0:
        data3=0
        INTEGRALALL[key3]=0
 
    if data0==None:
        requested_keys[key0]=key0
    if data1==None:
        requested_keys[key1]=key1
    if data2==None:
        requested_keys[key2]=key2
    if data3==None:
        requested_keys[key3]=key3
    #if data0==None:
    #    INTEGRALALL[key0]="REQUIRED"
    #if data1==None:
    #    INTEGRALALL[key1]="REQUIRED"
    #if data2==None:
    #    INTEGRALALL[key2]="REQUIRED"
    #if data3==None:
    #    INTEGRALALL[key3]="REQUIRED"
    if data0!=None and data1!=None and data2!=None and data3!=None:
        #print(data0,data1,data2,data3)
        #print(b,d,DAB,DCD,i1,k1,p,q)
        INTEGRALALL[keyn]= -(b*DAB+d*DCD)/q*data0 -i_d/2/q*data1 +k_d/2/q*data2 -p/q*data3
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
        sizeI=len(requested_keys.keys())
        for keyr in INTEGRALALL.keys():
            getV=requested_keys.get(keyr)
            if getV!=None:
                requested_keys.pop(keyr)
        sizeE=len(requested_keys.keys())
        print(LOOP,sizeI,sizeE)
        LOOP+=1


# In[ ]:


keyslist=[kr for kr in requested_keys.keys()]

for kr in keyslist:
    N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=list(kr) 
    if max([i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3])<=1:
        print(kr)
        N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=kr
        INTEGRALALL.get(kr)
        #HORIZONTAL("X",N,i1,i2,i3,j1,j2,j3,k1-1,k2,k3,l1,l2,l3)
        #HORIZONTAL("Y",N,i1,i2,i3,j1,j2,j3,k1,k2-1,k3,l1,l2,l3)
        #HORIZONTAL("Z",N,i1,i2,i3,j1,j2,j3,k1,k2,k3-1,l1,l2,l3)

for kr in keyslist:
    N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=list(kr) 
    if max([i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3])<=2:
        N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=kr
        INTEGRALALL.get(kr)
        HORIZONTAL("X",N,i1,i2,i3,j1,j2,j3,k1-1,k2,k3,l1,l2,l3)
        HORIZONTAL("Y",N,i1,i2,i3,j1,j2,j3,k1,k2-1,k3,l1,l2,l3)
        HORIZONTAL("Z",N,i1,i2,i3,j1,j2,j3,k1,k2,k3-1,l1,l2,l3)


# In[ ]:


print(len(requested_keys.keys()))
for keyr in INTEGRALALL.keys():
    getV=requested_keys.get(keyr)
    if getV!=None:
        requested_keys.pop(keyr)
print(len(requested_keys.keys()))
keyslist=[kr for kr in requested_keys.keys()]
for kr in keyslist:
    N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=list(kr) 
    if max([i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3])<=2 and min([i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3])>=0:
        print(kr)
        N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=kr


# In[ ]:


print(len(requested_keys.keys()))
for keyr in INTEGRALALL.keys():
    getV=requested_keys.get(keyr)
    if getV!=None:
        requested_keys.pop(keyr)
print(len(requested_keys.keys()))
keyslist=[kr for kr in requested_keys.keys()]
for kr in keyslist:
    N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=list(kr) 
    if max([i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3])<=1 and min([i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3])>=0:
        print(kr)
        N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=kr


# In[ ]:


N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=(0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0)
HORIZONTAL("X",N,i1,i2,i3,j1,j2,j3,k1-1,k2,k3,l1,l2,l3)
HORIZONTAL("Y",N,i1,i2,i3,j1,j2,j3,k1,k2-1,k3,l1,l2,l3)
HORIZONTAL("Z",N,i1,i2,i3,j1,j2,j3,k1,k2,k3-1,l1,l2,l3)


# In[ ]:


N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=(0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0)
HORIZONTAL("X",N,i1,i2,i3,j1,j2,j3,k1-1,k2,k3,l1,l2,l3)
HORIZONTAL("Y",N,i1,i2,i3,j1,j2,j3,k1,k2-1,k3,l1,l2,l3)
HORIZONTAL("Z",N,i1,i2,i3,j1,j2,j3,k1,k2,k3-1,l1,l2,l3)


# In[ ]:


N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=(0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0)
HORIZONTAL("X",N,i1,i2,i3,j1,j2,j3,k1-1,k2,k3,l1,l2,l3)
HORIZONTAL("Y",N,i1,i2,i3,j1,j2,j3,k1,k2-1,k3,l1,l2,l3)
HORIZONTAL("Z",N,i1,i2,i3,j1,j2,j3,k1,k2,k3-1,l1,l2,l3)


# In[ ]:


N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=(0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0)
HORIZONTAL("X",N,i1,i2,i3,j1,j2,j3,k1-1,k2,k3,l1,l2,l3)
HORIZONTAL("Y",N,i1,i2,i3,j1,j2,j3,k1,k2-1,k3,l1,l2,l3)
HORIZONTAL("Z",N,i1,i2,i3,j1,j2,j3,k1,k2,k3-1,l1,l2,l3)


# In[ ]:


N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=(0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0)
HORIZONTAL("X",N,i1,i2,i3,j1,j2,j3,k1-1,k2,k3,l1,l2,l3)
HORIZONTAL("Y",N,i1,i2,i3,j1,j2,j3,k1,k2-1,k3,l1,l2,l3)
HORIZONTAL("Z",N,i1,i2,i3,j1,j2,j3,k1,k2,k3-1,l1,l2,l3)


# In[ ]:


N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=(0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0)
HORIZONTAL("X",N,i1,i2,i3,j1,j2,j3,k1-1,k2,k3,l1,l2,l3)
HORIZONTAL("Y",N,i1,i2,i3,j1,j2,j3,k1,k2-1,k3,l1,l2,l3)
HORIZONTAL("Z",N,i1,i2,i3,j1,j2,j3,k1,k2,k3-1,l1,l2,l3)


# In[ ]:


N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=(0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0)
HORIZONTAL("X",N,i1,i2,i3,j1,j2,j3,k1-1,k2,k3,l1,l2,l3)
#HORIZONTAL("Y",N,i1,i2,i3,j1,j2,j3,k1,k2-1,k3,l1,l2,l3)
#HORIZONTAL("Z",N,i1,i2,i3,j1,j2,j3,k1,k2,k3-1,l1,l2,l3)


# In[ ]:


print(len(requested_keys.keys()))
for keyr in INTEGRALALL.keys():
    getV=requested_keys.get(keyr)
    if getV!=None:
        requested_keys.pop(keyr)
print(len(requested_keys.keys()))
keyslist=[kr for kr in requested_keys.keys()]
for kr in keyslist:
    N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=list(kr) 
    if max([i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3])<=1 and min([i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3])>=0:
        print(kr)
        N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=kr


# In[ ]:


def pt(i):
    if i==(1,0,0):
        return "x"
    if i==(0,1,0):
        return "y"
    if i==(0,0,1):
        return "z"
    if i==(0,0,0):
        return "s"
    return ""


# In[ ]:


requested_keys_save=copy.deepcopy(requested_keys)


# In[ ]:


#
# TRANSFER 
#  (N; i j+1 k l ) <= (N;i+1 j k l), (N; i j k l)
#  (N; i j k l+1 ) <= (N;j k+1 l), (N; i j k l)
#
def try_to_compute(kr):
    print("try to compute:",kr)
    N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=kr
    HORIZONTAL("X",N,i1,i2,i3,j1,j2,j3,k1-1,k2,k3,l1,l2,l3)
    HORIZONTAL("Y",N,i1,i2,i3,j1,j2,j3,k1,k2-1,k3,l1,l2,l3)
    HORIZONTAL("Z",N,i1,i2,i3,j1,j2,j3,k1,k2,k3-1,l1,l2,l3)
    if INTEGRALALL.get(kr)!=None:
        return 1
    else:
        return 0

def TRANSFER(D,N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3):
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
        
    data0=INTEGRALALL.get(key0)
    data1=INTEGRALALL.get(key1)
    if data0==None:
        requested_keys[key0]=key0
    if data1==None:
        requested_keys[key1]=key1
    if data0!=None and data1!=None:
        #print(key1,"TRANSFERED",keyn)
        #print(data0,data1,data2,data3)
        INTEGRALALL[keyn]= data0+DAB*data1
        return 0
    else:
        print(key1,"TRANSFER",keyn)
        print("FAILS ...key0=",key0)
        return key0
    if D=="X":
        key0=(N,i1,i2,i3,j1,j2,j3,k1+1,k2,k3,l1,l2,l3)
        key1=(N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
        keyn=(N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1+1,l2,l3)
        DCD=ZCD
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
    data0=INTEGRALALL.get(key0)
    data1=INTEGRALALL.get(key1)
    if data0==None:
        requested_keys[key0]=key0
    if data1==None:
        requested_keys[key1]=key1
    if data0!=None and data1!=None:
        #print(key1,"TRANSFERED",keyn)
        #print(data0,data1,data2,data3)
        INTEGRALALL[keyn]= data0+DCD*data1
        return 0
    else:
        print(key1,"TRANSFER",keyn)
        print("FAILS ...key0=",key0)
        return key0

requested_keys=dict()
sizeI=0
sizeE=1
while (sizeI!=sizeE):
    for N in range(1):
        ijkl_iter=list(product(range(2), repeat=12))
        for ijkl in ijkl_iter:
            i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,i3=ijkl
            if (i1+i2+i3)<=1 and (j1+j2+j3)<=1 and (k1+k2+k3)<=1 and (l1+l2+l3)<=1:
                V=TRANSFER("X",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
                if V!=0:
                    W=try_to_compute(V)
                    if W==1:
                        V=TRANSFER("X",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)   
                V=TRANSFER("Y",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
                if V!=0:
                    W=try_to_compute(V)
                    if W==1:
                        V=TRANSFER("Y",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3) 
                V=TRANSFER("Z",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
                if V!=0:
                    W=try_to_compute(V)
                    if W==1:
                        V=TRANSFER("Z",N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3) 
        sizeI=len(requested_keys.keys())
        for keyr in INTEGRALALL.keys():
            getV=requested_keys.get(keyr)
            if getV!=None:
                requested_keys.pop(keyr)
        sizeE=len(requested_keys.keys())
        print(sizeI,sizeE)


# # We have "transferred" the two-electron integrals, but there are still missing ones...
# ## Could we retrieve them using the symmetric property of the integrals?
# ## [AB|CD]=[BA|CD]=[AB|DC]=[BA|CD] and so and on...
# 

# In[ ]:


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
for ijkl in itr_ijkl:
    pt=tuple([0]+list(ijkl))
    N,ii1,ii2,ii3,ij1,ij2,ij3,ik1,ik2,ik3,il1,il2,il3=pt
    if (ii1+ii2+ii3)<=1 and (ij1+ij2+ij3)<=1 and (ik1+ik2+ik3)<=1 and (il1+il2+il3)<=1:
        print("Checks ", pt)
        F=0
        for ptn in interchanged(pt):
            N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=ptn
            getW=INTEGRALALL.get((N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3))
            if getW!=None:
                print(pt,"INDEX INTERCHANGED, AND FOUND" ,ptn)
                F=1
                break
        if F==0:
            print("\n\nMissing!\n\n")
            
              
              


# In[ ]:




