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
# $[A,B|A+T,B+T]$  or $[A,B+(T,0,0)|A+(0,T,0),B+(0,0,T)]$  
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
# 
# ## A design to compute the electron integrals with all possible type of shells.
# 
# Each shell is indicated by a key (i,j,k), namely, (0,0,0),(1,0,0),(0,1,0) and so on. A key shows the exponents of Cartesian Gaussians
# $
# (a,A)=(x-Ax)^i(y-Ay)^j(z-Az)\exp(-a|r-A|^2)
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
# ### In the following we compute [(z1,A)(z2,A)|(z4,A)(z4,A)] and [(z1,A)(z2,B)|(z4,A)(z4,B)]  by symbolic differentiation.  
# ### We compute [A,B+(T,0,0)|A+(0,T,0),B+(0,0,T)] and its series expansion with respect to T
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


# In[2]:


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



AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
z1,z2,z3,z4=symbols("z1 z2 z3 z4")
f=T0000(z1,z2,z3,z4,[AX,AY,AZ],[BX,BY,BZ],[CX,CY,CZ],[DX,DY,DZ])
print(f)


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




# In[3]:


import numpy
from sympy.abc import x
from sympy.utilities.lambdify import implemented_function
from sympy import lambdify


# In[4]:


TTlist=dict()
Integralslist=dict()
IntegralslistABAB=dict()
IntegralslistAAAA=dict()
AOIntegralslist=dict()


def indexijkl(i,j,k,l):
    A=[(i,j,k,l),(j,i,k,l),(i,j,l,k),(j,i,l,k)]
    A+=[(k,l,i,j),(l,k,i,j),(k,l,j,i),(l,k,j,i)]
    A.sort()
    return A[0]


ORBITKEYS=[(0,0,0),(1,0,0),(0,1,0),(0,0,1)]
NORBITALS=len(ORBITKEYS)


from sympy import symbols  



print('\
\n#  We compute two-center integrals \
\n#  \
\n#  The integral [AB|AB] becomes falsely simgular.\
\n#  So we compute [AB|A+T, B+T] (with the shift of the coordinates of the last two centers) to get it as the polynomial of T.\
\n#  After this loop, we elicit the correct value of that type of the integrals.\
\n#  \
')
TTlist=dict()
TTlistAAAA=dict()
TTlistABAB=dict()
Flist=dict()
ComputedKeys=dict()
def ComputeAllTwoERIAAAA():
    AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
    z1,z2,z3,z4=symbols("z1 z2 z3 z4")
    T=symbols("T", positive=True)


    for ID in range(NORBITALS):
        for JD in range(NORBITALS):
            for KD in range(NORBITALS):
                for LD in range(NORBITALS):

                    key1=ORBITKEYS[ID]
                    key2=ORBITKEYS[JD]
                    key3=ORBITKEYS[KD]
                    key4=ORBITKEYS[LD]
                    keys1234=key1+key2+key3+key4
                    
                    #print(key1,key2,key3,key4)
                    #if ID!=KD or JD!=LD:
                    #formula=GetTwoEIntegral(key1,key2,key3,key4)
                    RRA=[AX,AY,AZ]
                    RRB=[BX+T,BY,BZ]
                    RRC=[CX,CY+T,CZ]
                    RRD=[DX,DY,DZ+T]

                    if TTlist.get(indexijkl(key1,key2,key3,key4))==None:
                        print("\n",key1,key2,key3,key4)
                        fml=T0000(z1,z2,z3,z4,RRA,RRB,RRC,RRD)
                        listkeys1234=list(keys1234)
                        for i in range(12):
                            currentkeys=[x for x in listkeys1234]
                            currentkeys[i] = currentkeys[i]-1
                            nkey1=tuple(currentkeys[0:3])
                            nkey2=tuple(currentkeys[3:6])
                            nkey3=tuple(currentkeys[6:9])
                            nkey4=tuple(currentkeys[9:12])
                            fmt=TTlist.get(indexijkl(nkey1,nkey2,nkey3,nkey4))
                                
                            dkey=None
                            CRD=None
 
                            if i<3:
                                dkey=key1
                                CRD=[AX,AY,AZ]
                                denom=2*z1
                            if 3<=i<6:
                                dkey=key2
                                CRD=[BX,BY,BZ]
                                denom=2*z2
                            if 6<=i<9:
                                dkey=key3
                                CRD=[CX,CY,CZ]
                                denom=2*z3
                            if 9<=i<12:
                                dkey=key4
                                CRD=[DX,DY,DZ]
                                denom=2*z4
                            if fmt!=None:
                                fml=fmt
                                break

                        if fmt==None:
                            print("Not Computable from below",key1,key2,key3,key4)
                            
                        if fmt !=None:
                            print("Computable from below",nkey1,nkey2,nkey3,nkey4)
                            for ifl,var in zip(list(dkey),CRD):
                                print(ifl,var)
                                if ifl==1:
                                    fml=fml.diff(var)
                            fml=fml/denom
                        fmlAAAA=fml.subs([(BX,AX),(BY,AY),(BZ,AZ),(CX,AX),(CY,AY),(CZ,AZ),(DX,AX),(DY,AY),(DZ,AZ)])

                        fmlABAB=fml.subs([(CX,AX),(CY,AY),(CZ,AZ),(DX,BX),(DY,BY),(DZ,BZ)])

                        TTlist[indexijkl(key1,key2,key3,key4)]=fml
                        TTlistAAAA[indexijkl(key1,key2,key3,key4)]=fmlAAAA

ComputeAllTwoERIAAAA()


# In[5]:


T=symbols("T", positive=True)
TTlistAAAA


# In[6]:


expr=TTlistAAAA.get(((1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0)))
print(expr)


# In[7]:


#
# [A,A+(T,0,0)|A+(0,T.0),A+(0,0,T)] and series expansion with respect to T
#
for k in TTlistAAAA.keys():
    expr=TTlistAAAA.get(k)
    #print(expr)
    print(sympy.series(expr.subs([(AX,1),(AY,0),(AZ,0),(z1,1),(z2,1),(z3,1),(z4,1)]),T,0))

