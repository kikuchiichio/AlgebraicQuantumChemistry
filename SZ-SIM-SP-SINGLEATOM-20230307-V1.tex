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


# In[ ]:


#
#  Custom Function to compute Boys F(N,X). It can compute the limit F(N,xX>0).
#
#  As usual, MyFN(N,X) only returns 'MyFN(N,X)'.
#  The Symbolic Diffrentiation generates Diff(F(N,X),x )-> F(N+1,X).
#  The command expr.doit() returns the proper evaluation for expr.
#
class MyFN(Function):
    """
    MyFN can be evaluated by using the doit() method.
    """
    # Define automatic evaluation on explicit numbers
    @classmethod
    def eval(cls, n, y):
        # Number is the base class of Integer, Rational, and Float
        #if all(isinstance(i, Symbol) for i in [x, y, z]):
        #   return x*y + z
        if all(isinstance(i, Number) for i in [n, x]):
            if x==0:
                return (-1)**(n)/(2*n+1)
    # Define numerical evaluation with evalf().
    def _eval_evalf(self, prec):
        return self.doit(deep=False)._eval_evalf(prec)
    # Define full evaluation to Add and Mul in doit(). This effectively
    # treats FMA(x, y, z) as just a shorthand for x*y + z that is useful
    # to have as a separate expression in some contexts and which can be
    # evaluated to its expanded form in other contexts.
    def doit(self, deep=True, **hints):
        n, x = self.args
        # Recursively call doit() on the args whenever deep=True.
        # Be sure to pass deep=True and **hints through here.
        print(n,x,type(x))
        if deep:
            x = x.doit(deep=deep, **hints)
            n = n.doit(deep=deep, **hints)
        if (type(x)==sympy.core.numbers.Zero):   
            return (-1)**(n)/(2*n+1)
        if (type(x)==sympy.S.Zero):   
            return (-1)**(n)/(2*n+1)
        return (-1)**n*BOYS(n,x)
    # Define FMA.rewrite(Add) and FMA.rewrite(Mul).
    def _eval_rewrite(self, rule, args, **hints):
        n, x= self.args
        if rule in [Add, Mul]:
            return self.doit()
    # Define differentiation.
    def fdiff(self, argindex):
        # argindex indexes the args, starting at 1
        n, x= self.args
        if argindex == 1:
            return 0
        elif argindex == 2:
            return MyFN(n+1,x)
    # Define code printers for ccode() and cxxcode()
    def _ccode(self, printer):
        n, x = self.args
        _n, _x = printer._print(n), printer._print(x)
        return "fma(%s, %s)" % (_n, _x)
    def _cxxcode(self, printer):
        n, x = self.args
        _n, _x = printer._print(n), printer._print(x)
        return "std::fma(%s, %s)" % (_n,_x)


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


TNSTRING=''
for i in range(11):
    TNSTRING+= 'TN'+str(i)+' '
TNS=symbols(TNSTRING)
def TN(N):
    return TNS[N]



     


# In[ ]:


AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
z1,z2,z3,z4=symbols("z1 z2 z3 z4")
ISAB=S000000(z1,z2,[AX,AY,AZ],[BX,BX,BZ])
IKAB=K000000(z1,z2,[AX,AY,AZ],[BX,BY,BZ])
ISABZZ=ISAB.diff(AZ).diff(BZ)/z1/z2/4 #(PZ;A|PZ;B)
IKABZZ=IKAB.diff(AZ).diff(BZ)/z1/z2/4 #(PZ;A|-1/2 \nabra^2| PZ;B)
fssss=T0000(z1,z2,z3,z4,[AX,AY,AZ],[BX,BY,BZ],[CX,CY,CZ],[DX,DY,DZ])


# In[ ]:


from sympy import lambdify
AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
z1,z2,z3,z4=symbols("z1 z2 z3 z4")
vssss=lambdify([z1,z2,z3,z4,AX, AY ,AZ, BX ,BY, BZ, CX, CY, CZ, DX, DY, DZ],fssss)


# In[ ]:


vssss(1,1,1,1,2,1,3,1,2,3,3,2,1,2,3,1)


# In[ ]:


import numpy
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






NORBITALS=5
COEF=[[1.0,0.0,0.0],[0.678914,0.430129,0.0],[0.444635,0.535328,0.154329]] 
EXPON=[[0.270950,0.0,0.0],[0.151623,0.851819,0.0],[0.109818,0.405771,2.22766]] 
EXPOn2=[[0.0,0.0,0.0],[0.0,0.0,0.0],[
0.9942,
0.23103,
0.075139
]]
COEFn2s=[[0.0,0.0,0.0],[0.0,0.0,0.0],[
    -0.099967,
    0.39951,
    0.70012
]]
COEFn2p=[[0.0,0.0,0.0],[0.0,0.0,0.0],[
0.15592,
0.60768,
0.39196,
]]
#R_length=1.4632
atomA=[0,0,0]
NORBITALS=5
atoms=[atomA,atomA,atomA,atomA,atomA]
EXPOS=[EXPON[2],EXPOn2[2],EXPOn2[2],EXPOn2[2],EXPOn2[2]]
COEFS=[COEF[2],COEFn2s[2],COEFn2p[2],COEFn2p[2],COEFn2p[2]]
ZETAS=[5.67,1.72,1.72,1.72,1.72]
#EXPOS=[EXPON[0],EXPON[0],EXPON[0],EXPON[0],EXPON[0]]
#COEFS=[COEF[0],COEF[0],COEF[0],COEF[0],COEF[0]]

ORBITKEYS=[(0,0,0,0),(0,0,0,1),(1,0,0,0),(0,1,0,0),(0,0,1,0)]

ZS=[6]

DA=[0]*3
CA=[0]*3
DB=[0]*3
CB=[0]*3
DC=[0]*3
CC=[0]*3
DD=[0]*3
CD=[0]*3
PI=numpy.pi


# In[ ]:


#
#  Compute the formulas of two-electron integrals where s, px, py, and pz are used.
#  Symbolic differentiation is applied to [s(A)s(B)|s(C)s(D)](=:fssss)
#
#  IntegralsFormulasList: The list of the integrals (Dictionary)
#                         with keys (i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3)
#  (i1,i2,i3) -> (x-Ax)^(i1)*(y-Ay)^(i2)*(z-Az)^(i3)*exp(-z1|r-A|^2)
#
def GetTwoEIntegral(key1,key2,key3,key4,Ilist,itype=0):
    AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
    z1,z2,z3,z4=symbols("z1 z2 z3 z4")
    fml=fssss
    keys1234=key1+key2+key3+key4

    if Ilist.get(keys1234)!=None:
        print("Found Formula!",len(Ilist))
        return Integralslist.get(keys1234)
    print("Compute Formula..",len(Ilist))
    denom=1
    for ifl,var in zip([key1,key2,key3,key4],[z1,z2,z3,z4]):
        for jfl in ifl:
            if jfl==1:
                denom *= (2*var)
                       
    for ifl,var in zip(list(keys1234),[AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ]):
        if ifl==1:
            fml=fml.diff(var)
    fml=fml/denom
    if itype==1:
        print("lambdify")
        vml=lambdify([z1,z2,z3,z4,AX, AY ,AZ, BX ,BY, BZ, CX, CY, CZ, DX, DY, DZ],fml)
    else:
        vml=fml
    #Integralslist[keys1234]=vml
    return vml

IntegralsFormulasList=dict()
IntegralsFormulasList[(0,0,0,0,0,0,0,0,0,0,0,0)]=fssss
setofkeys=[]
ORBITKEYSSP=[(0,0,0),(1,0,0),(0,1,0),(0,0,1)]
for key1 in ORBITKEYSSP:
    for key2 in ORBITKEYSSP:
        for key3 in ORBITKEYSSP:
            for key4 in ORBITKEYSSP:
                setofkeys.append(indexijkl(key1,key2,key3,key4))
                nk1,nk2,nk3,nk4=indexijkl(key1,key2,key3,key4)
                keys1234=nk1+nk2+nk3+nk4
                if IntegralsFormulasList.get(keys1234)==None:
                    vml=GetTwoEIntegral(key1,key2,key3,key4,IntegralsFormulasList,itype=0)
                    IntegralsFormulasList[keys1234]=vml
 

   
                    
                    
                


# In[ ]:


#
#  Computation of all of the two-electron integrals for every combination of the orbitals.
#  Note that we compute only the representative of [pq|rs], 
#    for there are equivalences concerning the permutation of p,q,r,s.
#    We use indexijkl(i,j,k,l) for any index i,j,k, and l, to get the minimum in the equivalent elements. 
#
#
TTlist=dict()
Flist=dict()
import copy
def ComputeAllTwoERI():
    AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
    z1,z2,z3,z4=symbols("z1 z2 z3 z4")
    T=symbols("T", positive=True)

    #TT=[[[[0 for _ in range(3)] for _  in range(3)] for _ in range(3)] for _ in range(3)]
    
    for ID in range(NORBITALS):
        for JD in range(NORBITALS):
            for KD in range(NORBITALS):
                for LD in range(NORBITALS):
                    #print(ID,JD,KD,LD)
                    #RRA=[AX,AY,AZ]=atoms[ID]
                    #RRB=[BX,BY,BZ]=atoms[JD]
                    #RRC=[CX,CY,CZ]=atoms[KD]
                    #RRD=[DX,DY,DZ]=atoms[LD]
                    RRA=copy.deepcopy(atoms[ID])
                    RRB=copy.deepcopy(atoms[JD])
                    RRC=copy.deepcopy(atoms[KD])
                    RRD=copy.deepcopy(atoms[LD])
                    RRB[0]+=T
                    RRC[1]+=T
                    RRD[2]+=T

                    key1=ORBITKEYS[ID]
                    key2=ORBITKEYS[JD]
                    key3=ORBITKEYS[KD]
                    key4=ORBITKEYS[LD]
                    #print(key1,key2,key3,key4)
                    ik1=key1[:3]
                    ik2=key2[:3]
                    ik3=key3[:3]
                    ik4=key4[:3]
                    #print(ik1,ik2,ik3,ik4)
                    nk1,nk2,nk3,nk4=indexijkl(ik1,ik2,ik3,ik4)
                    fml=IntegralsFormulasList.get(nk1+nk2+nk3+nk4)
                    if None==fml:
                        print("\nFormula not found",nk1,nk2,nk3,nk4)
                    else:
                        print("\nFormula found",nk1,nk2,nk3,nk4)
                        

                    #print(RRA+RRB+RRC+RRD,vml)                    
                    if TTlist.get(indexijkl(ID,JD,KD,LD))==None:
                        vml=copy.deepcopy(fml)    
                        for par,var in zip(RRA+RRB+RRC+RRD,[AX,AY,AZ,BX,BY
                                                            ,BZ,CX,CY,CZ,DX,DY,DZ]):
                            vml=vml.subs(var,par)
                        print("I compute [",ID,JD,"|",KD,LD,"]")
                        N=3
                        for i in range(N):
                            #print(i,EXPON[N-1][i],COEF[N-1][i],ZETA1,ZETA2)
                            CA[i]=EXPOS[ID][i]*ZETAS[ID]**2
                            DA[i]=COEFS[ID][i]*((2.0*CA[i]/PI)**0.75)
                            CB[i]=EXPOS[JD][i]*(ZETAS[JD]**2)
                            DB[i]=COEFS[JD][i]*((2.0*CB[i]/PI)**0.75)
                            CC[i]=EXPOS[KD][i]*(ZETAS[KD]**2)
                            DC[i]=COEFS[KD][i]*((2.0*CC[i]/PI)**0.75)
                            CD[i]=EXPOS[LD][i]*(ZETAS[LD]**2)
                            DD[i]=COEFS[LD][i]*((2.0*CD[i]/PI)**0.75)
                            if ik1 in [(1,0,0),(0,1,0),(0,0,1)]:
                                DA[i]=COEFS[ID][i]*((2.0*CA[i]/PI)**0.75)*2*CA[i]**0.5
                            if ik2 in [(1,0,0),(0,1,0),(0,0,1)]:
                                DB[i]=COEFS[JD][i]*((2.0*CB[i]/PI)**0.75)*2*CB[i]**0.5
                            if ik3 in [(1,0,0),(0,1,0),(0,0,1)]:
                                DC[i]=COEFS[KD][i]*((2.0*CC[i]/PI)**0.75)*2*CC[i]**0.5
                            if ik4 in [(1,0,0),(0,1,0),(0,0,1)]:
                                DD[i]=COEFS[LD][i]*((2.0*CD[i]/PI)**0.75)*2*CD[i]**0.5                           
 


                        N=3
                        V=0
                        VW=0
                        for I in range(N):
                            ca=CA[I]
                            vmlI=copy.deepcopy(vml)
                            vmlI=vmlI.subs(z1,ca)
                            for J in range(N):
                                cb=CB[J]
                                vmlJ=copy.deepcopy(vmlI)
                                vmlJ=vmlJ.subs(z2,cb)
                                for K in range(N):
                                    cc=CC[J]
                                    vmlK=copy.deepcopy(vmlJ)
                                    vmlK=vmlK.subs(z3,cb)
                                    for L in range(N):
                                        print(".", end="")
                                        cd=CD[L]
                                        vmlL=copy.deepcopy(vmlK)
                                        vmlL=vmlL.subs(z4,cc)

                                        VP=vmlL
      
                                        V=V+VP*DA[I]*DB[J]*DC[K]*DD[L]
                        TTlist[indexijkl(ID,JD,KD,LD)]=V
                        
ComputeAllTwoERI()


# In[ ]:


str(TTlist[(4,4,4,4)])[:1000]


# In[ ]:


import copy
TTlist_save=copy.deepcopy(TTlist)

def Pruning(expr):
#
#
#   To get the constant term from a power series which include negative exponent terms.
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

AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
z1,z2,z3,z4=symbols("z1 z2 z3 z4")
T=symbols("T", positive=True)

#
#   Some integrals remain as formulas, because the uncautious numerical substitutions 
#   shall cause the false simgularity in them.
#   We elicit and remove the false singular part from those integrals to evaluate them properly.
#
from tqdm.notebook import tqdm
for vk in tqdm(TTlist.keys()):
    v=TTlist.get(vk)
    if v.is_Number==False:
        v=v.expand()
        #print(vk,v)
        #T2=sympy.sympify(str(v).replace("exp","sympy.exp").replace("erf","sympy.erf"))
        try:
            w=sympy.series(v,T,0)
            w=sympy.N(w)
            w=w.removeO()
            #print("\n",vk,w)
            Pw=Pruning(w)
            #print("-->",Pw)
            TTlist[vk]=Pw
        except:
            print("error")
            pass


# In[ ]:


#
# The computation just before would return the following result.
#
D1_=[13.6005402665707, 0.477244991446140, 0, 0, 0, 0.0229990206856726, 0, 0, 0, 0.0519175975243284, 0, 0, 0.0519175975243283, 0, 0.0519175975243283, -0.0169732887821173, 0, 0, 0, 0.127969449422991, 0, 0, 0, 3.97735891302247, 0, 0, 3.97735891302247, 0, 3.97735891302247, 0.0383141411800388, 0, 0, 0, 0.0248484870289021, 0, 0, 0, 0, 0, 0, 0, 0, 0.0383141411800388, 0, 0, 0, 0.0248484870289021, 0, 0, 0, 0, 0, 0, 0, 0.0383141411800388, 0, 0, 0, 0.0248484870289021, 0, 0, 0, 0, 0, 0, 0.379921021388003, 0, 0, 0, 15.1288455925001, 0, 0, 15.1288455925001, 0, 15.1288455925001, 0.267144049220419, 0, 0, 0, 0, 0, 0, 0, 0, 0.267144049220419, 0, 0, 0, 0, 0, 0, 0, 0.267144049220419, 0, 0, 0, 0, 0, 0, 8.31606699556591, 0, 0, 7.36344342811287, 0, 7.36344342811287, 0.476311783726522, 0, 0, 0, 0, 0.476311783726528, 0, 0, 0, 8.31606699556591, 0, 7.36344342811286, 0.476311783726521, 0, 8.31606699556597]


# In[ ]:


D1=list()
for ky in TTlist.keys():
    print(ky,sympy.N(TTlist[ky]))
    D1.append(sympy.N(TTlist[ky]))


# In[ ]:


print(D1)


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


def S000000(A,B,RA,RB):
#
#   OVERLAP INTEGRAL between exp(-A*|r-RA|^2) and exp(-B*|r-RB|^2)
#
    RAB2=sum([(c1-c2)**2 for c1,c2 in zip(RA,RB)])
    PI=sympy.pi
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



SIntegralslist=dict()
KIntegralslist=dict()
VABCIntegralslist=dict()
VAAAIntegralslist=dict()

Sss=S000000(z1,z2,[AX,AY,AZ],[BX,BY,BZ])
Kss=K000000(z1,z2,[AX,AY,AZ],[BX,BY,BZ])
Vss=V000000(z1,z2,[AX,AY,AZ],[BX,BY,BZ],[CX,CY,CZ])
VAAA=V000000(z1,z2,[AX,AY,AZ],[AX,AY,AZ],[AX,AY,AZ])
VABC=V000000(z1,z2,[AX,AY,AZ],[BX,BY,BZ],[CX,CY,CZ])


# In[ ]:


VABC_ZZ=VABC.diff(AZ).diff(BZ)/z1/z2/4


# In[ ]:


AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ") 
z1,z2=symbols("z1 z2")

[Sss.subs(BX,AX).subs(BY,AY).subs(BZ,AZ),Sss.subs(BX,AX).subs(BY,AY).subs(BZ,AZ).subs(z2,z1)]


# In[ ]:


Sss.diff(AX).diff(BX).subs(BX,AX).subs(BY,AY).subs(BZ,AZ)


# In[ ]:


#
#  Substitution to 'expr' 
#
def TTZZZZE(expr,ca,cb,cc,cd,RA,RB,RC,RD,IOP=0):
    AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
    z1,z2,z3,z4=symbols("z1 z2 z3 z4")
    T=symbols("T",positive=True)
    if IOP!=0:
        print(RA,RB,RC,RD)
        print(ca,cb,cc,cd)
    symbolslist=[z1,z2,z3,z4,AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ]
    valueslist=[ca,cb,cc,cd]
    for RS in [RA,RB,RC,RD]:
        valueslist.extend(RS)
    V=copy.deepcopy(expr)
    for s,v in zip(symbolslist,valueslist):
        V=V.subs(s,v)
    return V
        


# In[ ]:


def GetSintegral(key1,key2):
    AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
    z1,z2,z3,z4=symbols("z1 z2 z3 z4")
    fml=Sss
    keys12=key1+key2
    if SIntegralslist.get(keys12)!=None:
        return SIntegralslist.get(keys12)
    denom=1
    for ifl,var in zip([key1,key2],[z1,z2]):
        for jfl in ifl:
            if jfl==1:
                denom *= (2*var)
    for ifl,var in zip(list(keys12),[AX,AY,AZ,BX,BY,BZ]):
        if ifl==1:
            fml=fml.diff(var)
    fml=fml/denom
    SIntegralslist[keys12]=fml
    return fml

def GetKintegral(key1,key2):
    AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
    z1,z2,z3,z4=symbols("z1 z2 z3 z4")
    fml=Kss
    keys12=key1+key2
    if KIntegralslist.get(keys12)!=None:
        return KIntegralslist.get(keys12)
    denom=1
    for ifl,var in zip([key1,key2],[z1,z2]):
        for jfl in ifl:
            if jfl==1:
                denom *= (2*var)
    for ifl,var in zip(list(keys12),[AX,AY,AZ,BX,BY,BZ]):
        if ifl==1:
            fml=fml.diff(var)
    fml=fml/denom
    KIntegralslist[keys12]=fml
    return fml

def GetVintegral(key1,key2):
    AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
    z1,z2,z3,z4=symbols("z1 z2 z3 z4")
    fml=Vss
    keys12=key1+key2
    if VABCIntegralslist.get(keys12)!=None:
        return VABCIntegralslist.get(keys12)
    denom=1
    for ifl,var in zip([key1,key2],[z1,z2]):
        for jfl in ifl:
            if jfl==1:
                denom *= (2*var)
    for ifl,var in zip(list(keys12),[AX,AY,AZ,BX,BY,BZ]):
        if ifl==1:
            fml=fml.diff(var)
    fml=fml/denom
    VABCIntegralslist[keys12]=fml
    return fml


def GetVAAAintegral(key1,key2):
    
    AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
    z1,z2,z3,z4=symbols("z1 z2 z3 z4")
    T=symbols("T",positive=True)
    keys12=key1+key2
    if VAAAIntegralslist.get(keys12)!=None:
        return VAAAIntegralslist.get(keys12)
 
    fml=Vss
    print(fml)
    denom=1
    for ifl,var in zip([key1,key2],[z1,z2]):
        for jfl in ifl:
            if jfl==1:
                denom *= (2*var)
    for ifl,var in zip(list(keys12),[AX,AY,AZ,BX,BY,BZ]):
        if ifl==1:
            fml=fml.diff(var)
    fml=fml/denom
    fml=fml.subs(z2,z1).subs(BX,AX).subs(BY,AY).subs(BZ,AZ).subs(CX,AX).subs(CY,AY).subs(CZ,AZ+T)
    VAAAIntegralslist[keys12]=fml
    return fml


def ComputeOneERI():
    VAB_C=[[[0 for _ in range(NORBITALS)] for _  in range(NORBITALS)] for _ in range(NORBITALS)]
    for ID in range(NORBITALS):
        for JD in range(NORBITALS):
            for KD in range(len(ZS)):
                    print("\n",ID,JD,KD,"\n")
                    AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
                    z1,z2,z3,z4=symbols("z1 z2 z3 z4")
                    T=symbols("T",positive=True)
                    RA=[AX,AY,AZ]=atoms[ID]
                    RB=[BX,BY,BZ]=atoms[JD]
                    RC0=[CX,CY,CZ]=atoms[KD]

                    if ID==JD==KD:
                        RC=[CX+T,CY+T,CZ+T]
                    if not (ID==JD==KD):
                        RC=[CX,CY,CZ]   
                    RC=[CX,CY,CZ]
                    key1=ORBITKEYS[ID]
                    key2=ORBITKEYS[JD]
                    key3=ORBITKEYS[KD]
                    nkey1=key1[:3]
                    nkey2=key2[:3]
                    nkey3=key3[:3]
                    #print(key1,key2)
                    formula=0
                    formula=GetVintegral(nkey1,nkey2)
                    #print(formula==VABC)
                    #print(ID,JD,KD,key1,key2,formula)
                    ZC=ZS[KD]
                    N=3
                    for i in range(N):
                        #print(i,EXPON[N-1][i],COEF[N-1][i],ZETA1,ZETA2)

                        CA[i]=EXPOS[ID][i]*ZETAS[ID]**2
                        DA[i]=COEFS[ID][i]*((2.0*CA[i]/PI)**0.75)
                        CB[i]=EXPOS[JD][i]*(ZETAS[JD]**2)
                        DB[i]=COEFS[JD][i]*((2.0*CB[i]/PI)**0.75)

                        if nkey1 in [(1,0,0),(0,1,0),(0,0,1)]:
                            DA[i]=COEFS[ID][i]*((2.0*CA[i]/PI)**0.75)*2*CA[i]**0.5
                        if nkey2 in [(1,0,0),(0,1,0),(0,0,1)]:
                            DB[i]=COEFS[JD][i]*((2.0*CB[i]/PI)**0.75)*2*CB[i]**0.5

                    N=3
                    V=0
                    for I in range(N):            
                        for J in range(N):
                                ca=CA[I]
                                cb=CB[J]
 
                                V+=V000000(ca,cb,RA,RB,RC)*DA[I]*DB[J]*(-ZC)
 
                    V=sympy.N(V)
                    #print(RA,RB,RC,RD,V)
                    VAB_C[ID][JD][KD]=V

    SAB=[[0 for _ in range(NORBITALS)] for _  in range(NORBITALS)]              
    for ID in range(NORBITALS):
        for JD in range(NORBITALS):
            RA=[AX,AY,AZ]=atoms[ID]
            RB=[BX,BY,BZ]=atoms[JD]
            key1=ORBITKEYS[ID]
            key2=ORBITKEYS[JD]
            nkey1=key1[:3]
            nkey2=key2[:3]
            formula=GetSintegral(nkey1,nkey2)
            N=3
            for i in range(N):
 
                CA[i]=EXPOS[ID][i]*ZETAS[ID]**2
                DA[i]=COEFS[ID][i]*((2.0*CA[i]/PI)**0.75)
                CB[i]=EXPOS[JD][i]*(ZETAS[JD]**2)
                DB[i]=COEFS[JD][i]*((2.0*CB[i]/PI)**0.75)
                if nkey1 in [(1,0,0),(0,1,0),(0,0,1)]:
                    DA[i]=COEFS[ID][i]*((2.0*CA[i]/PI)**0.75)*2*CA[i]**0.5
                if nkey2 in [(1,0,0),(0,1,0),(0,0,1)]:
                    DB[i]=COEFS[JD][i]*((2.0*CB[i]/PI)**0.75)*2*CB[i]**0.5

            N=3
            V=0
            for I in range(N):            
                for J in range(N):
                            ca=CA[I]
                            cb=CB[J]
                            #V000000(a,b,RA,RB,RC)
                            #V=V+S000000(ca,cb,RA,RB)*DA[I]*DB[J]
                            V=V+TTZZZZE(formula,ca,cb,0,0,RA,RB,[0,0,0],[0,0,0])*DA[I]*DB[J]
            V=sympy.N(V)
            #print(RA,RB,V)
            SAB[ID][JD]=V

    KAB=[[0 for _ in range(NORBITALS)] for _  in range(NORBITALS)]              
    for ID in range(NORBITALS):
        for JD in range(NORBITALS):
            RA=[AX,AY,AZ]=atoms[ID]
            RB=[BX,BY,BZ]=atoms[JD]
            key1=ORBITKEYS[ID]
            key2=ORBITKEYS[JD]
            nkey1=key1[:3]
            nkey2=key2[:3]
            formula=GetKintegral(nkey1,nkey2)
            N=3
            for i in range(N):
                #print(i,EXPON[N-1][i],COEF[N-1][i],ZETA1,ZETA2)
                CA[i]=EXPON[N-1][i]*ZETAS[ID]**2
                DA[i]=COEF[N-1][i]*((2.0*CA[i]/PI)**0.75)
                CB[i]=EXPON[N-1][i]*(ZETAS[JD]**2)
                DB[i]=COEF[N-1][i]*((2.0*CB[i]/PI)**0.75)
                CA[i]=EXPOS[ID][i]*ZETAS[ID]**2
                DA[i]=COEFS[ID][i]*((2.0*CA[i]/PI)**0.75)
                CB[i]=EXPOS[JD][i]*(ZETAS[JD]**2)
                DB[i]=COEFS[JD][i]*((2.0*CB[i]/PI)**0.75)
                if nkey1 in [(1,0,0),(0,1,0),(0,0,1)]:
                    DA[i]=COEFS[ID][i]*((2.0*CA[i]/PI)**0.75)*2*CA[i]**0.5
                if nkey2 in [(1,0,0),(0,1,0),(0,0,1)]:
                    DB[i]=COEFS[JD][i]*((2.0*CB[i]/PI)**0.75)*2*CB[i]**0.5
            
            N=3
            V=0
            for I in range(N):            
                for J in range(N):
                            ca=CA[I]
                            cb=CB[J]
                            #V000000(a,b,RA,RB,RC)
                            #V=V+K000000(ca,cb,RA,RB)*DA[I]*DB[J]
                            V=V+TTZZZZE(formula,ca,cb,0,0,RA,RB,[0,0,0],[0,0,0])*DA[I]*DB[J]
            V=sympy.N(V)
            #print(RA,RB,V)
            KAB[ID][JD]=V
    return SAB,KAB,VAB_C

SIntegralslist=dict()
KIntegralslist=dict()
VABCIntegralslist=dict()
VAAAIntegralslist=dict()            
SAB,KAB,VAB_C=ComputeOneERI()


# (VAB_C
# -4.13982720469660
# -0.677230081932664
# -1.10291256788737
# -0.411305470289750
# -1.10291256788737
# -0.411305470289749
# -1.26524587342335
# -1.22661546805825

# In[ ]:


SAB


# In[ ]:


for i in range(2):
    for j in range(2):
        for k in range(2):
            print(VAB_C[i][j][k])


# In[ ]:


for i in range(2):
    for j in range(2):
        for k in range(2):
            print(VAB_C[i][j][k].subs(R,1.463))


# In[ ]:


expr=0.059915339049473*sympy.sqrt(2)*sympy.pi**(3/2)*sympy.erf(2.32188700396897*sympy.sqrt(2)*sympy.sqrt(T**2))/sympy.sqrt(T**2 )


# In[ ]:


VAAA=V000000(z1,z2,[AX,AY,AZ],[AX,AY,AZ],[AX,AY,AZ+T])


# In[ ]:


sympy.N(sympy.series(expr,T,0))


# In[ ]:


VABCIntegralslist.get((0,0,0,0,0,0))


# In[ ]:


VABC


# In[ ]:


AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
z1,z2,z3,z4=symbols("z1 z2 z3 z4")
V000000(z1,z2,[AX,AY,AZ],[BX,BY,BZ],[CX,CY,CZ])


# In[ ]:


[GetVintegral((0,0,1),(0,0,1))-VABC_ZZ,
IKABZZ - GetKintegral((0,0,1),(0,0,1)),
ISABZZ - GetSintegral((0,0,1),(0,0,1))]


# In[ ]:


GetVintegral((0,0,0),(0,0,0))


# In[ ]:


for i in range(len(SAB)):
    print(SAB[i])


# In[ ]:


IOP=2
N=3
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
            P[I][J]=lcao[I]*lcao[J]*2+lcao2[I]*lcao2[J]*0

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
        for j in range(i+1,NORBITALS):
            RPQ=[(c1-c2)**2 for c1,c2 in zip(atoms[i],atoms[j])]
            RPQ2=sum(RPQ)
            #print(RPQ2)
            ENT+=ZS[i]*ZS[j]/sympy.sqrt(RPQ2)
            


    return EN,ENT,F,F2,H,P
x,y,z,u,v,w=symbols("x y z u v w")
EN,ENT,FM,FM2,HM,PM=SCF_SIMBOL3(IOP,N,R,ZETA1,ZETA2,ZA,ZB)


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


lcao=[x,y]
lcao2=[p,q]
SPART=series(GetNormS(lcao,SAB)-1,R,1.5).removeO().expand()
SPART2=series(GetNormS(lcao2,SAB)-1,R,1.5).removeO().expand()
SPART3=series(vSw(lcao,SAB,lcao2),R,1.5).removeO().expand()


# In[ ]:


OBJE=sympy.N(EPART-2*e*SPART)
OBJ=OBJE.expand()


# In[ ]:


import copy
OBJ0=copy.deepcopy(OBJ)
#One can try various electronic configurations:
# LCAFO alpha(x,x,x,x); LCAO beta (p,p,-p,-p)
#OBJ=OBJ0.subs([(y,x),(u,x),(w,x),(q,p),(r,-p),(s,-p)])

#One can try various electronic configurations:
# LCAFO alpha(x,y,u,w); LCAO beta (0,0,0,0)
#OBJ=OBJ0.subs([(p,0),(q,0),(r,0),(s,0),(t,0),(f,0)])


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


getF


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

