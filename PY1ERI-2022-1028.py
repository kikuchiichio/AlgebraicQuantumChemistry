#!/usr/bin/env python
# coding: utf-8

# # One-Electron Integrals through Obara-Saika recursion method.
# 
# ## We compute functions of the form 
# 
# 
# 
# Let capital letters A, B,and C represent the centers of orbitals and their coordinates. 
# 
# We use several variables in the following.
# $
# p=a+b
# $
# 
# $
# P=(a\cdot A+ b\cdot B)/(a+b)
# $
# 
# $
# PC=P-C,\ AB=A-B,...,{\rm and\ so\ on}.
# $
# 
# With the Gaussian-type orbitals:
# 
# $
# \phi_A(r)=(x-Ax)^{i1}(y-Ay)^{i2}(z-Az)^{i2}\exp(-a|r-A|^2)
# $
# 
# and
# 
# $
# \phi_B(r)=(x-Bx)^{j1}(y-By)^{j2}(z-Vz)^{j3}\exp(-b|r-B|^2)
# $
# 
# we compute the one-center integral
# 
# $
# V_{AB;C} = \int dr \phi_A(r) \phi_B(r)\frac{1}{|r-r_C|}
# $
# 
# We use auxilary integrals $I(N;i1,i2,i3,j1,j2,j3)$ such that
# $
#  I(0;i1,i2,i3,j1,j2,j3)=V_{AB;C}
# $
# 
# 
# 
# We also use the symbolic differentiation to compute the integrals. The keys $(i1,i2,i3,j1,j2,j3)$ shall indicate the differentiation.
# 
# The term $(i1,i2,i3)$ is the insruction to for differention with respect to the first orbital at (Ax,Ay,Az),
# whereby 
# (0,1,0) corresponds to d/dAy; (1,1,0) to d^2/dAx/dAy, and so on.
#     
# Likewise
# 
# $(j1,j2,j3)$: the differention with respect to the second orbital.
# 
# 
# 
# 
# ## We use several functions to conduct recursive computations.
# 
# In the folllowing, we omit the subscript index (1,2,3) which represents (x,y,z). 
# 
# (i+1) means the shift from i with the increasement to one of the coordinate by 1; the direction of the increasment shall be clear in the contexts. 
# 
# ##  \<Vertical1E\> puts forth the integrals in this way:  
# 
# 
# In X direction, we use
# $$
# \theta^N_{i+1,j}=X_{PA}\theta^N_{i,j}+\frac{1}{2p}(i\theta^N_{i+1,j}+j\theta^N_{i,j-1})
# -X_{PC}\theta^{N+1}_{i,j}-\frac{1}{2p}(i\theta^{N+1}_{i+1,j}+j\theta^{N+1}_{i,j-1})
# $$
# 
# 
# ## \<Horizontal1E\> puts forth the integrals in this way:
# 
# 
# In X direction, we use
# $$
# \theta^N_{i,j+1}=\theta^N_{i+1,j}+X_{AB}\theta^N_{i,j}
# $$
# 
# 
# 
# ## Computational Steps
# 
# 1. <1> we prepare I(N;0,0,0,0,0,0) for N=0,...,Nmax
# 
# 1. <2> For any I(N;i1,i2,i3,j1,j2,j3), we try the above formulas to see if the integral is compted from the other integrals which have already been computed. We use the function <late_evaluation>.
# 
# 1. <3> If all necessary ingredients are prepared, we compute the integral.
# 
# 1. <4> If some of the necessary ingredients are lacking, we request their computations in the following computations.
# 
# 1. <5> Then we check and compute the requestedintegrals, by repeating <2>--<4>; we might request more of not-yet-computed integrals. 
# 
# 1. <C1> In the computation, we write and rewrite the list of requested1E integrals and that of computed ones.
# 
# 1. <C2> The recusive process finally touches its bottom, where it shall find {I(N;0,0,0,0,0,0),N=1,...Nmax}. Hence it terminates after a finite numer of computations.
# 
# For example, we compute as follows.
# 
#     
# 
# 
# > requested1E_keys=dict() # Keys of the requested1E integrals
#     
# > integrals1E_all=dict()  # List of the computed integrals.
# 
# > NMAX=11
#     
# > for i in range(NMAX):
# 
# >    integrals1E_all[(i,0,0,0,0,0,0,0)]=TN(i)    # I(i,0,0,0,0,0,0),i=1,...NMAX
# 
# > ckeys=(0,0,0,0,0,0,0,2)  
# 
# > late_evaluate1E(ckeys)  # Try once to compute the integrals at <ckeys>.  
# 
# 
# > CurrentN=list(ckeys)[0]
# 
# > import copy
#     
# > IsFound=None
# 
# > while(IsFound==None and CurrentN<NMAX):
#                                           
# >    requested1E_keys_copy=copy.deepcopy(requested1E_keys)
#     
# >    for ckey in requested1E_keys_copy.keys(): # Loop over the requested1E integrals to compute them
#     
# >        if min(list(ckey))>=0:
#     
# >            F=late_evaluate1E(ckey)
#     
# >            if F==1:
#     
# >                print("\n\nCOMPUTED",ckey,integrals1E_all.get(ckey))
#     
# >                requested1E_keys.pop(ckey) # Remove the key from the list of the requested1E integrals
# 
# >    late_evaluate1E(ckeys)  # Try to compute again the integral at <ckeys>.
#     
# >    IsFound=integrals1E_all.get(ckeys)  # Is the integral computed or not?
#     
# >    if IsFound!=None:
#     
# >        print(ckeys, "is computed")
#     
# > print(IsFound)  
# 
# 
# 
# The integrals are numerically evaluated in two ways: 
#     
# 1.   (W1) By Obara-Saika recurison mehod
# 2.   (W2) By formulas made by symbolic differentiation of I(0, 0,0,0, 0,0,0)
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
    PI=sympy.pi
    #print(A,B,RAB2,(PI/(A+B))**1.5)
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
    PI=sympy.pi
    return A*B/(A+B)*(3-2*A*B*RAB2/(A+B))*(PI/(A+B))**1.5*sympy.exp(-A*B*RAB2/(A+B))   

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


# In[ ]:


#
#  TEST
#
AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")
z1,z2,z3,z4=symbols("z1 z2 z3 z4")
#f=T0000(z1,z2,z3,z4,[AX,AY,AZ],[BX,BY,BZ],[CX,CY,CZ],[DX,DY,DZ])



#
# TEST
#
#((S000000(z1,z2,[AX,AY,AZ],[BX,BY,BZ]).diff(AX)).diff(AX)).subs([(BX,AX),(BY,AY),(BZ,AZ)])

from sympy import Function, Symbol
from sympy import Number


## Obara-Saika Scheme for Two-electron integrals
#
# We prepare dummy expressions for THE FUNCTIONS TN00(N,a,b,RA,RB,RC): TNS[0],TNS[1],...., WHICH SHALL BE USED LATER
#  THEY ARE AUXILLARY INTEGRALS; IF N=0 ,THEY ARE THE ONE-ELECTRON INTGRALS.
#

TNSTRING=''
for i in range(11):
    TNSTRING+= 'TN'+str(i)+' '
TNS=symbols(TNSTRING)
def TN(N):
    return TNS[N]



    
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
XPC, YPC, ZPC=symbols("XPC YPC ZPC")
#
#
#
#  We put p=a*b/(a+b).
#  Provisionally pare left as they are without substitution.
#  At the end of the computation, we have to replace them.
#
a,b,c,d=symbols("a b c d")
p,q=symbols("p q")

#
#  We compute the integrals using general Cartesian Gaussian functions:
#          (i,j,k,a,RA):=(x-XA)^i (y-YA)^j (z-ZA)^j exp(-a|r-RA|^2)
#
#  VIZ.
#   The 3D intrgral of  (i1,j1,k1,a,RA)*(i2,j2,k2,b,RB)*1/|r-rC||
#
#  For brevity, we write:
#   i=(i1,i2,i3), j=(j1,j2,j3),
# 
#  To compute the auxillary integdals T(N,i,j) we lift from T(N,i=0.j=0)
#  WHEN N=0 THEY GIVE THE ONE-ELECTRON INTEGRALS; OBARA-SAIKA RECURSION USES THEM WITH N>=1.
#
#  BY VERTICAL RECUSION IN OBARA-SAIKA, WE COMPUTE 
#  (N: i+1 j ) from (N: i j) , (N i-1 j ) , (N i j-1), (N+1 i j) , (N+1 i-1 j )  (N+1 i j-1)
#
#  WHERE i+1 is the every possible move from (i1,i2,i3) by adding one or zero to each of (i1,i2,i3), 
#  except zero-move. 
#
#  THE FOLLOWING FUNCTION LIFTS UP IN D=X/Y/Z DIRECTIONS AND COMPUTES THE INTGRALS.

def VERTICAL1E(D,N,i1,i2,i3,j1,j2,j3,INFO=0):
    alpha=p*q/(p+q)
    
    if D=="X":
        #print("X")
        key0=(N,i1,i2,i3,j1,j2,j3)
        key1=(N,i1-1,i2,i3,j1,j2,j3)
        key2=(N,i1,i2,i3,j1-1,j2,j3)
        key3=(N+1,i1,i2,i3,j1,j2,j3)
        key4=(N+1,i1-1,i2,i3,j1,j2,j3)
        key5=(N+1,i1,i2,i3,j1-1,j2,j3)
        keyn=(N,i1+1,i2,i3,j1,j2,j3)
        i_d,j_d=i1,j1
        #print(keyn)
        DPA=XPA
        DPC=XPC
    if D=="Y":
        #print("Y")
        key0=(N,i1,i2,i3,j1,j2,j3)
        key1=(N,i1,i2-1,i3,j1,j2,j3)
        key2=(N,i1,i2,i3,j1,j2-1,j3)
        key3=(N+1,i1,i2,i3,j1,j2,j3)
        key4=(N+1,i1,i2-1,i3,j1,j2,j3)
        key5=(N+1,i1,i2,i3,j1,j2-1,j3)
        keyn=(N,i1,i2+1,i3,j1,j2,j3)
        i_d,j_d=i2,j2
        #print(keyn)
        DPA=YPA
        DPC=YPC
    if D=="Z":
        #print("Z")
        key0=(N,i1,i2,i3,j1,j2,j3)
        key1=(N,i1,i2,i3-1,j1,j2,j3)
        key2=(N,i1,i2,i3,j1,j2,j3-1)
        key3=(N+1,i1,i2,i3,j1,j2,j3)
        key4=(N+1,i1,i2,i3-1,j1,j2,j3)
        key5=(N+1,i1,i2,i3,j1,j2,j3-1)
        keyn=(N,i1,i2,i3+1,j1,j2,j3)
        i_d,j_d=i3,j3
        #print(keyn)
        DPA=ZPA
        DPC=ZPC
    if INFO==1:
        print(D,key0,key1,key2,key3,key4,key5,keyn)
    if None!=integrals1E_all.get(keyn):
        if INFO==1:
            print("by vertical already computed",keyn)
        return
    if min(list(key0))<0 and min(list(key1))<0 and min(list(key2))<0 and min(list(key3))<0 and min(list(key4))<0 and min(list(key5))<0:
        if INFO==1:
            print("by vertical formula is not applicable")
        return
    data0=integrals1E_all.get(key0)
    data1=integrals1E_all.get(key1)
    data2=integrals1E_all.get(key2)
    data3=integrals1E_all.get(key3)
    data4=integrals1E_all.get(key4)
    data5=integrals1E_all.get(key5)
    #print(data0,data1,data2,data3)
    #nx.add_star(IG, [keyn,key0 , key1, key2,key3])
    if min(list(key1))<0:
        data1=0
        integrals1E_all[key1]=0
    if min(list(key2))<0:
        data2=0
        integrals1E_all[key2]=0
    if min(list(key3))<0:
        data3=0
        integrals1E_all[key3]=0
    if min(list(key4))<0:
        data4=0
        integrals1E_all[key4]=0
    if min(list(key5))<0:
        data5=0
        integrals1E_all[key5]=0
    if data0==None:
        #print("data 0 requested1E")
        requested1E_keys[key0]=key0
    if data1==None:
        #print("data 1 requested1E")
        requested1E_keys[key1]=key1
    if data2==None:
        #print("data 2 requested1E")
        requested1E_keys[key2]=key2
    if data3==None:
        #print("data 3 requested1E")
        requested1E_keys[key3]=key3
    if data4==None:
        #print("data 3 requested1E")
        requested1E_keys[key4]=key4
    if data5==None:
        #print("data 3 requested1E")
        requested1E_keys[key5]=key5
    #print(requested1E_keys.keys())
    if data0!=None and data1!=None and data2!=None and data3!=None and data4!=None and data5!=None:
        if (INFO==1):
            print(keyn,"by vertical WRITTEN")
        integrals1E_all[keyn]= DPA*data0+1/2/p*(i_d*data1+j_d*data2)-DPC*data3-1/2/p*(i_d*data4+j_d*data5)
    else:
        if (INFO==1):
            print(keyn,"NOT WRITTEN")
            print(key0,key1,key2,key3,key4,key5)
            print(data0,data1,data2,data3,data4,data5)
            
        else:
            pass
    return
    
#
#  The indices of the integrals which are required, but has not yet been computed.
#
requested1E_keys=dict()
        
#
#  The dictionary of computed integrals
#
integrals1E_all=dict()
for i in range(11):
    integrals1E_all[(i,0,0,0,0,0,0,0)]=TN(i)        

#for a in integrals1E_all.keys():
#    N,i1,i2,i3,j1,j2,j3,k1,k2,k3,l1,l2,l3=a
#    if N==0 and min(list(a))>=0:
#        print(a)
#
#  THERE ARE KEYS requested1E TO COMPUTE INTEGRALS. BECAUSE THEY WERE NOT PREPARED IN ADVANCE.
#



#
#  NEXT WE DO HORIZONTAL RECUSIONS.
#
# HORIZONRAL
# (N; i  j+1 ) <= (N; i j), (N; i+1 j)
#
#  SIMILARLY WE USE THE FUNCTION WHICH MAKES THE HORIZONTAL SHIFT ALONG X/Y/Z DIRECTIONS
#

def HORIZONTAL1E(D,N,i1,i2,i3,j1,j2,j3,INFO=0):
    if D=="X":
        key0=(N,i1+1,i2,i3,j1,j2,j3,)
        key1=(N,i1,i2,i3,j1,j2,j3)
        keyn=(N,i1,i2,i3,j1+1,j2,j3)
        DAB=XAB
        #print(D,key0,key1,keyn)
    if D=="Y":
        key0=(N,i1,i2+1,i3,j1,j2,j3,)
        key1=(N,i1,i2,i3,j1,j2,j3)
        keyn=(N,i1,i2,i3,j1,j2+1,j3)
        DAB=YAB
        #print(D,key0,key1,keyn)
    if D=="Z":
        key0=(N,i1,i2,i3+1,j1,j2,j3,)
        key1=(N,i1,i2,i3,j1,j2,j3)
        keyn=(N,i1,i2,i3,j1,j2,j3+1)
        DAB=ZAB
    data0=integrals1E_all.get(key0)
    data1=integrals1E_all.get(key1)

    if min(list(key0))<0:
        data0=0
        integrals1E_all[key0]=0
    if min(list(key1))<0:
        data1=0
        integrals1E_all[key1]=0
 
    if data0==None:
        requested1E_keys[key0]=key0
    if data1==None:
        requested1E_keys[key1]=key1
    if min(list(key0))<0 and min(list(key1))<0:
        #
        # Maybe in this case, the formula cannot be applied.
        #
        if (INFO==1):
            print(D,key0,key1,"HORIZONTAL1E does nothing for ",keyn)
        return
    if data0!=None and data1!=None:
        #print(data0,data1,data2,data3)
        #print(b,d,DAB,DCD,i1,k1,p,q)
        if (INFO==1):
            print(D,key0,key1,"HORIZONTAL 1E compute",keyn)
        integrals1E_all[keyn]= data0 + DAB*data1
    else:
        if (INFO==1):
            print(keyn,"NOT WRITTEN")
            print(key0,key1,keyn)
            print(data0,data1)


        


# In[ ]:


requested1E_keys=dict()
integrals1E_all=dict()
NMAX=11
for i in range(NMAX):
    integrals1E_all[(i,0,0,0,0,0,0)]=TN(i)    

def late_evaluate1E(ckeys,INFO=0):
#
# THIS FUNCTION COMPUTES THE TWO-ELECTRON INTEGRAL AT CKEYS THROUGH RECUSION,
#   USING THE DATA WHICH HAVE ALREADY BEEN COMPUTED,
#   AND WRITING THE RESULTS WHICH HAVE JUST BEEN COMPUTED
#
    N,i1,i2,i3,j1,j2,j3=ckeys
    if integrals1E_all.get(ckeys)!=None:
        print("FOUND",ckeys)
        return 1
    # VERTICAL:
    
    VERTICAL1E("X",N,i1-1,i2,i3,j1,j2,j3,INFO)
    VERTICAL1E("Y",N,i1,i2-1,i3,j1,j2,j3,INFO)
    VERTICAL1E("Z",N,i1,i2,i3-1,j1,j2,j3,INFO)
    HORIZONTAL1E("X",N,i1,i2,i3,j1-1,j2,j3,INFO)
    HORIZONTAL1E("Y",N,i1,i2,i3,j1,j2-1,j3,INFO)
    HORIZONTAL1E("Z",N,i1,i2,i3,j1,j2,j3-1,INFO)
    return 0
 
ckeys=(0,1,0,0,0,0,0)
print(ckeys)
late_evaluate1E(ckeys)    
#requested1E_keys.keys()

CurrentN=list(ckeys)[0]

import copy
IsFound=None
while(IsFound==None and CurrentN<NMAX):
    requested1E_keys_copy=copy.deepcopy(requested1E_keys)
    for ckey in requested1E_keys_copy.keys():
        if min(list(ckey))>=0:
            print("requested1E",ckey)
            F=late_evaluate1E(ckey)
            if F==1:
                print("\n\nCOMPUTED",ckey,integrals1E_all.get(ckey))
                requested1E_keys.pop(ckey)

    late_evaluate1E(ckeys)
    IsFound=integrals1E_all.get(ckeys)
    if IsFound!=None:
        print(ckeys, "is computed")
print(IsFound)        

ckeys=(0,2,0,0,0,0,0)   
print("\n",ckeys)

late_evaluate1E(ckeys)    
#requested1E_keys.keys()

CurrentN=list(ckeys)[0]

import copy
IsFound=None
while(IsFound==None and CurrentN<NMAX):
    requested1E_keys_copy=copy.deepcopy(requested1E_keys)
    for ckey in requested1E_keys_copy.keys():
        if min(list(ckey))>=0:
            print("requested1E",ckey)
            F=late_evaluate1E(ckey)
            if F==1:
                print("\n\nCOMPUTED",ckey,integrals1E_all.get(ckey))
                requested1E_keys.pop(ckey)

    late_evaluate1E(ckeys)
    IsFound=integrals1E_all.get(ckeys)
    if IsFound!=None:
        print(ckeys, "is computed")
print(IsFound)        


# In[ ]:


#requested1E_keys=dict()
#integrals1E_all=dict()
NMAX=11
for i in range(NMAX):
    integrals1E_all[(i,0,0,0,0,0,0)]=TN(i)
NMAX=11
ckeys=(0,0,0,0,0,0,1)   
print("\n",ckeys)
late_evaluate1E(ckeys,1)    
#requested1E_keys.keys()

CurrentN=list(ckeys)[0]

loopcount=0
import copy
IsFound=None
while(IsFound==None and CurrentN<NMAX and loopcount<=5):
    print("\n")
    loopcount+=1
    requested1E_keys_copy=copy.deepcopy(requested1E_keys)
    print(requested1E_keys.keys())
    for ckey in requested1E_keys_copy.keys():
        if min(list(ckey))>=0:
            #print("requested1E",ckey)
            F=late_evaluate1E(ckey,1)
            if F==1:
                #print("\n\nCOMPUTED",ckey,integrals1E_all.get(ckey))
                requested1E_keys.pop(ckey)

    late_evaluate1E(ckeys,1)
    IsFound=integrals1E_all.get(ckeys)
    if IsFound!=None:
        print(ckeys, "is computed")
print(IsFound)        


# In[ ]:


NMAX=11
for i in range(NMAX):
    integrals1E_all[(i,0,0,0,0,0,0)]=TN(i)
NMAX=11
ckeys=(0,1,0,0,0,0,1)   
print("\n",ckeys)
late_evaluate1E(ckeys,1)    
#requested1E_keys.keys()

CurrentN=list(ckeys)[0]

loopcount=0
import copy
IsFound=None
while(IsFound==None and CurrentN<NMAX and loopcount<=5):
    print("\n")
    loopcount+=1
    requested1E_keys_copy=copy.deepcopy(requested1E_keys)
    print(requested1E_keys.keys())
    for ckey in requested1E_keys_copy.keys():
        if min(list(ckey))>=0:
            #print("requested1E",ckey)
            F=late_evaluate1E(ckey,1)
            if F==1:
                #print("\n\nCOMPUTED",ckey,integrals1E_all.get(ckey))
                requested1E_keys.pop(ckey)

    late_evaluate1E(ckeys,1)
    IsFound=integrals1E_all.get(ckeys)
    if IsFound!=None:
        print(ckeys, "is computed")
print(IsFound)   


# In[ ]:


#
# We have computed integrals at various keys by the recustion.
#
ckeys=(0,0,0,0,0,0,0)
v00=integrals1E_all.get(ckeys)
ckeys=(0,1,0,0,0,0,0)
vx0=integrals1E_all.get(ckeys)
ckeys=(0,0,0,0,0,0,1)
v0z=integrals1E_all.get(ckeys)
ckeys=(0,0,0,1,0,0,0)
vz0=integrals1E_all.get(ckeys)
ckeys=(0,1,0,0,0,0,1)  
vxz=integrals1E_all.get(ckeys)
print([v00,vx0,v0z,vz0,vxz])


# In[ ]:


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

[X_A,Y_A,Z_A]=[0.67,1.4,0.54]
[X_B,Y_B,Z_B]=[0.98,1.5,0.73]
[X_C,Y_C,Z_C]=[-.930,0,1.93]
[X_D,Y_D,Z_D]=[1.205,1.02,1.504]


[X_AB,Y_AB,Z_AB]=[X_A-X_B,Y_A-Y_B,Z_A-Z_B]
[X_CD,Y_CD,Z_CD]=[X_C-X_D,Y_C-Y_D,Z_C-Z_D]

aa=1.05
bb=1.54
cc=2.3
dd=1.2

X_P=(aa*X_A+bb*X_B)/(aa+bb)
Y_P=(aa*Y_A+bb*Y_B)/(aa+bb)
Z_P=(aa*Z_A+bb*Z_B)/(aa+bb)

pp=aa+bb
qq=cc+dd

X_Q=(cc*X_C+dd*X_D)/(cc+dd)
Y_Q=(cc*Y_C+dd*Y_D)/(cc+dd)
Z_Q=(cc*Z_C+dd*Z_D)/(cc+dd)


[X_PQ,Y_PQ,Z_PQ]=[X_P-X_Q,Y_P-Y_Q,Z_P-Z_Q]
PQ2=X_PQ**2+Y_PQ**2+Z_PQ**2

[X_PA,Y_PA,Z_PA]=[X_P-X_A,Y_P-Y_A,Z_P-Z_A]
[X_PC,Y_PC,Z_PC]=[X_P-X_C,Y_P-Y_C,Z_P-Z_C]
PC2=X_PC**2+Y_PC**2+Z_PC**2


V_T_0_00=2*sympy.pi/pp*KXYZAB(aa,bb,[X_A,Y_A,Z_A],[X_B,Y_B,Z_B])*BOYS(0,pp*PC2)
V_T_1_00=2*sympy.pi/pp*KXYZAB(aa,bb,[X_A,Y_A,Z_A],[X_B,Y_B,Z_B])*BOYS(1,pp*PC2)
V_T_2_00=2*sympy.pi/pp*KXYZAB(aa,bb,[X_A,Y_A,Z_A],[X_B,Y_B,Z_B])*BOYS(2,pp*PC2)
V_T_3_00=2*sympy.pi/pp*KXYZAB(aa,bb,[X_A,Y_A,Z_A],[X_B,Y_B,Z_B])*BOYS(3,pp*PC2)
V_T_4_00=2*sympy.pi/pp*KXYZAB(aa,bb,[X_A,Y_A,Z_A],[X_B,Y_B,Z_B])*BOYS(4,pp*PC2)
fABC=V000000(z1,z2,[AX,AY,AZ],[BX,BY,BZ],[CX,CY,CZ])


# In[ ]:


print([sympy.N(fABC.subs([(z1,aa),(z2,bb),(AX,X_A),(AY,Y_A),(AZ,Z_A),(BX,X_B),(BY,Y_B),(BZ,Z_B),(CX,X_C),(CY,Y_C),(CZ,Z_C)])),
 sympy.N(v00.subs([(p,pp),(a,aa),(b,bb),\
                    (XPQ,X_PQ),(XAB,X_AB),(XCD,X_CD),\
                    (YPQ,Y_PQ),(YAB,Y_AB),(YCD,Y_CD),\
                    (ZPQ,Z_PQ),(ZAB,Z_AB),(ZCD,Z_CD),\
                    (XPA,X_PA),(YPA,Y_PA),(ZPA,Z_PA),\
                    (YAB,Y_AB),(YAB,Y_AB),(ZAB,Z_AB),\
                    (TNS[0],V_T_0_00),(TNS[1],V_T_1_00),(TNS[2],V_T_2_00),(TNS[3],V_T_3_00),(TNS[4],V_T_4_00)])).expand()])


# In[ ]:


fABC_X0=fABC.diff(AX)/2/z1
print([sympy.N(fABC_X0.subs([(z1,aa),(z2,bb),(AX,X_A),(AY,Y_A),(AZ,Z_A),(BX,X_B),(BY,Y_B),(BZ,Z_B),(CX,X_C),(CY,Y_C),(CZ,Z_C)])),
sympy.N(vx0.subs([(p,pp),(a,aa),(b,bb),\
                    (XPQ,X_PQ),(XAB,X_AB),(XCD,X_CD),\
                    (YPQ,Y_PQ),(YAB,Y_AB),(YCD,Y_CD),\
                    (ZPQ,Z_PQ),(ZAB,Z_AB),(ZCD,Z_CD),\
                    (XPA,X_PA),(YPA,Y_PA),(ZPA,Z_PA),\
                    (YAB,Y_AB),(YAB,Y_AB),(ZAB,Z_AB),\
                    (XPC,X_PC),(YPC,Y_PC),(ZPC,Z_PC),\
                    (TNS[0],V_T_0_00),(TNS[1],V_T_1_00),(TNS[2],V_T_2_00),(TNS[3],V_T_3_00),(TNS[4],V_T_4_00)])).expand()])


# In[ ]:


fABC_Z0=fABC.diff(AZ)/2/z1
print([sympy.N(fABC_Z0.subs([(z1,aa),(z2,bb),(AX,X_A),(AY,Y_A),(AZ,Z_A),(BX,X_B),(BY,Y_B),(BZ,Z_B),(CX,X_C),(CY,Y_C),(CZ,Z_C)])),
sympy.N(vz0.subs([(p,pp),(a,aa),(b,bb),\
                    (XPQ,X_PQ),(XAB,X_AB),(XCD,X_CD),\
                    (YPQ,Y_PQ),(YAB,Y_AB),(YCD,Y_CD),\
                    (ZPQ,Z_PQ),(ZAB,Z_AB),(ZCD,Z_CD),\
                    (XPA,X_PA),(YPA,Y_PA),(ZPA,Z_PA),\
                    (YAB,Y_AB),(YAB,Y_AB),(ZAB,Z_AB),\
                    (XPC,X_PC),(YPC,Y_PC),(ZPC,Z_PC),\
                    (TNS[0],V_T_0_00),(TNS[1],V_T_1_00),(TNS[2],V_T_2_00),(TNS[3],V_T_3_00),(TNS[4],V_T_4_00)])).expand()])


# In[ ]:


fABC_0Z=fABC.diff(BZ)/2/z2
print([sympy.N(fABC_0Z.subs([(z1,aa),(z2,bb),(AX,X_A),(AY,Y_A),(AZ,Z_A),(BX,X_B),(BY,Y_B),(BZ,Z_B),(CX,X_C),(CY,Y_C),(CZ,Z_C)])),
sympy.N(v0z.subs([(p,pp),(a,aa),(b,bb),\
                    (XPQ,X_PQ),(XAB,X_AB),(XCD,X_CD),\
                    (YPQ,Y_PQ),(YAB,Y_AB),(YCD,Y_CD),\
                    (ZPQ,Z_PQ),(ZAB,Z_AB),(ZCD,Z_CD),\
                    (XPA,X_PA),(YPA,Y_PA),(ZPA,Z_PA),\
                    (YAB,Y_AB),(YAB,Y_AB),(ZAB,Z_AB),\
                    (XPC,X_PC),(YPC,Y_PC),(ZPC,Z_PC),\
                    (TNS[0],V_T_0_00),(TNS[1],V_T_1_00),(TNS[2],V_T_2_00),(TNS[3],V_T_3_00),(TNS[4],V_T_4_00)])).expand()])


# In[ ]:


fABC_XZ=fABC.diff(AX).diff(BZ)/2/z1/2/z2
print([sympy.N(fABC_XZ.subs([(z1,aa),(z2,bb),(AX,X_A),(AY,Y_A),(AZ,Z_A),(BX,X_B),(BY,Y_B),(BZ,Z_B),(CX,X_C),(CY,Y_C),(CZ,Z_C)])),
sympy.N(vxz.subs([(p,pp),(a,aa),(b,bb),\
                    (XPQ,X_PQ),(XAB,X_AB),(XCD,X_CD),\
                    (YPQ,Y_PQ),(YAB,Y_AB),(YCD,Y_CD),\
                    (ZPQ,Z_PQ),(ZAB,Z_AB),(ZCD,Z_CD),\
                    (XPA,X_PA),(YPA,Y_PA),(ZPA,Z_PA),\
                    (YAB,Y_AB),(YAB,Y_AB),(ZAB,Z_AB),\
                    (XPC,X_PC),(YPC,Y_PC),(ZPC,Z_PC),\
                    (TNS[0],V_T_0_00),(TNS[1],V_T_1_00),(TNS[2],V_T_2_00),(TNS[3],V_T_3_00),(TNS[4],V_T_4_00)])).expand()])


# # Obara-Saika Recusion for overlap and kinetic integrals
# 
# The intgrals are decomposed into the products of the contributions of one-dimentional integrals:
# 
# $$
# S_{ab}=\int d^3r G_a(r-A) G_b(r-B)=\int dx G_j^{x}(x-A_x)G_j^{x}(x-B_x) \times \int dy G_k^{y}(y-A_y)G_l^{y}(y-B_y)\int dz G_a^{m}(z-A_z)G_n^{z}(z-B_z)
# $$
# with $a=(i,k,m)$, $b=(j,l,n)$
# 
# 
# For this reason, we have only to build the recursion relation of the one-dimentional integrals:
# 
# 
# $$
# S^x_{i+1,j}=X_{PA}S_{i,j}+\frac{1}{2p}(iS^x_{i-1,j}+jS^x{i,j-1})
# $$
# 
# $$
# S^x_{i,j+1}=X_{PB}S^x_{i,j}+\frac{1}{2p}(iS^x_{i-1,j}+jS^x{i,j-1})
# $$
# $$
# S_{00}=K_{ab}^x\sqrt{\frac{\pi}{p}}
# $$
# 
# For the kinetic integrals, note that they are given by
# $$
# T_{a,b}=\int d^3r G_a(r)\left( -\frac{1}{2}\nabla^2 \right) G_b(r)
# $$
# $$
# =\int d^3r G_a(r)\left( -\frac{1}{2}\nabla^2 \right) G_b(r)
# $$
# $$
# =T^x_{ij}S^y_{kl}S^z_{mn}+S^x_{ij}T^y_{kl}S^z_{mn}+S^x_{ij}S^y_{kl}T^z_{mn}
# $$
# 
# $$
# T^x_{i+1,j}=X_{PA}T^x_{i,j}+\frac{1}{2(a+b)}(iT^x_{i-1,j}+jT^x_{i,j-1})+\frac{b}{a+b}(2aS^x_{i+1,j}-iS^x_{i-1,j})\}
# $$
# 
# $$
# T^x_{i,j+1}=X^x_{PB}T_{i,j}+\frac{1}{2(a+b)}(iT^x_{i-1,j}+jT^x_{i,j-1})+\frac{a}{a+b}(2bS^x_{i,j+1}-jS^x_{i,j-1})\}
# $$
# $$
# T^x_{00}=\left[a-2a^2\left(X_{PA}^2+\frac{1}{2p}\right)\right]S^x_{00}
# $$
# 
# In the following, we compute the analytic formulas for $S^{x}_{ij}$ and $T^{x}_{ij}$; wherby we can get the integrals along other direction through the symbolic replacement $X$ with $Y$ or $Z$.

# In[ ]:


def VERTICALSA(i,j,INFO=0):

    key0=(i,j)
    key1=(i-1,j)
    key2=(i,j-1)
    keyn=(i+1,j)


    data0=integralsS_all.get(key0)
    data1=integralsS_all.get(key1)
    data2=integralsS_all.get(key2)

    if min(list(key0))<0:
        data0=0
        integralsS_all[key0]=0
    if min(list(key1))<0:
        data1=0
        integralsS_all[key1]=0
    if min(list(key2))<0:
        data2=0
        integralsS_all[key2]=0
    if data0==None:
        requestedS_keys[key0]=key0
    if data1==None:
        requestedS_keys[key1]=key1
    if data2==None:
        requestedS_keys[key2]=key2
    if min(list(key0))<0 and min(list(key1)) and min(list(key2))<0:
        #
        # Maybe in this case, the formula cannot be applied.
        #
        if (INFO==1):
            print(XPA,key0,key1,"VERTICAL1SA does nothing for ",keyn)
        return
    if data0!=None and data1!=None:
        #print(data0,data1,data2,data3)
        #print(b,d,DAB,DCD,i1,k1,p,q)
        if (INFO==1):
            print(key0,key1,"VERTICAL1SA computes",keyn)
        integralsS_all[keyn]= XPA*data0 + 1/2/p*(i*data1+j*data2)
    else:
        if (INFO==1):
            print(keyn,"NOT WRITTEN")
            print(key0,key1,keyn)
            print(data0,data1)
            
def VERTICALSB(i,j,INFO=0):

    key0=(i,j)
    key1=(i-1,j)
    key2=(i,j-1)
    keyn=(i,j+1)


    data0=integralsS_all.get(key0)
    data1=integralsS_all.get(key1)
    data2=integralsS_all.get(key2)

    if min(list(key0))<0:
        data0=0
        integralsS_all[key0]=0
    if min(list(key1))<0:
        data1=0
        integralsS_all[key1]=0
    if min(list(key2))<0:
        data2=0
        integralsS_all[key2]=0
    if data0==None:
        requestedS_keys[key0]=key0
    if data1==None:
        requestedS_keys[key1]=key1
    if data2==None:
        requestedS_keys[key2]=key2
    if min(list(key0))<0 and min(list(key1)) and min(list(key2))<0:
        #
        # Maybe in this case, the formula cannot be applied.
        #
        if (INFO==1):
            print(XPB,key0,key1,"VERTICAL1SB does nothing for ",keyn)
        return
    if data0!=None and data1!=None:
        #print(data0,data1,data2,data3)
        #print(b,d,DAB,DCD,i1,k1,p,q)
        if (INFO==1):
            print(key0,key1,"VERTICAL1SB computes",keyn)
        integralsS_all[keyn]= XPB*data0 + 1/2/p*(i*data1+j*data2)
    else:
        if (INFO==1):
            print(keyn,"NOT WRITTEN")
            print(key0,key1,keyn)
            print(data0,data1)            

def KXAB(a,b,A,B):
    mu=a*b/(a+b)
    RAB2=(A-B)**2
    return sympy.exp(-mu*RAB2)

def SD00(a,b,AX,BX):
#
#   OVERLAP INTEGRAL between exp(-A*|r-RA|^2) and exp(-B*|r-RB|^2)
#
    RAB2=(AX-BX)**2
    PI=sympy.pi

    return (PI/(a+b))**(1/2)*sympy.exp(-a*b*RAB2/(a+b))

requestedS_keys=dict()
integralsS_all=dict()

TNSTRING=''
for i in range(11):
    TNSTRING+= 'S'+str(i)+' '
NS=symbols(TNSTRING)
def SN(N):
    return NS[N]

TNSTRING=''
for i in range(11):
    TNSTRING+= 'K'+str(i)+' '
NK=symbols(TNSTRING)
def KN(N):
    return NK[N]

integralsS_all[(0,0)]=SN(0)   

XPA, YPA, ZPA=symbols("XPA YPA ZPA")
XPB, YPB, ZPB=symbols("XPB YPB ZPB")
def late_evaluateS(ckeys,INFO=0):
#
# THIS FUNCTION COMPUTES THE TWO-ELECTRON INTEGRAL AT CKEYS THROUGH RECUSION,
#   USING THE DATA WHICH HAVE ALREADY BEEN COMPUTED,
#   AND WRITING THE RESULTS WHICH HAVE JUST BEEN COMPUTED
#
    print("ckeys",ckeys)
    i,j=ckeys
    if integralsS_all.get(ckeys)!=None:
        print("FOUND",ckeys)
        return 1
    # VERTICAL:
    
    VERTICALSA(i-1,j,INFO)
    VERTICALSB(i,j-1,INFO)

    return 0
 
ckeys=(1,0)
print(ckeys)
late_evaluateS(ckeys,1)    
#requested1E_keys.keys()

#CurrentN=list(ckeys)[0]
iloop=0
import copy
IsFound=None
while(IsFound==None and iloop<=5):
    iloop+=1
    requestedS_keys_copy=copy.deepcopy(requestedS_keys)
    for ckey in requestedS_keys_copy.keys():
        if min(list(ckey))>=0:
            print("requested S",ckey)
            F=late_evaluateS(ckey,1)
            if F==1:
                print("\n\nCOMPUTED",ckey,integralsS_all.get(ckey))
                requestedS_keys.pop(ckey)

    late_evaluateS(ckeys)
    IsFound=integralsS_all.get(ckeys)
    if IsFound!=None:
        print(ckeys, "is computed")
print(IsFound)        
ckeys=(0,1)
print(ckeys)
late_evaluateS(ckeys,1)    
#requested1E_keys.keys()

#CurrentN=list(ckeys)[0]
iloop=0
import copy
IsFound=None
while(IsFound==None):
    iloop+=1
    requestedS_keys_copy=copy.deepcopy(requestedS_keys)
    for ckey in requestedS_keys_copy.keys():
        if min(list(ckey))>=0:
            print("requested S",ckey)
            F=late_evaluateS(ckey,1)
            if F==1:
                print("\n\nCOMPUTED",ckey,integralsS_all.get(ckey))
                requestedS_keys.pop(ckey)

    late_evaluateS(ckeys)
    IsFound=integralsS_all.get(ckeys)
    if IsFound!=None:
        print(ckeys, "is computed")
print(IsFound)   


ckeys=(1,1)
print(ckeys)
late_evaluateS(ckeys,1)    
#requested1E_keys.keys()

#CurrentN=list(ckeys)[0]
iloop=0
import copy
IsFound=None
while(IsFound==None):
    iloop+=1
    requestedS_keys_copy=copy.deepcopy(requestedS_keys)
    for ckey in requestedS_keys_copy.keys():
        if min(list(ckey))>=0:
            print("requested S",ckey)
            F=late_evaluateS(ckey,1)
            if F==1:
                print("\n\nCOMPUTED",ckey,integralsS_all.get(ckey))
                requestedS_keys.pop(ckey)

    late_evaluateS(ckeys)
    IsFound=integralsS_all.get(ckeys)
    if IsFound!=None:
        print(ckeys, "is computed")
print(IsFound)   

ckeys=(0,2)
print(ckeys)
late_evaluateS(ckeys,1)    
#requested1E_keys.keys()

#CurrentN=list(ckeys)[0]
iloop=0
import copy
IsFound=None
while(IsFound==None):
    iloop+=1
    requestedS_keys_copy=copy.deepcopy(requestedS_keys)
    for ckey in requestedS_keys_copy.keys():
        if min(list(ckey))>=0:
            print("requested S",ckey)
            F=late_evaluateS(ckey,1)
            if F==1:
                print("\n\nCOMPUTED",ckey,integralsS_all.get(ckey))
                requestedS_keys.pop(ckey)

    late_evaluateS(ckeys)
    IsFound=integralsS_all.get(ckeys)
    if IsFound!=None:
        print(ckeys, "is computed")
print(IsFound)   

ckeys=(2,0)
print(ckeys)
late_evaluateS(ckeys,1)    
#requested1E_keys.keys()

#CurrentN=list(ckeys)[0]
iloop=0
import copy
IsFound=None
while(IsFound==None):
    iloop+=1
    requestedS_keys_copy=copy.deepcopy(requestedS_keys)
    for ckey in requestedS_keys_copy.keys():
        if min(list(ckey))>=0:
            print("requested S",ckey)
            F=late_evaluateS(ckey,1)
            if F==1:
                print("\n\nCOMPUTED",ckey,integralsS_all.get(ckey))
                requestedS_keys.pop(ckey)

    late_evaluateS(ckeys)
    IsFound=integralsS_all.get(ckeys)
    if IsFound!=None:
        print(ckeys, "is computed")
print(IsFound)   


# In[ ]:


ckeys=(0,0)
sd00=integralsS_all.get(ckeys)
ckeys=(1,0)
sd10=integralsS_all.get(ckeys)
ckeys=(0,1)
sd01=integralsS_all.get(ckeys)
ckeys=(1,1)
sd11=integralsS_all.get(ckeys)
ckeys=(0,2)
sd02=integralsS_all.get(ckeys)
ckeys=(2,0)
sd20=integralsS_all.get(ckeys)
[sd10,sd01,sd11,sd02,sd20]


# In[ ]:


print("Comparison of Oberlap S[i,j] between Obara-Saika and Symbolic differentiation.\nVaridation by numeric substitution.")
SAB=S000000(z1,z2,[AX,AY,AZ],[BX,BY,BZ])
SXAB=SD00(z1,z2,AX,BX)
S_X_00=(sympy.pi/(z1+z2))**(1/2)*KXAB(z1,z2,AX,BX)
s00=sd00.subs(SN(0),S_X_00)
print(s00.subs(z1,aa).subs(z2,bb).subs(AX,X_A).subs(BX,X_B))
print(SXAB.subs(z1,aa).subs(z2,bb).subs(AX,X_A).subs(BX,X_B))


w1=sd01.subs(SN(0),S_X_00).subs(XPB,X_P-X_B).subs(z1,aa).subs(z2,bb).subs(AX,X_A).subs(BX,X_B)
expr=SXAB.diff(BX)/2/z2
w2=expr.subs(z1,aa).subs(z2,bb).subs(AX,X_A).subs(BX,X_B)
print([w1,w2])
w1=sd10.subs(SN(0),S_X_00).subs(XPA,X_P-X_A).subs(z1,aa).subs(z2,bb).subs(AX,X_A).subs(BX,X_B)
expr=SXAB.diff(AX)/2/z1
w2=expr.subs(z1,aa).subs(z2,bb).subs(AX,X_A).subs(BX,X_B)
print([w1,w2])
w1=(sd11.subs(SN(0),S_X_00).subs(XPA,X_P-X_A).subs(XPB,X_P-X_B).subs(z1,aa).subs(z2,bb).subs(AX,X_A).subs(BX,X_B).subs(p,aa+bb))
expr=SXAB.diff(AX).diff(BX)/2/z1/2/z2
w2=(expr.subs(z1,aa).subs(z2,bb).subs(AX,X_A).subs(BX,X_B))
print([w1,w2])

w1=(sd20.subs(SN(0),S_X_00).subs(XPA,X_P-X_A).subs(z1,aa).subs(z2,bb).subs(AX,X_A).subs(BX,X_B).subs(p,aa+bb))
expr=SXAB.diff(AX).diff(AX)/2/z1/2/z1+SXAB*0.5/z1
w2=(expr.subs(z1,aa).subs(z2,bb).subs(AX,X_A).subs(BX,X_B))
print([w1,w2])
w1=(sd02.subs(SN(0),S_X_00).subs(XPB,X_P-X_B).subs(z1,aa).subs(z2,bb).subs(AX,X_A).subs(BX,X_B).subs(p,aa+bb))
expr=SXAB.diff(BX).diff(BX)/2/z2/2/z2+SXAB*0.5/z2
w2=(expr.subs(z1,aa).subs(z2,bb).subs(AX,X_A).subs(BX,X_B))
print([w1,w2])


# In[ ]:


def VERTICALTA(i,j,INFO=0):
    key0=(i,j)
    key1=(i-1,j)
    key2=(i,j-1)
    key3=(i+1,j)
    key4=(i-1,j)
    keyn=(i+1,j)
    data0=integralsT_all.get(key0)
    data1=integralsT_all.get(key1)
    data2=integralsT_all.get(key2)
    data3=integralsS_all.get(key3)
    data4=integralsS_all.get(key4)
    if min(list(key0))<0:
        data0=0
        integralsT_all[key0]=0
    if min(list(key1))<0:
        data1=0
        integralsT_all[key1]=0
    if min(list(key2))<0:
        data2=0
        integralsT_all[key2]=0
    if min(list(key3))<0:
        data3=0
    if min(list(key4))<0:
        data4=0
    if data0==None:
        requestedT_keys[key0]=key0
    if data1==None:
        requestedT_keys[key1]=key1
    if data2==None:
        requestedT_keys[key2]=key2
    if data3==None:
        print("S(",key3,") is not computed.")
        return
    if data4==None:
        print("S(",key4,") is not computed.")
        return

    if min(list(key0))<0 and min(list(key1)) and min(list(key2)) <0:
        #
        # Maybe in this case, the formula cannot be applied.
        #
        if (INFO==1):
            print(key0,key1,"VERTICAL1TA does nothing for ",keyn)
        return
    if data0!=None and data1!=None:
        #print(data0,data1,data2,data3)
        #print(b,d,DAB,DCD,i1,k1,p,q)
        if (INFO==1):
            print(key0,key1,"VERTICAL1TA compute",keyn)
        integralsT_all[keyn]= XPA*data0 + 1/2/p*(i*data1+j*data2)+b/p*(2*a*data3-i*data4)
    else:
        if (INFO==1):
            print(keyn,"NOT WRITTEN")
            print(key0,key1,keyn)
            print(data0,data1)
            
def VERTICALTB(i,j,INFO=0):

    key0=(i,j)
    key1=(i-1,j)
    key2=(i,j-1)
    keyn=(i,j+1)
    key3=(i,j+1)
    key4=(i,j-1)

 
    data0=integralsT_all.get(key0)
    data1=integralsT_all.get(key1)
    data2=integralsT_all.get(key2)
    data3=integralsS_all.get(key3)
    data4=integralsS_all.get(key4)
    if min(list(key0))<0:
        data0=0
        integralsT_all[key0]=0
    if min(list(key1))<0:
        data1=0
        integralsT_all[key1]=0
    if min(list(key2))<0:
        data2=0
        integralsT_all[key2]=0
    if min(list(key3))<0:
        data3=0
    if min(list(key4))<0:
        data4=0
    if data0==None:
        requestedT_keys[key0]=key0
    if data1==None:
        requestedT_keys[key1]=key1
    if data2==None:
        requestedT_keys[key2]=key2
    if data3==None:
        print("S(",key3,") is not computed.")
        return
    if data4==None:
        print("S(",key4,") is not computed.")
        return
    if min(list(key0))<0 and min(list(key1)) and min(list(key2))<0:
        #
        # Maybe in this case, the formula cannot be applied.
        #
        if (INFO==1):
            print(key0,key1,"VERTICAL1TB does nothing for ",keyn)
        return
    if data0!=None and data1!=None:
        #print(data0,data1,data2,data3)
        #print(b,d,DAB,DCD,i1,k1,p,q)
        if (INFO==1):
            print(key0,key1,"VERTICAL1TB compute",keyn)
        integralsT_all[keyn]= XPB*data0 + 1/2/p*(i*data1+j*data2)+a/p*(2*b*data3-j*data4)
    else:
        if (INFO==1):
            print(keyn,"NOT WRITTEN")
            print(key0,key1,keyn)
            print(data0,data1)

requestedT_keys=dict()
integralsT_all=dict()

integralsT_all[(0,0)]=KN(0)   
XPA, YPA, ZPA=symbols("XPA YPA ZPA")
XPB, YPB, ZPB=symbols("XPB YPB ZPB")
def late_evaluateT(ckeys,INFO=0):
#
# THIT FUNCTION COMPUTET THE TWO-ELECTRON INTEGRAL AT CKEYT THROUGH RECUTION,
#   UTING THE DATA WHICH HAVE ALREADY BEEN COMPUTED,
#   AND WRITING THE RETULTT WHICH HAVE JUTT BEEN COMPUTED
#
    print("ckeys",ckeys)
    i,j=ckeys
    if integralsT_all.get(ckeys)!=None:
        print("FOUND",ckeys)
        return 1
    # VERTICAL:
    
    VERTICALTA(i-1,j,INFO)
    VERTICALTB(i,j-1,INFO)

    return 0
 
ckeys=(1,0)
print(ckeys)
late_evaluateT(ckeys,1)    
CurrentN=list(ckeys)[0]
iloop=0
import copy
IsFound=None
while(IsFound==None and CurrentN<NMAX and iloop<=5):
    iloop+=1
    requestedT_keys_copy=copy.deepcopy(requestedT_keys)
    for ckey in requestedT_keys_copy.keys():
        if min(list(ckey))>=0:
            print("requested T",ckey)
            F=late_evaluateT(ckey,1)
            if F==1:
                print("\n\nCOMPUTED",ckey,integralsT_all.get(ckey))
                requestedT_keys.pop(ckey)

    late_evaluateT(ckeys)
    IsFound=integralsT_all.get(ckeys)
    if IsFound!=None:
        print(ckeys, "is computed")
print(IsFound)        


ckeys=(0,1)
print(ckeys)
late_evaluateT(ckeys,1)    
CurrentN=list(ckeys)[0]
iloop=0
import copy
IsFound=None
while(IsFound==None and CurrentN<NMAX and iloop<=5):
    iloop+=1
    requestedT_keys_copy=copy.deepcopy(requestedT_keys)
    for ckey in requestedT_keys_copy.keys():
        if min(list(ckey))>=0:
            print("requested T",ckey)
            F=late_evaluateT(ckey,1)
            if F==1:
                print("\n\nCOMPUTED",ckey,integralsT_all.get(ckey))
                requestedT_keys.pop(ckey)

    late_evaluateT(ckeys)
    IsFound=integralsT_all.get(ckeys)
    if IsFound!=None:
        print(ckeys, "is computed")
print(IsFound)   


ckeys=(1,1)
print(ckeys)
late_evaluateT(ckeys,1)    
CurrentN=list(ckeys)[0]
iloop=0
import copy
IsFound=None
while(IsFound==None and CurrentN<NMAX and iloop<=5):
    iloop+=1
    requestedT_keys_copy=copy.deepcopy(requestedT_keys)
    for ckey in requestedT_keys_copy.keys():
        if min(list(ckey))>=0:
            print("requested T",ckey)
            F=late_evaluateT(ckey,1)
            if F==1:
                print("\n\nCOMPUTED",ckey,integralsT_all.get(ckey))
                requestedT_keys.pop(ckey)

    late_evaluateT(ckeys)
    IsFound=integralsT_all.get(ckeys)
    if IsFound!=None:
        print(ckeys, "is computed")
print(IsFound)   

ckeys=(2,0)
print(ckeys)
late_evaluateT(ckeys,1)    
CurrentN=list(ckeys)[0]
iloop=0
import copy
IsFound=None
while(IsFound==None and CurrentN<NMAX and iloop<=5):
    iloop+=1
    requestedT_keys_copy=copy.deepcopy(requestedT_keys)
    for ckey in requestedT_keys_copy.keys():
        if min(list(ckey))>=0:
            print("requested T",ckey)
            F=late_evaluateT(ckey,1)
            if F==1:
                print("\n\nCOMPUTED",ckey,integralsT_all.get(ckey))
                requestedT_keys.pop(ckey)

    late_evaluateT(ckeys)
    IsFound=integralsT_all.get(ckeys)
    if IsFound!=None:
        print(ckeys, "is computed")
print(IsFound)  

ckeys=(0,2)
print(ckeys)
late_evaluateT(ckeys,1)    
CurrentN=list(ckeys)[0]
iloop=0
import copy
IsFound=None
while(IsFound==None and CurrentN<NMAX and iloop<=5):
    iloop+=1
    requestedT_keys_copy=copy.deepcopy(requestedT_keys)
    for ckey in requestedT_keys_copy.keys():
        if min(list(ckey))>=0:
            print("requested T",ckey)
            F=late_evaluateT(ckey,1)
            if F==1:
                print("\n\nCOMPUTED",ckey,integralsT_all.get(ckey))
                requestedT_keys.pop(ckey)

    late_evaluateT(ckeys)
    IsFound=integralsT_all.get(ckeys)
    if IsFound!=None:
        print(ckeys, "is computed")
print(IsFound)  


# In[ ]:


ckeys=(1,0)
tx10=integralsT_all.get(ckeys)
ckeys=(0,1)
tx01=integralsT_all.get(ckeys)

ckeys=(1,1)
tx11=integralsT_all.get(ckeys)
ckeys=(0,2)
tx02=integralsT_all.get(ckeys)
ckeys=(2,0)
tx20=integralsT_all.get(ckeys)


# In[ ]:


print([sd10,sd01])
print([tx01,tx10])


# In[ ]:


print("Comparison of Kinetic T[i,j] between Obara-Saika and Symbolic differentiation.\nVaridation by numeric substitution.")

ckeys=(1,0)
tx10=integralsT_all.get(ckeys)


SXAB00=SD00(a,b,AX,BX)
X_PA=((a*AX+b*BX)-(a+b)*AX)/(a+b)
X_PA.expand()
X_PB=((a*AX+b*BX)-(a+b)*BX)/(a+b)
X_PB.expand()
TXAB00=(a-2*a*a*(X_PA**2+1/2/p))*SXAB00

w1=sympy.N(tx10.subs(SN(0),SXAB00).subs(KN(0),TXAB00).subs(AX,X_A).subs(BX,X_B).subs(XPA,X_P-X_A).subs(a,aa).subs(b,bb).subs(p,aa+bb))
expr=sd10.subs(SN(0),SXAB00).subs(XPA,X_PA)
exprt=expr.diff(AX).diff(AX)*(-1/2)
w2=sympy.N(exprt.subs(AX,X_A).subs(BX,X_B).subs(XPA,X_P-X_A).subs(a,aa).subs(b,bb))
print([w1,w2])


w1=sympy.N(tx01.subs(SN(0),SXAB00).subs(KN(0),TXAB00).subs(AX,X_A).subs(BX,X_B).subs(XPB,X_P-X_B).subs(a,aa).subs(b,bb).subs(p,aa+bb))
expr=sd01.subs(SN(0),SXAB00).subs(XPB,X_PB)
exprt=expr.diff(BX).diff(BX)*(-1/2)
w2=sympy.N(exprt.subs(AX,X_A).subs(BX,X_B).subs(XPA,X_P-X_A).subs(a,aa).subs(b,bb))
print([w1,w2])

w1=sympy.N(tx20.subs(SN(0),SXAB00).subs(KN(0),TXAB00).subs(AX,X_A).subs(BX,X_B).subs(XPA,X_P-X_A).subs(XPB,X_P-X_B).subs(a,aa).subs(b,bb).subs(p,aa+bb))
expr=sd20.subs(SN(0),SXAB00).subs(XPA,X_PA).subs(XPB,X_PB).subs(a,aa).subs(b,bb).subs(p,aa+bb)
#print(expr)
exprt=expr.diff(BX).diff(BX)*(-1/2)
w2=sympy.N(exprt.subs(AX,X_A).subs(BX,X_B).subs(XPA,X_P-X_A).subs(a,aa).subs(b,bb))
print([w1,w2])

w1=sympy.N(tx02.subs(SN(0),SXAB00).subs(KN(0),TXAB00).subs(AX,X_A).subs(BX,X_B).subs(XPA,X_P-X_A).subs(XPB,X_P-X_B).subs(a,aa).subs(b,bb).subs(p,aa+bb))
expr=sd02.subs(SN(0),SXAB00).subs(XPA,X_PA).subs(XPB,X_PB).subs(a,aa).subs(b,bb).subs(p,aa+bb)
#print(expr)
exprt=expr.diff(AX).diff(AX)*(-1/2)
w2=sympy.N(exprt.subs(AX,X_A).subs(BX,X_B).subs(XPA,X_P-X_A).subs(a,aa).subs(b,bb))
print([w1,w2])

