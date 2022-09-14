#!/usr/bin/env python
# coding: utf-8

# In[74]:


import sympy as sympy
from sympy import symbols, Function,expand,core,sqrt
import numpy as np

from sympy import symbols, Function,expand,sqrt
import numpy as np
import copy

S12,T11,T12,T22,V11A,V12A,V22A,V11B,V12B,V22B,V1111,V2111,V2121,V2211,V2221,V2222=[0 for _ in range(16)]
S=[[0 for _ in range(2)] for _ in range(2)]
X=[[0 for _ in range(2)] for _  in range(2)]
NM=[[0 for _ in range(2)] for _  in range(2)]
XT=[[0 for _ in range(2)] for _ in  range(2)]
H=[[0 for _ in range(2)] for _ in  range(2)]
F=[[0 for _ in range(2)] for _ in  range(2)]
G=[[0 for _ in range(2)] for _ in  range(2)]
C=[[0 for _ in range(2)] for _ in  range(2)]
FPRIME=[[0 for _ in range(2)] for _  in range(2)]
CPRIME=[[0 for _ in range(2)] for _  in range(2)]
P=[[0 for _ in range(2)] for _  in range(2)]
ODLPP=[[0 for _ in range(2)] for _  in range(2)]
E=[[0 for _ in range(2)] for _  in range(2)]
TT=[[[[0 for _ in range(2)] for _  in range(2)] for _ in range(2)] for _ in range(2)]


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
#
#
#
ZA=2.0
ZB=1.0
#
# ATOMIC DISTANCE 
#
R=symbols("R",positive=True)
# WAVEFUNCTION (x,y)
x, y = symbols("x y")


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

