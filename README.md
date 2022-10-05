# AlgebraicQuantumChemistry
## Research of quantum chemistry through computational algebra

In this project,  we study the first-principles approach of quantum chemistry through symbolic computation.

In [KIKUCHI2013], Akihito Kikuchi presented an example of the first-principles approach of quantum chemistry, wherein a contemporary technique of computational algebraic geometry is adopted. The author of that article prepared the analytic energy functional of one hydrogen molecule. He approximated it by a polynomial, and, through symbolic differentiation, he composed a set of polynomial equations which gives the minima of the energy functional.  Finally, he was successful in solving it using the Gr\"obner basis technique. To construct one- or two-electron integrals, he used the Slater-Type Orbitals (STO) for atomic orbitals.

However, the STO  is not the standard way of today's quantum chemistry, for it is an unsettled problem how to construct analytical representations of two-electron integrals in polyatomic systems using STO.  Currently, as an alternative, most researchers use Gaussian-Type-Orbitals (GTO) because of the easiness of GTO in symbolic manipulation.

For this reason, we choose GTO as the basic ingredient of the research. We prepare the electron integrals using GTO, and according to the fundamental scheme of [KIKUCHI2013], we derive the polynomial equations to unravel the quantum states. We solve them through computational algebra packages, such as Singular, using symbolic-numeric computations.

In this repository, we put our computer programs which enable us to pursue the following targets.

(1) The analytical computation of atomic-orbital integrals by GTO

(2) The algebraic method to solve polynomial equations by the Groebner basis technique

(3) EXAMPLES of Un-restricted Hartree-Fock (UHF) or Restricted Hartree-Fock (RHF) computations.

[KIKUCHI2013] @article{KIKUCHI2013, author="Kikuchi,Akihito", title="An approach to first principles electronic structure computation by symbolic-numeric computation", journal="QScience Connect",
volume="2013:14",year=2013, note="http://dx.doi.org/10.5339/connect.2013.14"}


2022.9.22

I put three programs  in the repository:

(1) PYUHF.py  (Python program, UHF computation of simple molecule: HeH+)

(2) PYRHF.py  (Python program, RHF computation of simple molecule: HeH+)

These programs generate polynomial equations and write them in small sub-programs, which you should compute through the computer algebra package Singular. 

(3) ERI.py (Python program. By this program and symbolic differentiation, you can compute every analytic formula of one and two-electron integrals of GTO required in molecular orbital computations. Obara-Saika recursion scheme is implemented for the computations involving general Caetesian Gaussian functions; but we can do without it, since we can use symbolic differentiation. )

It is an initial simple lesson for beginners. Nevertheless, I expect that it shall reveal to you the algebraic scheme to execute full computations of quantum chemistry.
 
2022/9/28
I added three programs  in the repository:
hese Python programs compute the electronic structure of H3+ in the equilateral triangle. 
 
 The atoms are indexed by A, B, and C, or a, b, and c.
 The STO-3G are determined by the parameters ZETA, COEFF, and EXPON. 
 
 (4) SZ-SIM-H3P-RHF
       For RHF computation.  
       The variables (x,y,z) : the LCAO coefficients of 1s orbitals on the vertices.
       The variable e : the orbital energy
       The variable R: the length of the edges, fixed at a positive number.
 
 (5) SZ-SIM-H3P-UHF
       For UHF computation.  
       The variables (x,y,z) : the LCAO coefficients of 1s orbitals on the vertices, for spin alpha
       The variables e : the orbital energy, for spin alpha
       The variables (u,v,w) : the LCAO coefficients of 1s orbitals on the vertices, for spin beta
       The variables f : the orbital energy, for spin beta
       The variable R: the length of the edges, fixed at a positive number.
 
 (6) SZ-SIM-H3P-RHF-DeterminR, SZ-SIM-H3P-RHF-DeterminR-Version2
       The variables (x,x,x) : the LCAO coefficients of 1s orbitals on the vertices.
       The variable e : the orbital energy
       The variable R: the length of the edges, to be optimized as well as other variables.

2022/10/01
  We often meet limiting cases of molecular integrals where the center positions of involved atomic bases are coincident. 
  In those cases, the formulas generated for different atomic basis centers yield the terms 0/0, which should remain finite in theory.
  However, in practice, if we naively substitute the numerical parameters in those formuras, they falsely diverge. 
  To avoid this trouble, we should separate the coincident centers silghtly (by an interterminate T), 
  and then compute the limit by leading T to zero. 
  
  From the following programs you shall learn how we should treat the symbolic limits.
 
  (7)SZ-SIM-PZ4.py
      A test program to compute the analytic forumulas of molecular integrals and the electronic structure of a square molecule composed of pz orbitals.
  (8)SZ-SIM-SP.py
        A test program to compute the analytic forumulas of molecular integrals and the electronic structure or
        of a polyatomic molecule composed of s, px, py, pz orbitals.
        
       (7) (8) are still beta editions to check the well-composedness of the algorithm.
   
These programs generate polynomial equations and write them in small sub-programs (in "SCIPT.txt"),
which you should compute through the computer algebra package Singular. 

202/10/05
 (9) SZ-SIM-SP-VER2.py
        A test program to compute the analytic forumulas of molecular integrals and the electronic structure 
        of a polyatomic molecule composed of s, px, py, pz orbitals. I eliminated apparent bugs from SZ-SIM-SP.py. 
        It shows you a model use of functions in this library.  
        
