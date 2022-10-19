# AlgebraicQuantumChemistry
## Research of quantum chemistry through computational algebra
## Language: Python with sympy, numpy, and other libraries. 
## Aim: 
### Computation of analitycal formulas of molecular integrals with GTO (Gaussian-Type Orbitals)
### Study of electronic structures of molecule from the viewpoint of algebraic variety

In this project,  we study the first-principles approach of quantum chemistry through symbolic computation.

In [KIKUCHI2013], Akihito Kikuchi presented an example of the first-principles approach of quantum chemistry, wherein a contemporary technique of computational algebraic geometry is adopted. The author of that article prepared the analytic energy functional of one hydrogen molecule. He approximated it by a polynomial, and, through symbolic differentiation, he composed a set of polynomial equations which gives the minima of the energy functional.  Finally, he was successful in solving it using the Gr\"obner basis technique. To construct one- or two-electron integrals, he used the Slater-Type Orbitals (STO) for atomic orbitals.

However, the STO  is not the standard way of today's quantum chemistry, for it is an unsettled problem how to construct analytical representations of two-electron integrals in poly-atomic systems using STO.  Currently, as an alternative, most researchers use Gaussian-Type-Orbitals (GTO) because of the easiness of GTO in symbolic manipulation.

For this reason, we choose GTO as the basic ingredient of the research. We prepare the electron integrals using GTO, and according to the fundamental scheme of [KIKUCHI2013], we derive the polynomial equations to unravel the quantum states. We solve them through computational algebra packages, such as Singular, using symbolic-numeric computations.

In this repository, we put our computer programs which enable us to pursue the following targets.

(1) The analytical computation of atomic-orbital integrals by GTO

(2) The algebraic method to solve polynomial equations by the Groebner basis technique

(3) EXAMPLES of Unrestricted Hartree-Fock (UHF) or Restricted Hartree-Fock (RHF) computations.


The progrmams are written in Python, with the intensive usage of sympy library.


[KIKUCHI2013] 
@article{KIKUCHI2013, 
title="An approach to first principles electronic structure computation by symbolic-numeric computation", 
author="Kikuchi,Akihito", 
journal="QScience Connect",
volume="2013:14",year=2013, note=" http://dx.doi.org/10.5339/connect.2013.14 "}

We also make use of the technique of computer algebra expounded by following resources.

@book{kikuchiquantum2021,
  title={Quantum Mechanics built on Algebraic Geometry: Emerging Physics through Symbolic Computation},
  author={Kikuchi, Akihito},
  year={2021},
  publisher={Eliva Press}
}

@article{kikuchigalois,
  title={Galois and Class Field Theory for Quantum Chemists},
  author={Kikuchi, Ichio and Kikuchi, Akihito},
  publisher={OSF Preprints},journal={OSF Preprints},
  year={2020},
  note={\url{ https://osf.io/preprints/n46rf/ }}
}

@article{kikuchi2019computational,
  title={Computational Algebraic Geometry and Quantum Mechanics: An Initiative toward Post Contemporary Quantum Chemistry},
  author={Kikuchi, Akihito and Kikuchi, Ichio},
  journal={Journal of Multidisciplinary Research and Reviews},
  volume={1},
  pages={47--79},
  year={2019},
  publisher={Innovationinfo},
  note={\url{ https://www.innovationinfo.org/journal-of-multidisciplinary-research-and-reviews } \url{ DOI: 10.3619/JMRR.1000118 }}
}

@book{kikuchi2018computer,
  title={Computer Algebra and Materials Physics: A Practical Guidebook to Group Theoretical Computations in Materials Science},
  author={Kikuchi, Akihito},
  volume={272},
  year={2018},
  publisher={Springer}
}


2022/09/22

I put three programs  in the repository:

(1) PYUHF.py  (Python program, UHF computation of simple molecule: HeH+)

(2) PYRHF.py  (Python program, RHF computation of simple molecule: HeH+)

These programs generate polynomial equations and write them in small sub-programs, which you should compute through the computer algebra package Singular. 

(3) ERI.py (Python program. By this program and symbolic differentiation, you can compute every analytic formula of one and two-electron integrals of GTO required in molecular orbital computations. Obara-Saika recursion scheme is implemented for the computations involving general Cartesian Gaussian functions; but we can do without it, since we can use symbolic differentiation. ) This file is slightly rewritten into PYERI-20221013.py

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
  In those cases, the formulas (prepared for distinct atomic basis centers) yield the terms 0/0, which should remain finite in theory.
  However, in practice, if we naively substitute the numerical parameters in the formulas, they falsely diverge. 
  
  To solve this matter, we should separate the coincident centers slightly (by an indeterminate T), 
  and then compute the limit by leading T to zero. 
  
  In so far as we use numbers and symbols indiscriminately, we should take special care not to be bothered by numerical errors.
  
  From the following programs you shall learn how to do when you confront with this sort of difficulty.
 
  (7)SZ-SIM-PZ4.py
      A test program to compute the analytic formulas of molecular integrals and the electronic structure of a square molecule composed of pz orbitals.
  (8)SZ-SIM-SP.py
        A test program to compute the analytic formulas of molecular integrals and the electronic structure or
        of a poly-atomic molecule composed of s, px, py, pz orbitals.
        
       (7) (8) are still beta editions to check the soundness of the algorithm.
   
These programs generate polynomial equations and write them in small sub-programs (in "SCIPT.txt"),
which you should compute through the computer algebra package Singular. 

2020/10/05

 (9) SZ-SIM-SP-VER2.py
        A test program to compute the analytic formulas of molecular integrals and the electronic structure 
        of a polyatomic molecule composed of s, px, py, pz orbitals. I eliminated apparent bugs from SZ-SIM-SP.py. 
        It shows you a typical use of functions in this library.  
        
2022/10/19

(10) SIMPY-ERI-20221018.py
(11) SIMPY-ERI-20221019.py
      These programms demonstrate how to compute the analytic formulas of two-electron integrals smartly. 
      The components of the programs are extracted from the already-registered programs.  
      The tentative computations in (10) revealed the wrong behaviour in the Obara-Saika recursion of the programs (3)(7)(8).
      I removed this defect and rewrite the program into (11).
      The cause of that wrong behavior is discussed in the head commentary parts of (10) and (10). 
      To be short, the implementation in (10) uses recursion foumulas sometimes in an improper way, 
      whereby some of the terms in the recusion get out of the domain of definition and the contributions from certain terms are nullified.
      
      
      
