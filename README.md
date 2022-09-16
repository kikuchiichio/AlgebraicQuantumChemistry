# AlgebraicQuantumChemistry
Research of quantum chemistry through computational algebra

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


2022.9.14

I put three programs  in the repository:

1 PYUHF.py  (Python program, UHF computation of simple molecules)

2 PYRHF.py  (Python program, RHF computation of simple molecules)

These programs generate polynomial equations and write them in small sub-programs, which you should compute through the computer algebra package Singular. 

3 ERI.py (Python program. By this program and symbolic differentiation, you can compute every analytic formula of one and two-electron integrals of GTO required in molecular orbital computations. Obara-Saika recursion scheme shall be implemented soon. )

It is an initial simple lesson for beginners. Nevertheless, I expect that it shall reveal to you the algebraic scheme to execute full computations of quantum chemistry.
 


