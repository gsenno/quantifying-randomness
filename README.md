# Code for the paper "Quantifying the intrinsic randomness of quantum measurements".

There are two python files:

1. **beamsplitter_with_ineff_detectors.py:** This implements the simulation in the section entitled "Application to QRNG". In particular, it allows to reproduce the plot therein.
    - Library requirements: Numpy, PICOS, Qutip,Matplotlib and a SDP solver compatible with PICOS (we used Mosek).
    
2. **pm_simulability.py:** This implements the (feasibility version of the) SDP in Lemma 6 of Appendix B. In particular, it allows to verify that for theta=0, the POVM in the proof of Theorem 4 is, indeed, not a convex combination of projective measurements as claimed.
    - Library requirements: Numpy, PICOS, Qutip and a SDP solver compatible with PICOS (we used Mosek).

