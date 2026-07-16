"""CPAG Stage C0 prototype: convexified predictor-corrector aerocapture guidance.

Python spike of the Rataczak/McMahon/Boyd CPAG formulation (JGCD 2025,
doi:10.2514/1.G008685) on this repo's dynamics. Deliverables: SCP convergence
verification across the corridor + embedded-solver benchmark (Clarabel vs OSQP).
Stage C1 ports the winning formulation/solver to Rust as the 8th guidance scheme.
"""
