.. _differences:

About ASTRA Toolbox and ASTRA KernelKit
=======================================


What is ASTRA KernelKit?
^^^^^^^^^^^^^^^^^^^^^^^^

What is ASTRA Toolbox?
^^^^^^^^^^^^^^^^^^^^^^

Feature comparison with ASTRA Toolbox
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Is KernelKit compatible?
^^^^^^^^^^^^^^^^^^^^^^^^

Short answer: no. Maintaining the ASTRA interface
is not easily possible, and KernelKit does not aim to
be a drop-in replacement for ASTRA Toolbox. However, for
``ConeProjector`` and ``ConeBackprojector`` ASTRA Toolbox
compatible projectors are available, making it easy to cross-test
implementations with the ASTRA Toolbox. To replicate the toolbox 
entirely (i.e., same timings/kernel settings), have a look at the examples or tests.


