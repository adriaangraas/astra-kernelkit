.. _installation:

Best practices
==============

1. Use page-locked memory
-------------------------
Page-locked, or *pinned* memory, avoids paging.


.. code-block:: console

   (.venv) $ pip install lumache


2. Allocate large data in CUDA Unified Memory
---------------------------------------------
To use Lumache, first install it using pip:

.. code-block:: console

   (.venv) $ pip install lumache

Note: we haven't trie

3. Use projectors, not operators
--------------------------------


4. Use CUDA graphs for small reconstructions
--------------------------------------------

For smaller reconstructions, CPU time is a large bottleneck.
