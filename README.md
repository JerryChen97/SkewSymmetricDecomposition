# SkewSymmetricDecomposition
Using schur in scipy to decompose the skew symmetric real matrices (majorana modes)

`scipy.linalg.schur` can help us decompose a skew-symmetric real matrices, M = QAQ^T, where A is a block-diagonalized matrix as a direct sum of iσy, which is quite helpful in solving some (simple enough) fermionic systems.
However, zero modes are very common in these cases, which correspond to the null space of the matrix of interest; they could perturb the block-diagonalized form of the decomposition in basically random ways.
To avoid such annoying problems, I drafted a script by picking out those non-zero modes and separate zero modes.
