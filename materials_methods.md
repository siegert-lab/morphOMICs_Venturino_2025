## Reconstruction of 3D-traced microglia
...Alessandro traced using IMS...
The generated skeleton images were converted from .ims format (Imaris) to .swc
format[cite] by first obtaining the 3D positions (x, y and z) and the diameter
of each traced microglial process using the ImarisReader toolbox for
MATLAB (https://github.com/PeterBeemiller/ImarisReader/) and then
exporting for format standardization using the NL Morphology Con-
verter (http://neuroland.org/). Artifacts from the 3D reconstructions
automatically failed to be converted into .swc format.

## Topological morphology descriptor
A topological data analysis algorithm, the TMD, was used to extract
topological phenotypes, called persistence barcodes, from 3D
morphological structures (https://github.com/BlueBrain/TMD/;14). In
brief, the 3D-reconstructed microglia is represented as a tree T rooted
in its soma. The TMD summarizes this tree by calculating  a 'persistence
barcode’, where each bar represents a persistent microglial process
with respect to a filtering function, that is, the radial distance from the
soma. Note that the persistence barcode that the TMD associates with
T under this filtering function is invariant under rotations about the
root and rigid translations of T in R3.
Each bar is described by two numbers: the radial distance, di, at
which a process originates; and the distance, bi, when it merges with a
larger, more persistent process or with the soma. A bar can be equiva-
lently represented as a point (di, bi) in a ‘persistence diagram’. We could
therefore convolve each point in the persistence diagram with a Gauss-
ian kernel and discretize it to generate a matrix of pixel values, encoding
the persistence diagram in a vector, called the ‘persistence image’.[cite pi]

# Average and bootstrapped persistence images
To construct the ‘average persistence image’ of a given condition, all the
persistence barcodes of microglia from the same condition are com-
bined before Gaussian convolution and discretization are performed.
We also constructed average persistence images by performing first
the Gaussian convolution and discretization of individual microglia
persistence barcodes before taking the pixel-wise average. This pro-
duced qualitatively similar results.
The bootstrapping method subsamples the microglial population
within a given condition, thereby introducing variations around the
average persistence image. Starting from the population of all micro-
glia from the same condition, called the ‘starting population’ of size
n (Supplementary Table 4), the persistence barcodes of a predefined
number of unique microglia, called the ‘bootstrap size’, are combined
to calculate the ‘bootstrapped persistence image’. We iterated this
process a predefined number of times, nsamples, with replacement to
obtain the ‘bootstrap sample’.

## Dimensionality reduction
Uniform manifold approximation and projection. A fast, nonlinear
dimensionality reduction algorithm, UMAP75, was applied to visual-
ize the high-dimensional pixel space of bootstrapped persistence
images using a 2D representation while preserving local and global
structures in the bootstrap samples (https://github.com/lmcinnes/
umap/)75. Given a bootstrap sample containing multiple conditions,
a TMD distance matrix containing pairwise distances between boot-
strapped persistence images in the bootstrap sample is calculated.
Principal components are then obtained using a singular value decom-
position of the TMD distance matrix. The first seven principal compo-
nents, where the elbow in the singular values is located, were used as
input to UMAP with n_neighbors = 50, min_dist = 1.0 and spread = 3.0.
Note that we tested for a wide range of parameter values that did notResource
https://doi.org/10.1038/s41593-022-01167-6
qualitatively change any of the aforementioned observations
(Extended Data Fig. 2f).

[cite]Stockley, E., Cole, H., Brown, A. & Wheal, H. A system for
quantitative morphological measurement and electronic
modelling of neurons: three-dimensional reconstruction. J.
Neurosci. Methods 47, 39–51 (1993).

