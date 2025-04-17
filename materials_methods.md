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

Layer-specific morphological adaptations using MorphOMICs

 

3D reconstruction of layer-specific microglia

 

Reconstruction of microglia was performed in the same manner as in Colombo and Cubero, et al., (2022). Briefly, filtered and background-subtracted images were imported to Imaris 9.2.v (Bitplane Imaris) where the filament-tracing plugin was used to semi-automatically trace microglial processes. Seeding points were set to 1 µm and microglial soma was automatically detected when the diameter was 12 µm or greater. Disconnected segments were removed with a filtering smoothness of 0.6 µm. After the semi-automated tracing, cells that were either sitting at the border of the image or were partially traced were manually removed. (Write about blinding here)

 

VGluT2 staining was used to discriminate against V1 layers. VGluT2+ cells were highly concentrated in L1 and L4 (Supplementary Figure XXX, [Owe13] and references therein) forming a band. Microglia traces with somas located within the L1 or L4 band were labeled as L1 or L4 microglia, respectively. L2/3 microglia are those whose soma lie between L1 and L4 band, and L5/6 microglia are those below the L4 band.

 

The generated layer-discriminated skeleton images were converted from .ims format (Imaris) to .swc format [Stockley93] by first obtaining the 3D positions  and the diameter of each traced microglial process using the ImarisReader toolbox for MATLAB (https://github.com/PeterBeemiller/ImarisReader/) and then exporting for format standardization using the NL Morphology Converter (http://neuroland.org/). Artifacts from the 3D reconstructions automatically failed to be converted into .swc format.

 

Morphological analysis using morphOMICs

 

We implement morphOMICs [Colombo2022], a topological data analysis approach to extract microglia morphological phenotypes, to identify layer- and sex-specific morphological adaptations after saline, 1xKXA or saline+SAFit2 or 1xKXA+SAFit2. The algorithm uses a topological morphology descriptor, TMD [Kanari2018], to summarize the reconstructed complex branching morphology into a topological object called persistence barcode, where each bar in the barcode encodes a persistent microglial process and is described by a coordinate , where , the birth distance, is the radial distance from the soma where the process originates and , the death distance, is where it terminates and merges with a larger, more persistent process or with the soma. Each bar can then be represented as a point in a 2D-space, where each axis corresponds to the birth and death radial distances. Given a persistence barcode, we can convolve each point in the 2D-space with a Gaussian kernel with a fixed bandwidth, , to obtain the persistence image, a 100x100 pixel representation, which can be represented as a point in a 100x100 high-dimensional space.

 

Here, we only consider morphologies having barcodes with 5 bars or more for further analysis. To reduce biological variability coming from the highly dynamic terminal processes, we perform a bootstrapping procedure where we randomly subsample 30 unique microglia morphologies from a pool of a given condition (layer+sex+treatment) and take the union of their corresponding persistence barcodes to calculate the persistence images, with a kernel bandwidth µ. For each condition, we repeated this random sampling 1000 times per condition to obtain the pool of bootstrapped persistence images. Prior to manifold learning, we perform an initial dimensionality reduction by keeping bootstrapped persistence image pixels having a standard deviation greater than  across all the considered conditions. Then, we implemented the Uniform Manifold Approximation and Projection (UMAP [McInnes2018]) algorithm to nonlinearly infer a 2D manifold with the following parameters: "n_neighbors": 500, "min_dist": 0.05, "spread": 3.0, "random_state": 42541133 for reproducibility of results, and "metric": "manhattan" to reflect the definition of distances between persistence images (see [Kanari2018] for the robustness of the TMD distance). A Jupyter notebook, which reproduces the results in this paper, and the corresponding parameter file can be found in https://github.com/siegert-lab/V1_morphOMICs.

 

References:

[Colombo2022] Colombo, G., Cubero, R. J. A., Kanari, L., Venturino, A., Schulz, R., Scolamiero, M., ... & Siegert, S. (2022). A tool for mapping microglial morphology, morphOMICs, reveals brain-region and sex-dependent phenotypes. Nature Neuroscience, 25(10), 1379-1393.

[Kanari2018] Kanari, L., Dłotko, P., Scolamiero, M., Levi, R., Shillcock, J., Hess, K., & Markram, H. (2018). A topological representation of branching neuronal morphologies. Neuroinformatics, 16, 3-13.

[McInnes2018] McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426.

[Owe13] Owe, S. G., Erisir, A., & Heggelund, P. (2013). Terminals of the major thalamic input to visual cortex are devoid of synapsin proteins. Neuroscience, 243, 115-125.

[Stockley93] Stockley, E. W., Cole, H. M., Brown, A. D., & Wheal, H. V. (1993). A system for quantitative morphological measurement and electrotonic modelling of neurons: three-dimensional reconstruction. Journal of neuroscience methods, 47(1-2), 39-51.