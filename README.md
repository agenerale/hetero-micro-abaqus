# FEM Property Generation for MICRO2D

[Paper](https://link.springer.com/article/10.1007/s40192-023-00340-4)

## Description

This repository contains research code associated with work entitled *"MICRO2D: A Large, Statistically Diverse, Heterogeneous Microstructure Dataset"*. This specific repository contains scripts for evaluating the properties of 2D heterogeneous microstructure thermal and elastic effective properties with ABAQUS/CAE scripting.

If you find this code useful, interesting, or are open to collaboration, please reach out!
Alternatively, if you have any questions regarding the contents of this repository, feel free to contact either the first author of this work, Andreas Robertson [aerober@sandia.gov](aerober@sandia.gov), or myself at [agenerale3@gatech.edu](agenerale3@gatech.edu). 

Please consider citing this work (expand for BibTeX).

<details>
<summary>
Robertson, A.E., Generale, A.P., Kelly, C., Buzzy, M.O., Kalidindi, S.R. MICRO2D: A Large, Statistically Diverse, Heterogeneous Microstructure Dataset. Integrating Materials and Manufacturing Innovation, (2024).
</summary>

```bibtex
@article{robertson_micro2d_2024,
	title = {{MICRO2D}: {A} {Large}, {Statistically} {Diverse}, {Heterogeneous} {Microstructure} {Dataset}},
	volume = {13},
	issn = {2193-9772},
	shorttitle = {{MICRO2D}},
	url = {https://doi.org/10.1007/s40192-023-00340-4},
	doi = {10.1007/s40192-023-00340-4},
	abstract = {The availability of large, diverse datasets has enabled transformative advances in a wide variety of technical fields by unlocking data scientific and machine learning techniques. In Materials Informatics for Heterogeneous Microstructures capitalization on these techniques has been limited due to the extreme complexity of generating or curating sizeable heterogeneous microstructure datasets. Historically, this difficulty can be attributed to two main hurdles: quantification (i.e., measuring microstructure diversity) and curation (i.e., generating diverse microstructures). In this paper, we present a framework for curating large, statistically diverse mesoscale microstructure datasets composed of 2-phase microstructures. The framework generates microstructures which are statistically diverse with respect to their n-point statistics—the primary emphasis is on diversity in their 2-point statistics. The framework’s foundation is a proposed set of algorithms for synthesizing salient 2-point statistics and neighborhood distributions. We generate statistically diverse microstructures by using the outputs of these algorithms as inputs to a statistically conditioned Local-Global Decomposition generation procedure. Finally, we demonstrate the proposed framework by curating MICRO2D, a diverse, large-scale, and open source heterogeneous microstructure dataset comprised of 87, 379 2-phase microstructures. The contained microstructures are periodic and \$\$256 {\textbackslash}times 256\$\$pixels. The dataset also contains salient homogenized elastic and thermal properties computed across a range of constituent contrast ratios for each microstructure. Using MICRO2D, we analyze the statistical and property diversity achievable via the proposed framework. We conclude by discussing important areas of future research in microstructure dataset curation.},
	language = {en},
	number = {1},
	urldate = {2024-12-08},
	journal = {Integrating Materials and Manufacturing Innovation},
	author = {Robertson, Andreas E. and Generale, Adam P. and Kelly, Conlain and Buzzy, Michael O. and Kalidindi, Surya R.},
	month = mar,
	year = {2024},
	keywords = {2-point statistics, Big Data, Dataset Curation, Diffusion-based Deep Learning, Heterogeneous Microstructures, Local-Global Decompositions},
	pages = {120--154},
}
```
</details>

## Contents
This section provides a brief description of the contents of this repository.

1. *main.py*: Executes batches of 2D microstructures and returns separate output files with thermal and mechanical properties.

2. *abaqus_helpers.py*: Main functions for executing ABAQUS simulations and post-processing results.
