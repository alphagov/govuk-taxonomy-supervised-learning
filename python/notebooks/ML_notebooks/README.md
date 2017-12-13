# ML_notebooks

Notebooks in this folder are arranged by the method used to deal with the multilabel problem.

These are:

* Simply dropping the the multiple taxons, leaving each conten titem with a single taxon. This is least desireable, but can be considered to be the 'baseline' option.
* Using a convolutional neural network. The architecture of the network can be designed to output a binary matrix indicating presence of the content iteam in each of the taxons. The usual binary crossentropy is not suitable as a loss function however, due to the sparse nature of the matrix: categorical cross entropy with logits should  be used instead, but its not natively available in keras (although it is available in tensorflow).
