
Sparse Cocktail: Every Sparse Pattern Every Sparse Ratio All At Once
====================================================


Official code for ICML'24 Paper: "Sparse Cocktail: Every Sparse Pattern Every Sparse Ratio All At Once
". 

## Overview
![image](https://github.com/VITA-Group/Shake-to-Leak/assets/15967092/83186da0-8c9d-4b70-88da-3187a07246c4)
Sparse Neural Networks (SNNs) have received voluminous attention for mitigating the explosion in computational costs and memory footprints of modern deep neural networks. Most state-of-the-art training approaches seek to find a single high-quality sparse subnetwork with a preset sparsity pattern and ratio, making them inadequate to satisfy platform and resource variability. Recent approaches attempt to jointly train multiple subnetworks (we term as ``sparse co-training") with a fixed sparsity pattern, to allow switching sparsity ratios subject to resource requirements. In this work, we expand the scope of sparse co-training to cover diverse sparsity patterns and multiple sparsity ratios at once. We introduce Sparse Cocktail, the first sparse co-training framework that co-trains a suite of sparsity patterns simultaneously, loaded with multiple sparsity ratios which facilitate harmonious switch across various sparsity patterns and ratios at inference depending on the hardware availability. More specifically, Sparse Cocktail alternatively trains subnetworks generated from different sparsity patterns with a gradual increase in sparsity ratios across patterns and relies on an unified mask generation process and the Dense Pivot Co-training to ensure the subnetworks of different patterns orchestrate their shared parameters without canceling each other’s performance. Experiment results on image classification, object detection, and instance segmentation illustrate the favorable effectiveness and flexibility of Sparse Cocktail, pointing to a promising direction for sparse co-training.

## TODO
Update Readme.md
