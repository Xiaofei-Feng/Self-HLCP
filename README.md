Self-supervised clustering algorithm based on hierarchy of local density clusters and constraint propagation (Self-HLCP)
Abstract
    Clustering with a graph-cut scheme based on a weighted similarity graph has been an important research direction in the past years. Existing techniques for constructing a similarity graph usually focus more on local similarity while neglecting
the global structural information, resulting in poor performance on complex data. Self-supervised clustering has emerged to mitigate the lack of supervised information to improve graph-cut clustering, and it typically incorporates structural
information through regularization terms involving pairwise constraints or pseudo-labels, implicitly guiding data reconstruction during optimization. However, the current self-supervised clustering algorithms lack explicit consideration of
the structural information of data, missing some critical supervised constraints that are crucial for the correct discovery of clusters. To address this limitation, this article presents an explicit mechanism to directly mines and quantifies
pairwise constraints (must-link/cannot-link) from hierarchical density structures. Specifically, our method first partitions data into sub-clusters via local density distributions, then constructs soft constraints by evaluating similarity
and separability across hierarchical levels. These constraints are propagated to refine the similarity graph, ultimately enhancing spectral clustering performance. Experimental results on 12 real-world datasets demonstrate that compared to
existing graph cut and self-supervised algorithms, our method achieves average performance improvements of 8%, 15%, and 12% in ACC, ARI, and NMI metrics respectively.

INPUT: data:x, label:y
OUTPUT: labelnew

evaluation.py -- Indicator calculation
Self-HLCP.py -- Algorithm
