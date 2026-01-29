# scOT-LT: Optimal transport for label transfer in single-cell multi-omics integration
# Abstract
Single-cell multi-omics datasets are rapidly expanding, and integrating complementary modalities can provide a more comprehensive view of the molecular mechanisms underlying biological processes. However, cross-modality alignment remains challenging due to modality-specific measurement differences and mismatched cell-type proportions between modalities. Here, we present scOT-LT, a semi-supervised label-transfer framework that aligns scRNA-seq and scATAC-seq data using label-aware unbalanced optimal transport, which tolerates compisitional mismatch while favoring label-consistent correspondences. scOT-LT learns a shared embedding through UOT-guided alignment and transfers cell-type labels from the annotated scRNA-seq reference to unlabeled scATAC-seq via entropic OT coupling. Evaluations on multiple real-world datasets show that scOT-LT achieves strong modality mixing and high label-transfer accuracy, remains robust under downsampled scRNA-seq annotations, and can reliably detect novel cell types. Thus, scOT-LT not only improves integration and label transfer performance, but also yields an explicit and interpretable cross-modality coupling, providing a practical approach for multimodal integration and annotation.
# Overview
![](https://github.com/xiaoxi369/scUOTL/blob/main/figures/scOT-LT.png)
# System requirements
- Python >= 3.6
- numpy >= 1.26.4
- pandas >= 2.2.2
- scikit-learn >= 1.5.1
- scanpy >= 1.10.2
- POT >= 0.9.4
- seaborn >= 0.13.2
- torch >= 2.3.1
# Download code
Clone the repository with
 <pre>git clone https://github.com/xiaoxi369/scUOTL.git</pre>
# Running scUOTL
In terminal, run
 <pre>python src/main.py</pre> 
