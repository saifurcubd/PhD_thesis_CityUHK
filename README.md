
<p align="center">

# 🧬 Computational Intelligence Frameworks for Precision Oncology:
Early Cancer Detection and Anticancer Drug Combination Response Prediction


**PhD Thesis Repository**

**Saifur Rahaman**
Department of Computer Science
City University of Hong Kong

</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue)
![WEKA](https://img.shields.io/badge/WEKA-3.8-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![License](https://img.shields.io/badge/License-Academic-green)

</p>

---

# 📖 Overview

This repository contains the implementation materials, datasets, experimental resources, source code, notebooks, supplementary documents, and reproducibility materials supporting my PhD thesis entitled:

> **Computational Intelligence Frameworks for Precision Oncology: Early Cancer Detection and Prediction of Anticancer Drug Combination Responses**

The thesis develops novel computational intelligence frameworks addressing two major challenges in precision oncology:

* **Liquid biopsy-based early cancer detection**
* **Multimodal prediction of synergistic anticancer drug combination responses**

The repository is organized according to the thesis chapters to facilitate examination, reproducibility, and future research.

---

# 📑 Table of Contents

* Overview
* Research Framework
* Thesis Contributions
* Repository Structure
* Chapter 4 — CancerEMC
* Chapter 5 — ClinicalEarlyCancerDF
* Chapter 6 — MACSynDCRP
* Experimental Workflow
* Datasets
* Software Components
* Supplementary Materials
* Reproducibility
* Citation
* Contact
* Acknowledgements

---

# 🏗 Overall Research Framework

<p align="center">
<img src="/Docs/Figures/research_framework.png" width="900">
</p>

The proposed research integrates three complementary computational intelligence frameworks to address the continuum of precision oncology, from **early cancer detection** to **personalized anticancer therapy**.

---

# 🎯 Thesis Contributions

The thesis proposes three major computational frameworks.

| Chapter   | Framework                 | Research Focus                                                                                                      |
| --------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Chapter 4 | **CancerEMC**             | Ensemble machine learning framework for early cancer detection using cfDNA mutation profiles and protein biomarkers |
| Chapter 5 | **ClinicalEarlyCancerDF** | Generalized adaptive computational framework for optimal early cancer detection pipeline selection                  |
| Chapter 6 | **MACSynDCRP**            | Multimodal deep learning framework for synergistic anticancer drug combination response prediction                  |

---

# 📂 Repository Structure

```text
PhD_Thesis_CityU/

├── README.md
├── LICENSE
├── CITATION.cff
│
├── docs/
│   ├── Thesis.pdf
│   ├── Supplementary_Response.pdf
│   ├── Repository_Guide.pdf
│   └── figures/
│
├── Chapter4_CancerEMC/
├── Chapter5_ClinicalEarlyCancerDF/
├── Chapter6_MACSynDCRP/
│
├── shared/
│
└── legacy/
```

---

# 🔬 Chapter 4 — CancerEMC

<p align="center">
<img src="Docs/Figures/CancerEMC.png" width="850">
</p>

CancerEMC is an ensemble machine learning framework developed for liquid biopsy-based early cancer detection by integrating circulating cell-free DNA mutation signatures and circulating protein biomarkers.

### Main Contributions

* Multi-modal cfDNA and biomarker integration
* Ensemble learning framework
* Cancer detection and localization
* Validation using the CancerSEEK cohort

Repository:

```
Chapter4_CancerEMC/
```
---

# 🧬 Chapter 5 — ClinicalEarlyCancerDF

<p align="center">
<img src="Docs/Figures/ClinicalEarlyCancerDF.png" width="900">
</p>

ClinicalEarlyCancerDF represents one of the principal contributions of this thesis for liquid biopsy-based early cancer detection.

Unlike conventional prediction models, ClinicalEarlyCancerDF is designed as a **generalized adaptive computational framework** capable of automatically identifying an optimal computational pipeline for heterogeneous early cancer liquid biopsy datasets.

The framework integrates:

* Best Recursive Feature Elimination (bRFE)
* Efficient SMOTE-based Oversampling Method (ESOM)
* Ensemble Optimization Learning Framework (EOLF)

to adaptively determine:

* preprocessing strategy
* feature subset
* data balancing strategy
* optimization algorithm
* learning model

according to the characteristics of each dataset.

The framework was validated using multiple independent liquid biopsy datasets from different patient cohorts, demonstrating its capability to generate dataset-specific computational pipelines for early cancer detection.

Repository:
```
Chapter5_ClinicalEarlyCancerDF/
```
---

# 💊 Chapter 6 — MACSynDCRP

<p align="center">
<img src="Docs/Figures/MACSynDCRP.png" width="900">
</p>
MACSynDCRP is a multimodal deep learning framework developed for predicting synergistic anticancer drug combination responses.

The framework integrates:
* Drug molecular descriptors
* Graph neural networks
* Transcriptomics
* Pharmacogenomics
* Protein interaction networks
* Drug-Drug and Drug-Cell_line interaction
* VAE for feature reduction
* Deep Learning Prediction models

through an ensemble hybrid deep learning architecture for precision drug response prediction.

Repository:

```
Chapter6_MACSynDCRP/
```

---

# ⚙ Experimental Workflow

<p align="center">
<img src="Docs/Figures/Experimental_Workflow.png" width="950">
</p>

This repository documents the complete research workflow, including:

* Data acquisition
* Data preprocessing
* Feature engineering
* Adaptive feature selection
* Model optimization
* Machine learning and deep learning
* Performance evaluation
* External validation
* Explainability analysis

For ClinicalEarlyCancerDF, the computational pipeline is selected adaptively for each dataset using the proposed generalized framework.

---

# 💾 Datasets

This repository includes publicly available datasets and processed datasets used throughout the thesis.

Examples include:

* CancerSEEK
* DELFI
* External validation cohorts
* DrugComb
* Merck Drug Combination Dataset
* Additional benchmark datasets

Please refer to the corresponding chapter directories for dataset descriptions and source information.

---

# 🖥 Software Components

The implementation consists of multiple research modules developed throughout the PhD study.

These include:

* Python notebooks
* WEKA tools
* Feature selection modules
* Optimization modules
* Deep learning models
* Visualization scripts
* Experimental analysis tools

The implementation evolved over several years using Python and WEKA. The repository organizes these materials according to the methodology presented in the thesis to facilitate examination and reproducibility.

---

# 📚 Supplementary Materials

The repository also contains:

* Supplementary response documents
* Prediction outputs
* WEKA experiment results
* Ablation analyses
* Additional validation experiments
* Pipeline documentation

---

# 🔁 Reproducibility

This repository has been organized to improve the transparency and reproducibility of the research presented in the thesis.

Available materials include:

* Processed datasets
* Source code
* Implementation notebooks
* Experimental outputs
* Prediction files
* Software modules
* Documentation linking the implementation to the corresponding thesis chapters

Additional implementation materials and documentation will continue to be organized and updated.

---

# 📖 Citation

If you use this work, please cite:

> Saifur Rahaman.
> **Computational Intelligence Frameworks for Precision Oncology: Early Cancer Detection and Prediction of Anticancer Drug Combination Responses.**
> PhD Thesis, City University of Hong Kong.

---

# 📧 Contact

**Saifur Rahaman**

PhD Candidate
City University of Hong Kong

Email: *srahaman2-c@my.cityu.edu.hk, saifurcubd@gmail.com**

---

# 🙏 Acknowledgements

I sincerely thank my respected PhD **Supervisor (Prof. Ka Chun Wong)**, Examiners, Qualifying Panel members, and mentor from Harvard Medical School, collaborators, research partners, and the City University of Hong Kong for their invaluable guidance and support throughout this research.

---

<p align="center">

**This repository accompanies the PhD thesis and is intended for academic research, examination, transparency, and reproducibility purposes.**

</p>
