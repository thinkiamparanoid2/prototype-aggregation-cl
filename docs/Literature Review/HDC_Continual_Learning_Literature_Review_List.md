# Literature Review Reading List
## HDC Continual Learning for Crop Disease Recognition

This reading list is organized around the thesis direction:

> **Self-supervised visual representations → HDC/prototype-based continual learning → human-in-the-loop label budgets → crop-disease recognition**, with QHDC as a possible later extension.

---

## A. Essential HDC Foundations

1. **Kanerva, P. — “Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors.”**  
   - Year/Venue: 2009, *Cognitive Computation*  
   - Focus: Foundational HDC concepts, distributed representations, near-orthogonality, high-dimensional random vectors.  
   - Why it matters: Core theoretical foundation for the thesis.  
   - Link: https://dblp.org/rec/journals/cogcom/Kanerva09

2. **Kleyko, D. et al. — “A Survey on Hyperdimensional Computing aka Vector Symbolic Architectures, Part I: Models and Data Transformations.”**  
   - Year/Venue: 2022, *ACM Computing Surveys*  
   - Focus: HDC/VSA models, encodings, binding, bundling, transformations.  
   - Why it matters: Use to justify the chosen HDC representation and encoding design.  
   - Link: https://research.ibm.com/publications/a-survey-on-hyperdimensional-computing-aka-vector-symbolic-architectures-part-i-models-and-data-transformations

3. **Kleyko, D. et al. — “A Survey on Hyperdimensional Computing aka Vector Symbolic Architectures, Part II: Applications, Cognitive Models, and Challenges.”**  
   - Year/Venue: 2023, *ACM Computing Surveys*  
   - Focus: Applications, machine-learning use cases, challenges, open questions.  
   - Why it matters: Establishes HDC as a mature research field and identifies broader research gaps.  
   - Link: https://research.ibm.com/publications/a-survey-on-hyperdimensional-computing-aka-vector-symbolic-architectures-part-ii-applications-cognitive-models-and-challenges

4. **Vergés, C. et al. — “Classification using hyperdimensional computing: a review with comparative analysis.”**  
   - Year/Venue: 2025, *Artificial Intelligence Review*  
   - Focus: Modern HDC classification methods, encodings, classifiers, comparative evaluation.  
   - Why it matters: Strong recent reference for positioning HDC classification against conventional methods.  
   - Link: https://link.springer.com/article/10.1007/s10462-025-11181-2

5. **Heddes, M. et al. — “Torchhd: An Open Source Python Library to Support Research on Hyperdimensional Computing and Vector Symbolic Architectures.”**  
   - Year/Venue: 2023, *Journal of Machine Learning Research*  
   - Focus: PyTorch-based HDC/VSA library.  
   - Why it matters: Useful implementation and reproducibility reference.  
   - Link: https://www.jmlr.org/papers/v24/23-0300.html

---

## B. Online and Continual HDC

6. **Hernández-Cano, A. et al. — “OnlineHD: Robust, Efficient, and Single-Pass Online Learning Using Hyperdimensional System.”**  
   - Year/Venue: 2021, DATE  
   - Focus: Adaptive, single-pass HDC learning and online prototype refinement.  
   - Why it matters: Essential prior work showing that online HDC itself is not novel.  
   - Link: https://www.alehc.com/papers/date2021-onlinehd.pdf

7. **Hersche, M. et al. — “Constrained Few-Shot Class-Incremental Learning.”**  
   - Year/Venue: 2022, CVPR  
   - Focus: Frozen visual features, hyperdimensional embeddings, dynamic class memory, few-shot class-incremental learning.  
   - Why it matters: One of the closest conceptual predecessors to the thesis.  
   - Link: https://openaccess.thecvf.com/content/CVPR2022/html/Hersche_Constrained_Few-Shot_Class-Incremental_Learning_CVPR_2022_paper.html

8. **Yu, T. et al. — “Lifelong Intelligence Beyond the Edge using Hyperdimensional Computing.” (LifeHD)**  
   - Year/Venue: 2024, IPSN  
   - Focus: HDC-based lifelong/on-device learning from streams with limited supervision and resource constraints.  
   - Why it matters: Strong match to the edge, continual-learning, and limited-label motivation.  
   - Link: https://arxiv.org/abs/2403.04759

9. **Arockiaraj, A., Parikh, D., and Prasanna, V. — “ImageHD: Energy-Efficient On-Device Continual Learning of Visual Representations via Hyperdimensional Computing.”**  
   - Year/Venue: 2026, arXiv preprint  
   - Focus: HDC continual visual learning with deep visual features and bounded memory.  
   - Why it matters: Recent direct evidence that HDC + visual continual learning is an active research direction.  
   - Link: https://arxiv.org/abs/2604.21280

10. **Yan, Y. et al. — “XHDC: An Adaptive Hyperdimensional Computing Accelerator with Incremental Learning.”**  
    - Year/Venue: 2026, ISCAS  
    - Focus: Hardware acceleration for incremental HDC learning.  
    - Why it matters: Supports the update-cost and on-device efficiency discussion.  
    - Link: https://dblp1.uni-trier.de/rec/conf/iscas/YanHXCL26.html

---

## C. Prototype-Based Baselines

11. **Mensink, T. et al. — “Distance-Based Image Classification: Generalizing to New Classes at Near-Zero Cost.”**  
    - Year/Venue: 2013, *IEEE Transactions on Pattern Analysis and Machine Intelligence*  
    - Focus: Nearest Class Mean (NCM), low-cost incorporation of new classes.  
    - Why it matters: Critical evidence that cheap incremental prototype learning is not unique to HDC.  
    - Link: https://pubmed.ncbi.nlm.nih.gov/24051724/

12. **McDonnell, M. D. et al. — “RanPAC: Random Projections and Pre-trained Models for Continual Learning.”**  
    - Year/Venue: 2023, NeurIPS  
    - Focus: Frozen pretrained features, random projection, prototype accumulation for continual learning.  
    - Why it matters: Probably the most important classical prototype competitor to HDC in the current study.  
    - Link: https://papers.neurips.cc/paper_files/paper/2023/hash/2793dc35e14003dd367684d93d236847-Abstract-Conference.html

13. **Rebuffi, S.-A. et al. — “iCaRL: Incremental Classifier and Representation Learning.”**  
    - Year/Venue: 2017, CVPR  
    - Focus: Landmark class-incremental method using exemplars and nearest-mean classification.  
    - Why it matters: Important precedent combining neural representations with prototype-based class-incremental learning.  
    - Link: https://openaccess.thecvf.com/content_cvpr_2017/html/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.html

---

## D. Continual-Learning Neural Baselines

14. **Kirkpatrick, J. et al. — “Overcoming Catastrophic Forgetting in Neural Networks.”**  
    - Year/Venue: 2017, *PNAS*  
    - Focus: Elastic Weight Consolidation (EWC).  
    - Why it matters: The main theoretical reference for the EWC baseline.  
    - Link: https://pubmed.ncbi.nlm.nih.gov/28292907/

15. **Lopez-Paz, D. and Ranzato, M. — “Gradient Episodic Memory for Continual Learning.”**  
    - Year/Venue: 2017, NeurIPS  
    - Focus: Episodic memory, replay, continual-learning evaluation.  
    - Why it matters: Supports the replay baseline and forgetting/transfer metrics.  
    - Link: https://proceedings.nips.cc/paper_files/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html

---

## E. Human-in-the-Loop and Label-Efficiency Literature

16. **Settles, B. — “Active Learning Literature Survey.”**  
    - Year/Venue: 2009, University of Wisconsin Technical Report  
    - Focus: Active learning, uncertainty sampling, querying a human oracle under limited label budgets.  
    - Why it matters: The theoretical foundation for the agronomist-confirmation and human-label-budget component of the thesis.  
    - Link: https://minds.wisconsin.edu/handle/1793/60660

---

## F. Self-Supervised and Foundation-Model Representations

17. **Oquab, M. et al. — “DINOv2: Learning Robust Visual Features without Supervision.”**  
    - Year/Venue: 2024, *Transactions on Machine Learning Research*  
    - Focus: Strong transferable self-supervised visual representations.  
    - Why it matters: Supports the use of frozen SSL embeddings before HDC/prototype learning.  
    - Link: https://mlanthology.org/tmlr/2024/oquab2024tmlr-dinov2/

18. **Siméoni, O. et al. — “DINOv3.”**  
    - Year/Venue: 2025, Meta technical report / arXiv  
    - Focus: Large-scale self-supervised vision foundation models with strong frozen features.  
    - Why it matters: Directly relevant to the current DINOv3 encoder ladder in the project.  
    - Link: https://arxiv.org/abs/2508.10104

---

## G. HDC in Biological and Biomedical Domains

19. **Stock, M. et al. — “Hyperdimensional computing: A fast, robust, and interpretable paradigm for biological data.”**  
    - Year/Venue: 2024, *PLOS Computational Biology*  
    - Focus: HDC for biological and bioinformatics problems.  
    - Why it matters: Bridges general HDC research to biological and agricultural applications.  
    - Link: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012426

20. **Cumbo, F. et al. — “Designing vector-symbolic architectures for biomedical applications: ten tips and common pitfalls.”**  
    - Year/Venue: 2026, *PeerJ Computer Science*  
    - Focus: Practical VSA/HDC design guidance, including hybrid deep-feature pipelines.  
    - Why it matters: Useful methodological precedent for SSL feature extractor → HDC classifier designs.  
    - Link: https://sciety.org/articles/activity/10.7717/peerj-cs.3682

---

## H. Continual Learning for Plant Disease Recognition

21. **Wang — “Zero-exemplar deep continual learning for crop disease recognition: a study of total variation attention regularization in vision transformers.”**  
    - Year/Venue: 2024, *Frontiers in Plant Science*  
    - Focus: Zero-exemplar class-incremental crop-disease recognition; evaluates standard CL baselines.  
    - Why it matters: Essential evidence that continual plant-disease recognition already exists.  
    - Link: https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2023.1283055/full

22. **Li, Y. et al. — “Rehearsal-based class-incremental learning approaches for plant disease classification.”**  
    - Year/Venue: 2024, *Computers and Electronics in Agriculture*  
    - Focus: Rehearsal-based CIL for plant disease diagnosis and OOD evaluation.  
    - Why it matters: Direct agricultural continual-learning predecessor.  
    - Link: https://www.sciencedirect.com/science/article/pii/S0168169924006021

23. **Li, Y. et al. — “Non-exemplar class-incremental learning for continual plant diagnosis.”**  
    - Year/Venue: 2025, *Crop Protection*  
    - Focus: Non-exemplar CIL using self-supervised learning and prototype constraints.  
    - Why it matters: Very important because it already combines SSL, prototypes, and continual plant-disease learning, though not HDC.  
    - Link: https://www.sciencedirect.com/science/article/pii/S0261219424004976

24. **Lou, X. et al. — “VCPC: virtual contrastive constraint and prototype calibration for few-shot class-incremental plant disease classification.”**  
    - Year/Venue: 2025, *Plant Methods*  
    - Focus: Prototype calibration and contrastive constraints for few-shot class-incremental plant disease recognition.  
    - Why it matters: Shows that prototype-based agricultural FSCIL is already an active field.  
    - Link: https://link.springer.com/article/10.1186/s13007-025-01423-3

25. **Lin, et al. — “Virtual class-aware orthogonal and contrastive prototype calibration for few-shot class-incremental plant disease identification.” (VOCP)**  
    - Year/Venue: 2026, *Artificial Intelligence in Agriculture*  
    - Focus: Orthogonal prototype learning and prototype calibration for plant FSCIL.  
    - Why it matters: Particularly relevant because orthogonal prototype geometry is conceptually close to HDC.  
    - Link: https://www.sciencedirect.com/science/article/pii/S2589721726000796

26. **Gao, et al. — “Incremental learning with dynamic memory replay for crop-disease classification across diverse environments.”**  
    - Year/Venue: 2026, *Knowledge-Based Systems*  
    - Focus: Dynamic memory replay for crop-disease incremental learning across diverse environments.  
    - Why it matters: Recent replay-based agricultural CL context for memory and robustness comparisons.  
    - Link: https://www.sciencedirect.com/science/article/pii/S0950705126004430

---

## I. Plant-Disease Datasets and External Validation

27. **Wei, T. et al. — “Benchmarking In-the-Wild Multimodal Plant Disease Recognition and A Versatile Baseline.”**  
    - Year/Venue: 2024, ACM Multimedia  
    - Dataset: PlantWild  
    - Focus: 18,542 in-the-wild images across 89 classes with substantial visual variability.  
    - Why it matters: Strong candidate for external validation of the Dacon findings.  
    - Link: https://tqwei05.github.io/PlantWild/

28. **Zhang, et al. — “DLCPD-25: A Large-Scale and Diverse Dataset for Crop Disease and Pest Recognition.”**  
    - Year/Venue: 2025, *Sensors*  
    - Dataset: DLCPD-25  
    - Focus: 221,943 images, 23 crops, 203 disease/pest/healthy classes, long-tailed field/web data.  
    - Why it matters: Excellent candidate for a larger-scale HDC/prototype scalability study.  
    - Link: https://www.mdpi.com/1424-8220/25/22/7098

29. **Mohanty, S. P., Hughes, D. P., and Salathé, M. — “Using Deep Learning for Image-Based Plant Disease Detection.”**  
    - Year/Venue: 2016, *Frontiers in Plant Science*  
    - Dataset: PlantVillage  
    - Focus: Landmark plant-disease deep-learning benchmark on controlled images.  
    - Why it matters: Useful for discussing the gap between controlled-dataset accuracy and real-world field deployment.  
    - Link: https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2016.01419/full

---

## J. Optional QHDC / Quantum Extension

30. **Cumbo, F. et al. — “Quantum Hyperdimensional Computing: a foundational paradigm for quantum neuromorphic architectures.”**  
    - Year/Venue: 2026, *npj Unconventional Computing*  
    - Focus: Quantum implementations of HDC operations and initial IBM hardware validation.  
    - Why it matters: Main reference if the project later extends classical continual HDC into QHDC.  
    - Link: https://www.nature.com/articles/s44335-026-00064-6

31. **Xu, et al. — “QeHDC: Hyperdimensional Computing based on Quantum-enhanced binding and SuperClass Construction.”**  
    - Year/Venue: 2026, arXiv preprint  
    - Focus: Quantum-enhanced HDC with one-pass training, quantum binding, and superclass construction.  
    - Why it matters: Shows that QHDC is already expanding beyond a single foundational formulation.  
    - Link: https://arxiv.org/abs/2606.22421

---

# Recommended Reading Order

## First 12 Papers to Read

1. Kanerva — HDC foundation  
2. Kleyko et al. Part I — HDC models and transformations  
3. Vergés et al. — modern HDC classification review  
4. Hernández-Cano et al. — OnlineHD  
5. Hersche et al. — C-FSCIL  
6. Yu et al. — LifeHD  
7. Mensink et al. — NCM  
8. McDonnell et al. — RanPAC  
9. Siméoni et al. — DINOv3  
10. Wang — zero-exemplar continual crop-disease learning  
11. Li et al. — non-exemplar continual plant diagnosis  
12. Wei et al. — PlantWild  

---

# Literature Gap Suggested by These Papers

A defensible current gap is:

> **Existing work has independently demonstrated hyperdimensional online/class-incremental learning and continual plant-disease recognition, while recent agricultural few-shot class-incremental approaches increasingly rely on prototype-based representations. However, the relative contribution of hyperdimensional representation versus conventional prototype learning under a common self-supervised visual representation, controlled human-label budget, prequential evaluation, and held-out agricultural generalization remains insufficiently characterized.**

This gap fits the current experimental design using HDC, NCM, RanPAC, neural continual-learning baselines, SSL embeddings, human-label budgets, and held-out evaluation.
