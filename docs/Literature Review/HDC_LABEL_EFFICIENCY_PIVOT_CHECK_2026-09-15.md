# Proposed label-efficiency pivot: method check

Date: 2026-09-15. Status: proposal review only, not an approved protocol amendment or experiment authorization. No learner, protocol, or completed result was changed.

## Assessment

Matched-label held-out comparison is a useful research direction, not a guaranteed positive result. Historical training-pool scores motivate the hypothesis but cannot establish it. A successful conclusion must name the tested variant, dataset, representation arms, comparator implementations, and budget range; it cannot establish a general HDC advantage.

## Adaptive HDC must be specified, not just renamed

OnlineHD Section III uses similarity-weighted correction: on a mistaken prediction it adds a weighted encoded example to the true-class vector and subtracts a separately weighted contribution from the predicted-class vector; a correct prediction receives no update. Equation (1) uses learning rate eta and each class's cosine similarity. It is not merely a constant learning rate multiplying ordinary bundling. The paper also distinguishes single-pass and iterative training. [OnlineHD original paper, Section III](https://www.alehc.com/papers/date2021-onlinehd.pdf)

Our current [HDC learner](../../ssl_hdc_pilot/scripts/continual_hdc.py) keeps accumulated bipolar encodings and re-binarizes its classification prototypes. A variant that changes both the update and representation should be described as an OnlineHD-style adaptation, not automatically an exact reproduction or an isolated update-mechanism ablation. Before testing, specify prototype precision, encoding, initialization, zero-vector handling, learning rate, mistake handling, bootstrap order, and number of passes. Any tuning must precede test release. Corrective updates do not guarantee absence of class interference; that is a methodological inference, not a new experimental finding.

## The existing comparator is not full RanPAC

The [implemented comparator](../../ssl_hdc_pilot/scripts/continual_prototypes.py) explicitly omits Gram-matrix decorrelation and ridge inversion. It is a frozen Gaussian-projection/ReLU cosine-prototype ablation. Published RanPAC accumulates second-order statistics and uses a regularized linear solve; these are material parts of its prediction rule, not incidental implementation details. Its paper discusses validation of the ridge parameter. [RanPAC original paper, Sections 3.3 and 4.3 and Appendix C.1](https://papers.neurips.cc/paper_files/paper/2023/file/2793dc35e14003dd367684d93d236847-Paper-Conference.pdf)

Two honest scopes are available: retain the existing comparator and consistently call it **RanPAC-style prototype ablation**, or separately implement and validate the published classifier under the same fixed representations. The latter is stronger comparative evidence but expands the engineering and tuning work. Reusing our existing frozen/adapted embeddings also does not reproduce the paper's optional first-session representation adaptation.

## Conditions for a defensible proposed comparison

- Use identical bootstrap examples, purchased stream-example IDs, order, budgets, class-registration rules, and paired seeds across all four heads; threshold matching alone is insufficient.
- Freeze models and selection choices using development data before scoring the untouched test set in a locked release. Test-only final reporting does not mean test-driven development.
- Predeclare a scalar curve summary such as normalized AULC, its budget axis, and uncertainty/comparison rules, while showing Macro-F1 at every locked budget. Report bootstrap and upstream representation-selection supervision separately from additional purchased labels.
- Retain measured negative speed/memory evidence transparently as context. Demoting those outcomes and changing the thesis hypothesis requires an explicit prospective amendment, not retroactive removal.
- A positive result is conditional: if only Adaptive-HDC wins, attribute the finding to that tested variant, not vanilla HDC or HDC universally. A null result is informative and is not evidence of equivalence.

This note checks method identity and claim scope. It neither certifies that trusted_test has never been used nor approves opening it; that requires the repository's release/provenance checks.
