# Research Methodology & Mathematical Formulation

## 1. Multi-Model Deep Ensemble

Ensembling reduces both epistemic uncertainty (due to lack of knowledge or model misspecification) and optimization noise. The 5 diverse architectures span:
- Convolutions with large kernels: **ConvNeXt-Tiny**
- Efficient compound inverted bottlenecks: **EfficientNetV2-B0**
- Self-supervised Vision Transformers: **DINOv2-Small (Linear)**
- Domain-adapted Vision Transformers: **DINOv2-Small (Fine-Tuned)**
- Language-image contrastive representations: **CLIP ViT-B/32 (MLP)**

For an input image $x$, each model $m \in \{1, \dots, M\}$ outputs an uncalibrated probability vector:
$$p^{(m)}(y = c \mid x) = \operatorname{softmax}(z^{(m)}_c)$$

The ensemble posterior probability is computed via weighted or uniform probability averaging:
$$p_{\text{ens}}(y = c \mid x) = \sum_{m=1}^M \tilde{w}_m p^{(m)}(y = c \mid x), \quad \sum_{m=1}^M \tilde{w}_m = 1$$

## 2. Uncertainty Quantification

Three complementary indicators govern confidence assessment:

### A. Predictive Shannon Entropy
Entropy measures the global dispersion across the 29 classes:
$$H(p_{\text{ens}}) = -\sum_{c=1}^{C} p_{\text{ens}, c} \ln(p_{\text{ens}, c} + \epsilon)$$
- For a deterministic prediction: $H = 0$ nats.
- For maximum uniform ambiguity over 29 classes: $H = \ln(29) \approx 3.367$ nats.
- Threshold: $\tau_{\text{entropy}} = 1.20$ nats.

### B. Top-2 Probability Margin
The margin captures decision boundary ambiguity:
$$\Delta p = p_{(1)} - p_{(2)}$$
where $p_{(1)}$ and $p_{(2)}$ are the first and second largest probabilities in $p_{\text{ens}}$. A narrow margin indicates that the ensemble is wavering between two pathologies.
- Threshold: $\tau_{\text{margin}} = 0.15$.

### C. Inter-Model Disagreement
Disagreement quantifies consensus divergence among the heterogeneous architectures:
$$D = \frac{1}{M} \sum_{m=1}^M \mathbf{1}\left(\operatorname{argmax}_c p^{(m)}_c \neq \operatorname{argmax}_c p_{\text{ens}, c}\right)$$
- Threshold: $\tau_{\text{disagree}} = 0.40$ (i.e. more than 40% of voting models disagree with the ensemble decision).

## 3. Calibrated Abstention Decision Engine

An image is classified as `ACCEPTED` if and only if all of the following conditions hold simultaneously:
1. $\text{Suitability}(x) = \text{True}$ (passed blur, exposure, resolution checks).
2. $M_{\text{active}} \ge 3$ (at least 3 models participated in the ensemble).
3. $\max_c p_{\text{ens}, c} \ge 0.70$.
4. $H(p_{\text{ens}}) \le 1.20$.
5. $\Delta p \ge 0.15$.
6. $D \le 0.40$.

Otherwise, the system emits an active abstention (`ABSTAINED`, `INVALID_IMAGE`, or `MODEL_UNAVAILABLE`), suppressing disease diagnosis to prevent misdiagnosis.

## 4. Evaluation and Calibration Metrics

- **Expected Calibration Error (ECE):**
  $$\text{ECE} = \sum_{b=1}^B \frac{|B_b|}{N} |\operatorname{acc}(B_b) - \operatorname{conf}(B_b)|$$
- **Brier Score:**
  $$\text{BS} = \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^{29} (p_{ic} - y_{ic})^2$$
- **Selective Accuracy:**
  $$\text{Acc}_{\text{selective}} = \frac{\sum_{i: \text{Accept}_i} \mathbf{1}(\hat{y}_i = y_i)}{\sum_i \mathbf{1}(\text{Accept}_i)}$$
- **Selective Risk:**
  $$\text{Risk}_{\text{selective}} = 1 - \text{Acc}_{\text{selective}}$$
