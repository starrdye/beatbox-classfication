# Report Writing Plan
## *Spectral Fingerprints: Classifying and Comparing Human Beatbox Sounds to Mechanical Percussion*

---

## Overview

**Target:** Final Report — 4,000–5,000 words (40% of grade)
**Assessment breakdown:** Consultation 20% | Literature Review 10% | Draft Essay 20% | **Final Report 40%** | Presentation 10%

**Primary sources to draw from:**
- `research_proposal.md` / `APPENDIX A.docx` — original proposal
- `phase1_report.md` — unsupervised clustering pipeline
- `phase2_report.md` — supervised classification results
- `plan.md` — challenges and solutions log
- `proposal.md` — supervisor feedback and next steps
- `peak_alignment_clustering.py` — Phase 1 source code
- `phase2_classification.py` — Phase 2 source code
- `phase2_results.csv` — per-sample prediction data
- All `phase2_*.png` output figures

**Style inspiration:**
- Interactive layer chart: [biboamy.github.io/instrument-demo](https://biboamy.github.io/instrument-demo/index.html) — section flow: Abstract → Dataset → Model → Result, with accompanying interactive audio/visual display
- Academic writing style: [IJCAI 2018 paper (Chou et al.)](https://www.ijcai.org/proceedings/2018/0463.pdf) — dense but readable, equations, figures with captions, section-by-section analysis

---

## Report Structure

### Title Page
- **Title:** Spectral Fingerprints: Classifying and Comparing Human Beatbox Sounds to Mechanical Percussion
- **Subtitle / Descriptor:** A Digital Signal Processing and Machine Learning Study of Vocal Percussion
- **Author, Supervisor, Date, Module/Course**
- **Word count**

---

### Abstract (~150 words)

Summarise:
1. The research gap: no quantitative comparison of beatbox sounds vs. their real-instrument counterparts
2. Dataset: 9 participants, 4 sound types (`b`, `k`, `psh`, `nu`), 86–88 recordings
3. Methods used: Phase 1 (unsupervised peak-alignment clustering), Phase 2 (SVM, Random Forest, CNN with LOPO-CV)
4. Key result: CNN achieves **94.2% LOPO accuracy** from 76.1% Phase 1 baseline
5. Core finding: Individual differences (not ML model choice) are the primary barrier to generalisation

---

### 1. Introduction — Why This Study Matters (~500 words)

**Content to cover:**
- What is beatbox? The vocal art of imitating percussion — kick `{b}`, hi-hat `{t/psh}`, snare `{K/k}`
- Situate the study in the **HCI and audio synthesis** context: if we can measure spectral distance between voice and machine, we can build better generative audio models
- Identify the **research gap**: most existing work asks "can a computer label this sound?" — this study asks "how spectrally similar is this vocal sound to the real instrument it imitates?"
- State the two research questions:
  - **RQ1:** Can machine learning reliably classify beatbox sounds across different individuals?
  - **RQ2:** What acoustic features are most discriminative, and where do beatbox sounds spectrally diverge from their instrument counterparts?
- Introduce the two-phase structure of the study
- State the significance: moves beyond classification accuracy toward a quantitative **Acoustic Realism Score**

**Writing guidance:** Open with a hook — describe the phenomenon of beatboxing and immediately pose the scientific puzzle. Connect to real-world applications (MIDI transcription, voice-to-drum synthesis, accessible music-making tools for people without instruments).

---

### 2. Related Work & Literature Review (~400 words)

**Papers to cite and contextualise (from `research_proposal.md` reading list):**

| Paper | What to say |
|---|---|
| Stowell & Plumbley (2008) | Foundational work; established the primary sound taxonomy ({b}, {t}, {K}); provides acoustic baselines this study extends |
| Sinyor & McKay (2005) | Early classification attempt using ACE features; motivates why richer features (MFCCs) are needed |
| Kapur et al. (2004) | Music retrieval application of beatbox recognition; shows real-world demand for robust classification |
| Proctor et al. (2013) | MRI study of vocal tract during beatboxing; shows vocal tract shape varies per sound — directly motivates formant-based features (MFCCs) |
| Martanto & Kartowisastro (2025) | Most recent comparable ML study; compare accuracy numbers directly to Phase 2 results |
| Chou et al. (IJCAI 2018) | Instrument recognition CNN; architecture inspires the Phase 2 CNN design |

**Identify the gap:** None of the above studies directly measures the *acoustic distance* between a beatbox sound and the real instrument it imitates. This study introduces that direction with the proposed Realism Score.

---

### 3. Dataset (~400 words)

**Content:**
- **Participants:** 9 amateur participants (IDs: 1, 2, 3, 4, 5, 7, 8, 10, 11)
- **Sound categories:**

| Label | Sound | Description |
|---|---|---|
| `b` | Bass / Vocal Kick | Low-frequency labial plosive — imitates kick drum |
| `k` | K-Snare / Click | Velar stop consonant snap — imitates snare/rim shot |
| `psh` | Push Hi-hat | Sustained fricative noise — imitates open/closed hi-hat |
| `nu` | Neutral Hum | Sustained resonant tone — tonal beatbox sound |

- **File naming convention:** `{participant}/Phase 1/{participant}-{sound}-{take}.wav`
- **Total samples:** 88 (Phase 1 clustering), 86 (Phase 2 — 2 samples excluded as too short for feature extraction)
- **Class distribution:** b=20, k=22, nu=20, psh=24

**Data challenges to briefly mention here (full detail in Appendix B):**
- Clips do not start at the same time — onset alignment was necessary
- Some clips contain extra trailing silence ("silency trails") after the sound event
- Recording conditions vary across participants; no studio environment

---

### 4. Methodology (~1,200 words)

Split into two clearly labelled phases.

#### 4.1 Phase 1 — Unsupervised Clustering (Peak-Alignment Method)
*Source: `peak_alignment_clustering.py`, `phase1_report.md`*

**Include a pipeline flow diagram (recreate the ascii block from `phase1_report.md` Section 2 as a clean figure):**
```
Raw Audio (.wav)
       │
[Feature Extraction]  — Spectral Centroid, RMS Energy, Noisiness (time-series)
       │
[Peak Alignment]      — All samples aligned to their energy peak
       │
[Overlap Distance Matrix] — Pairwise overlap % → distance = 100 − overlap
       │
[K-Means Clustering]  — 4 clusters mapped to sound types by majority vote
       │
[Results & Evaluation] — Accuracy, Adjusted Rand Index, per-sample table
```

**Sub-sections:**

**4.1.1 Feature Extraction**
Three time-series extracted per clip using `librosa`:
- **Spectral Centroid** — weighted average frequency (brightness proxy):
  $$C = \frac{\sum_n f_n \cdot |X(n)|}{\sum_n |X(n)|}$$
- **RMS Energy** — amplitude envelope per frame:
  $$E_{RMS}(t) = \sqrt{\frac{1}{N}\sum_{n=0}^{N-1} y(n)^2}$$
- **Noisiness** — HPSS-derived percussive energy ratio:
  $$\text{Noisiness}(t) = \frac{\text{RMS}(y_{\text{perc}})(t)}{\text{RMS}(y)(t) + \epsilon}$$

**4.1.2 Peak Alignment**
- **Motivation:** Raw clips have different time offsets — comparing un-aligned time-series measures timing offset, not acoustic character
- **Method:** Anchor all signals at the energy peak frame index; extract fixed window (50 pre-peak frames + 100 post-peak frames = 151-frame aligned vector per feature)
- **Limitation:** Single-peak assumption fails for `nu` (sustained, no sharp peak) and some `b` samples (breath burst precedes the plosive)

**4.1.3 Overlap Distance**
IoU-inspired area-of-intersection over area-of-union metric applied to 1D time-series:
$$\text{Overlap}(s_1, s_2) = \frac{\sum_t \min(s_1(t), s_2(t))}{\sum_t \max(s_1(t), s_2(t))} \times 100$$
Distance = $100 - \text{mean overlap}$ across 3 features → 88×88 pairwise distance matrix.

**4.1.4 K-Means Clustering**
- 4 clusters (`n_clusters=4`), mapped to sound types by majority vote
- Geometric mismatch: K-means minimises within-cluster Euclidean variance but input here is a pairwise distance matrix — spectral clustering or DBSCAN would be more appropriate

#### 4.2 Phase 2 — Supervised Classification
*Source: `phase2_classification.py`, `phase2_report.md`*

**4.2.1 Feature Engineering — 38-Dimensional Feature Vector**

Present the full feature table (from `phase2_report.md` Section 2.3):

| # | Feature | Dims | Key discrimination |
|---|---|---|---|
| 1–13 | MFCC Mean | 13 | Spectral envelope / vocal tract formants |
| 14–26 | MFCC Std | 13 | Temporal stability of timbre |
| 27–28 | Centroid Mean + Std | 2 | Overall brightness |
| 29–30 | Rolloff Mean + Std | 2 | Upper spectral energy boundary |
| 31–32 | Flux Mean + Std | 2 | Rate of spectral change (attack) |
| 33–34 | ZCR Mean + Std | 2 | Noisiness proxy |
| 35–36 | RMS Mean + Std | 2 | Loudness and amplitude envelope |
| 37 | Attack Time | 1 | Onset-to-peak duration |
| 38 | Decay Rate | 1 | Post-peak energy slope |
| | **Total** | **38** | |

Explain *why MFCCs are the critical addition*:
> The `b` bass kick produces low-frequency chest resonance (low formants); the `k` click involves a velar stop (different formant pattern). MFCCs encode formant information directly (especially MFCC 2–4). This is precisely what Phase 1's three features could not capture — explaining the `b` accuracy jump from 38.1% → 90.0%.

**4.2.2 Log Mel-Spectrogram (CNN Input)**
- 2D time-frequency image: 64 mel bins × 128 time frames
- Pipeline: STFT → Mel filterbank (64 triangular filters) → log(power) → zero-pad/truncate to 128 frames
- At hop=512 and sr=22,050 Hz, 128 frames ≈ 3 seconds — well above any single beatbox sound duration
- Explain how to read: x-axis=time, y-axis=mel frequency, intensity=dB energy

**4.2.3 Model 1 — SVM (RBF Kernel)**
- Why SVM for n=86: maximises margin in high-dimensional space; RBF kernel handles non-linear boundaries
- Input scaled with `StandardScaler` (zero mean, unit variance) — required because SVM is distance-based
- RBF kernel: $K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)$
- Hyperparameter tuning via `GridSearchCV`: C ∈ {0.1, 1, 10, 100}, gamma ∈ {scale, auto, 0.001, 0.01}

**4.2.4 Model 2 — Random Forest**
- Ensemble of 200 decision trees; each trained on a bootstrap sample with random feature subsets (√38 ≈ 6 features per split)
- No scaling needed — decision trees are not distance-based
- Key advantage: **feature importance scores** — principled ranking of which descriptors are most discriminative

**4.2.5 Model 3 — CNN on Mel-Spectrograms**
- Architecture (3 convolutional blocks → classifier head):
  ```
  Input: (1 × 64 × 128)
  Block 1: Conv2D(1→16, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → (16×32×64)
  Block 2: Conv2D(16→32, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → (32×16×32)
  Block 3: Conv2D(32→64, 3×3) → BatchNorm → ReLU → AdaptiveAvgPool(4×4) → (64×4×4)
  Flatten → Dense(128) → ReLU → Dropout(0.5) → Dense(4) → Softmax
  ```
- Data augmentation (critical for n=86): time shift ±12 frames, frequency masking (10 bins), time masking (16 frames), amplitude jitter ±15%, per-sample normalisation
- Training: Adam optimizer, L2 weight decay 1e-4, cosine annealing LR schedule, 80 epochs per LOPO fold

**4.2.6 Validation — Leave-One-Participant-Out (LOPO-CV)**
- Why not random 80/20 split: samples from the same participant in both train and test sets would leak individual-specific information, inflating accuracy
- LOPO trains on 8 participants, tests on the held-out 9th — genuine generalisation to an unseen individual
- 9 folds; each fold: ~77 train / ~9 test samples
- This is the scientifically valid question for HCI applications: does the model generalise to a new user?

---

### 5. Libraries and Tools (~150 words)

| Library | Use |
|---|---|
| `librosa` 0.10.x | Audio loading, MFCC, spectral centroid, RMS, spectral flux, ZCR, onset detection, HPSS, mel-spectrogram |
| `scikit-learn` 1.x | SVM (SVC), Random Forest, StandardScaler, GridSearchCV, confusion_matrix, accuracy_score |
| `PyTorch` 2.x | CNN definition, Dataset/DataLoader, Adam optimizer, CosineAnnealingLR, CrossEntropyLoss |
| `numpy` | Array operations, distance matrix construction, feature aggregation |
| `matplotlib` / `seaborn` | Confusion matrix heatmaps, bar charts, feature importance plots |
| `pandas` | Results CSV export and tabular analysis |

**References:**
- McFee et al. (2015) — librosa: Audio and Music Signal Analysis in Python. *SciPy 2015*.
- Pedregosa et al. (2011) — Scikit-learn: Machine Learning in Python. *JMLR* 12, 2825–2830.
- Paszke et al. (2019) — PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS 2019*.

---

### 6. Results (~700 words)

#### 6.1 Phase 1 Results

| Sound | Correct | Total | Accuracy |
|---|---|---|---|
| `nu` | 18 | 20 | 90.0% |
| `psh` | 21 | 24 | 87.5% |
| `k` | 19 | 22 | 86.4% |
| **`b`** | **8** | **21** | **38.1% ← major problem** |
| **Overall** | **67** | **88** | **76.1%** |

Key finding: `b` is acoustically ambiguous — two different production strategies (deep chest resonance vs. consonant-heavy forward `b`) split the sound into separate clusters. Phase 1's three features cannot capture the formant-level distinction between `b` and `k`.

Error patterns:
- Pattern A: `b` misclassified as `k` — Participants 1, 2, 5, 7, 8 (12 cases)
- Pattern B: Participant 11 complete cluster failure (6 cases)
- Pattern C: Participant 3's `psh` misclassified as `k` (3 cases)

#### 6.2 Phase 2 Results (main accuracy table)

| Model | Overall | `b` | `k` | `nu` | `psh` |
|---|---|---|---|---|---|
| Phase 1 — K-Means | 76.1% | 38.1% | 86.4% | 90.0% | 87.5% |
| Phase 2 — SVM | 82.6% | **90.0%** | 59.1% | **100.0%** | 83.3% |
| Phase 2 — RF | 83.7% | **90.0%** | 72.7% | 90.0% | 83.3% |
| **Phase 2 — CNN** | **94.2%** | **90.0%** | **95.5%** | **100.0%** | **91.7%** |

**Key observations to write up:**

1. **MFCCs fix `b`:** From 38.1% → 90.0% across all three Phase 2 models — confirms that formant structure captured by MFCCs is what distinguishes `b` from `k`
2. **The `b`/`k` trade-off in SVM:** MFCCs bring `b` and `k` closer in MFCC feature space than they were in Phase 1. The SVM's hyperplane — drawing a smooth boundary — drops `k` accuracy to 59.1%
3. **CNN resolves the trade-off:** Full mel-spectrogram preserves fine-grained spatial patterns that simultaneously separate `b` and `k` (95.5% for `k`, 90.0% for `b`)
4. **`nu` is trivially classifiable:** 100% accuracy in both SVM and CNN — sustained hum is spectrally completely distinct from all three percussive sounds
5. **Individual differences are the primary bottleneck:** 4/5 CNN errors come from Participant 5 alone

#### 6.3 Feature Importance (Random Forest)
- Embed: `phase2_feature_importance.png`
- MFCCs dominate top rankings (especially MFCC 1–5: broad spectral envelope and low-frequency formants)
- Attack Time and Decay Rate rank in top 10 — temporal envelope shape is genuinely discriminative
- Spectral Rolloff and Centroid rank lower once MFCCs are available (they partially duplicate MFCC 1–2 information)

#### 6.4 Per-Participant Analysis (LOPO)
- Embed: `phase2_lopo_per_participant.png`
- **Participant 11:** Complete failure in Phase 1 (6/6 errors), partially fixed in Phase 2 (SVM and RF still fail; CNN succeeds on all 10 samples)
- **Participant 5:** New failure mode for CNN — 4/9 samples misclassified (all as `k`); suggests a genuine production-style outlier
- Conclusion: Individual vocal style can dominate over sound-type acoustic signal

#### 6.5 Confusion Matrices
- Embed: `phase2_confusion_svm.png`, `phase2_confusion_rf.png`, `phase2_confusion_cnn.png`
- Walk through notable off-diagonal cells acoustically (e.g., SVM's `k`→`psh` confusion for Participant 1)

#### 6.6 Per-Sound Comparison Across Models
- Embed: `phase2_per_sound_comparison.png`
- Highlight the non-monotonic `k` accuracy: drops 86.4% → 59.1% (SVM), recovers to 72.7% (RF), jumps to 95.5% (CNN)

---

### 7. Discussion (~500 words)

**Content:**
- **Why does representation matter more than model choice?** The jump from 3 time-series features → 38-dim MFCC vector lifts `b` accuracy by 51.9 pp. The jump from the feature vector → mel-spectrogram then lifts `k` by 22.8 pp (RF → CNN). Input feature choice is the single most important decision.
- **What does "individual differences are the primary bottleneck" mean scientifically?** The classifier learns sound-type patterns from 8 participants and must generalise to the 9th. When that 9th participant's vocal tract shape and articulation style is systematically different (Participant 5, 11), even a strong model fails. This has direct implications for any commercial beatbox-to-MIDI application.
- **Implications for the Realism Score:** Phase 2 proves we can reliably classify beatbox sounds (94.2%). The next step — comparing beatbox MFCC feature centroids to actual instrument samples via cosine similarity — is now technically feasible and is the natural Phase 3.
- **Limitations:** Small dataset (86 samples, 9 participants), single recording session per participant, no professional beatboxers, no actual instrument ground-truth yet collected. The `b`/`k` acoustic ambiguity is not fully resolved even by the CNN.

---

### 8. Conclusion (~200 words)

- Restate the two research questions and answer them directly
- Summarise the progression: 76.1% → 94.2% through better features and model choice
- State the key scientific finding: individual variation > model architecture as the primary barrier to perfect classification
- Recommend next steps:
  1. Expand dataset with more participants and professional beatboxers
  2. Collect actual instrument samples (acoustic kick, hi-hat, snare; TR-808)
  3. Compute the Acoustic Realism Score (cosine similarity in MFCC space)
  4. Investigate Participant 5's production technique specifically

---

### References

Full bibliography. Include at minimum:

1. Martanto, J., & Kartowisastro, I. H. (2025). Beatbox Classification to Distinguish User Experiences Using Machine Learning Approaches. *Journal of Computer Science*, 21(4), 961–970.
2. Stowell, D., & Plumbley, M. D. (2008). Characteristics of the beatboxing vocal style. Technical Report C4DM-TR-08-01, Queen Mary University of London.
3. Sinyor, E., & McKay, C. (2005). Beatbox classification using ACE. *Proceedings of ISMIR 2005*.
4. Kapur, A., Benning, M., & Tzanetakis, G. (2004). Query-by-Beat-Boxing: Music Retrieval for the DJ. *Proceedings of ISMIR 2004*.
5. Proctor, M., et al. (2013). Paralinguistic mechanisms of production in human beatboxing: A real-time MRI study. *Journal of the Acoustical Society of America*, 133(2).
6. McFee, B., et al. (2015). librosa: Audio and Music Signal Analysis in Python. *Proceedings of SciPy 2015*, 18–25.
7. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
8. Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS 2019*.
9. Peeters, G., et al. (2011). A large set of audio features for sound description (similarity and classification) in the CUIDADO project. Technical Report, IRCAM.
10. Chou, S-Y., et al. (2018). Learning-Based Automatic Audio Segmentation. *IJCAI 2018*, Proc. 0463.
11. MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*.

---

### Appendix A — Research Proposal
*(Reproduce content from `APPENDIX A.docx` / `research_proposal.md` verbatim — the original ISC proposal with title, outline, reading list, schedule, contact hours, and assessment breakdown)*

---

### Appendix B — Challenges Encountered

Write in a reflective, candid tone. This section is honest about what failed and why.

**B.1 Audio Clips Do Not Start at the Same Time**
- Raw recordings have variable offsets before the sound event begins
- Some clips contain extra leading silence or ambience; clips do not share a consistent onset position
- **Impact:** Naive frame-by-frame comparison measures timing offset, not acoustic character
- **Solution:** Peak alignment — anchor all time-series at the energy peak frame index, extract a fixed window around it (`align_to_peak()` in `peak_alignment_clustering.py`)
- **Residual limitation:** Single-peak assumption fails for sustained sounds (`nu`) and `b` samples where a breath burst precedes the plosive

**B.2 Extra Trailing Silence ("Silency Trails")**
- Some clips have extended silence after the sound event (participants held the recording button too long)
- Trailing silence inflates RMS mean, pads noisiness series with zeros, distorts decay rate calculation
- **Attempted mitigation:** Onset detection with `librosa.onset.onset_detect` to anchor the feature window; post-peak window of 100 frames limits how much silence is captured
- **Ongoing issue:** Some trailing silence still enters the fixed 100-post-peak window

**B.3 Supervised Clustering Accuracy Too Low (Phase 1 → Phase 2 decision)**
- Initial unsupervised approach achieved only 76.1% overall, with `b` at 38.1%
- Root cause analysis identified three compounding factors:
  1. Three features (centroid, energy, noisiness) insufficient to distinguish `b` from `k` at formant level
  2. K-means geometrically mismatched to a pairwise distance matrix
  3. Overlap percentage metric sensitive to amplitude scale — same-shaped but different-volume sounds get low overlap
- **Decision:** Transition to supervised classification (Phase 2) with MFCC-based feature vector

**B.4 Individual Differences Dominate Acoustic Signal**
- Participant 11 (Phase 1): all `b` and `k` sounds cluster as `psh` — complete participant-level failure (6/6 errors)
- Participant 5 (Phase 2 CNN): 4/9 samples misclassified — systematic production style outlier
- **Impact:** No single trained model generalises perfectly across all individuals
- **Response:** LOPO cross-validation explicitly quantifies this; confirmed as the primary scientific barrier

**B.5 Small Dataset for Deep Learning**
- 86 total samples is very small for a CNN — overfitting occurred in early experiments without augmentation
- **Solution:** Five augmentation strategies applied during training only: time shift ±12 frames, frequency masking, time masking, amplitude jitter ±15%, per-sample normalisation
- **Limitation:** Augmentation helps but cannot substitute for more diverse real recordings

**B.6 Librosa Frame Size Warnings**
- Short audio clips triggered runtime warnings when processed with default frame size n_fft=2048 (larger than the signal length)
- **Solution:** Dynamic frame size adjustment: `frame = max(256, 2 ** int(np.log2(min(N_FFT, len(y)))))`

**B.7 Subjective Sound Boundary**
- Beatbox sounds are performed differently across individuals; the `b` category in particular contains two distinct acoustic production strategies ("deep chest resonance" vs. "consonant-forward `b`")
- Some participants' `b` samples are acoustically closer to `k` than to other participants' `b` sounds
- **Implication:** The 38.1% Phase 1 `b` accuracy partially reflects genuine acoustic ambiguity in the category, not only model weakness. The solution (MFCCs in Phase 2) proves the ambiguity is resolvable with richer features.

---

## Interactive Layer Chart Specification

**Inspired by:** [biboamy.github.io/instrument-demo](https://biboamy.github.io/instrument-demo/index.html)
**Also reference:** [IJCAI 2018 paper](https://www.ijcai.org/proceedings/2018/0463.pdf) — layer diagrams with explanatory annotations

**What to build:** A supplementary HTML/JS page showing the CNN architecture as an interactive diagram.

**Architecture flow to visualise:**

```
[Input: 64×128 Log Mel-Spectrogram]
                 ↓
[Conv Block 1: Conv2D(1→16, 3×3) → BatchNorm → ReLU → MaxPool(2×2)]
  Output shape: 16 × 32 × 64   |   Learns: Basic edges and frequency bands
                 ↓
[Conv Block 2: Conv2D(16→32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)]
  Output shape: 32 × 16 × 32   |   Learns: Combined spectral-temporal patterns
                 ↓
[Conv Block 3: Conv2D(32→64, 3×3) → BatchNorm → ReLU → AdaptiveAvgPool(4×4)]
  Output shape: 64 × 4 × 4 = 1,024 values   |   Learns: High-level sound signatures
                 ↓
[Flatten → Dense(128) → ReLU → Dropout(0.5)]
                 ↓
[Output: Dense(4) → Softmax → {b, k, nu, psh}]
```

**Interactive features to implement:**
- Hover over each layer block → tooltip shows tensor shape + one-sentence explanation of what the layer does
- Click a sound label at the output → highlight path / show which mel-spectrogram patterns activate that class
- Display 4 example mel-spectrogram images (one per sound type) at the input node
- Optional: click-to-play audio samples alongside each mel-spectrogram

**Embed in report as:**
A link or screenshot with caption: *"Interactive CNN architecture diagram — hover over each layer for tensor shapes and descriptions."*

---

## Figures to Include in Report

| Figure | File | Caption |
|---|---|---|
| Phase 1 pipeline diagram | (recreate from ascii in phase1_report.md) | "Phase 1 unsupervised clustering pipeline" |
| RMS comparison across sounds | `rms_comparison.png` | "Peak-aligned RMS energy envelopes by sound type across all participants" |
| RMS heatmap | `rms_heatmap.png` | "Per-participant RMS energy heatmap — colour = amplitude" |
| Feature importance chart | `phase2_feature_importance.png` | "Top 20 Random Forest feature importances (full dataset, 500 trees)" |
| LOPO per-participant accuracy | `phase2_lopo_per_participant.png` | "Per-participant LOPO-CV accuracy by model with Phase 1 baseline" |
| Per-sound comparison bar chart | `phase2_per_sound_comparison.png` | "Per-sound accuracy: Phase 1 vs Phase 2 models" |
| SVM confusion matrix | `phase2_confusion_svm.png` | "Confusion matrix — SVM (RBF kernel), LOPO-CV, 82.6% overall" |
| RF confusion matrix | `phase2_confusion_rf.png` | "Confusion matrix — Random Forest (200 trees), LOPO-CV, 83.7% overall" |
| CNN confusion matrix | `phase2_confusion_cnn.png` | "Confusion matrix — CNN on Mel-Spectrograms, LOPO-CV, 94.2% overall" |
| CNN layer diagram | (build as interactive chart) | "CNN architecture with per-layer tensor shapes and explanations" |
| Example mel-spectrograms | (generate from audio_data) | "Representative log mel-spectrogram for each of the 4 sound types" |

---

## Writing Tone & Style Notes

- **Academic but accessible** — explain all technical terms on first use (e.g., "MFCCs (Mel-Frequency Cepstral Coefficients) are…")
- **Equations should be numbered** and referenced in text (e.g., "…computed using Equation (3)")
- **Captions for all figures** — each caption should describe what the reader should specifically notice, not just what is shown
- **LOPO validation deserves emphasis** — present as a point of methodological strength: most papers use random splits; this study's LOPO is more rigorous and more realistic
- **Do not hide the challenges** — the appendix and discussion should honestly address failures; this demonstrates scientific maturity and critical thinking
- **Voice:** Prefer active for methodology ("We extracted…", "The model learns…") and passive for results ("An accuracy of 94.2% was achieved…")
- **The `b`/`k` trade-off story** is the most interesting analytical thread — follow it across Phase 1 → Phase 2 SVM → Phase 2 CNN as a narrative arc through the results

---

## Suggested Writing Order

1. **Section 3 (Dataset)** + **4.1 (Phase 1 methodology)** — most grounded in concrete, finished work
2. **Section 4.2 (Phase 2 methodology)** — most detailed technical section; reuse the feature tables from `phase2_report.md`
3. **Section 6 (Results)** — embed all figures, fill in the accuracy tables, walk through error patterns
4. **Section 5 (Libraries)** — brief enumeration, straightforward
5. **Section 7 (Discussion)** — synthesise findings across both phases; address the individual-differences question
6. **Section 2 (Related Work)** — contextualise each paper relative to your findings
7. **Section 1 (Introduction) + Abstract** — write last, once the full story is clear
8. **Appendix B (Challenges)** — draw from `plan.md` and personal notes; write candidly
9. **Appendix A** — reproduce proposal material from `APPENDIX A.docx`

---

## Word Count Guide

| Section | Target Words |
|---|---|
| Abstract | ~150 |
| 1. Introduction | ~500 |
| 2. Related Work | ~400 |
| 3. Dataset | ~400 |
| 4. Methodology | ~1,200 |
| 5. Libraries | ~150 |
| 6. Results | ~700 |
| 7. Discussion | ~500 |
| 8. Conclusion | ~200 |
| **Total (main body)** | **~4,200** |

> Appendices are typically excluded from the word count — confirm with your supervisor.
