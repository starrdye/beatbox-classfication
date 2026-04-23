# Beatbox Classification Project

This project focuses on the classification and analysis of beatbox sounds compared to traditional drum hits.

## Project Structure

The codebase is organized into several phases of analysis, each represented by corresponding Python scripts and generated reports.

## Audio Data Structure

The `audio_data` directory is the central location for all audio files used in this project. It is structured to support both participant recordings and extracted drum hits for comparison.

```mermaid
graph TD
    Root[audio_data/] --> Participants[Participant Directories 1-11/]
    Root --> Drums[drums/]
    
    subgraph Participants_Folder [Participant Structure]
        P1[1/] --> Ph1[Phase 1/]
        P1 --> Ph2[Phase 2/]
        P1 --> Ph3[Phase 3/]
        Ph1 --> W1[Individual Sound Clips e.g., P1-b-01.wav]
    end
    
    subgraph Drums_Folder [Drum Hits Structure]
        Drums --> BD[bd/ Bass Drum]
        Drums --> CHH[chh/ Closed Hi-Hat]
        Drums --> SD[sd/ Snare Drum]
        BD --> DW1[Extracted WAV files]
    end
```

### Participant Data
- **Phase 1**: Contains individual sound clips organized by sound class (e.g., `b` for bass kick, `k` for snare).
- **Phase 2**: Contains patterns and full recordings used for supervised classification.
- **Phase 3**: Used for realism analysis and comparison with professional sounds.

### Drum Data
- **drums/**: Contains professional drum hits extracted from the ENST dataset, categorized into `bd` (bass drum), `chh` (closed hi-hat), and `sd` (snare drum).

---

<div align="center">

## 📊 Project Findings and AI Layer Structure Poster

### Model Performance Findings

The Convolutional Neural Network (CNN) demonstrated a significant accuracy improvement over traditional machine learning and unsupervised clustering.

```mermaid
xychart-beta
    title "Overall Classification Accuracy by Model (%)"
    x-axis ["Phase 1 K-Means", "Phase 2 SVM", "Phase 2 Random Forest", "Phase 2 CNN"]
    y-axis "Accuracy %" 0 --> 100
    bar [76.1, 82.6, 83.7, 94.2]
```

<br>

<p align="center">
  <img src="public/phase2_confusion_cnn.png" width="45%" title="CNN Confusion Matrix">
  &nbsp;&nbsp;
  <img src="public/phase2_per_sound_comparison.png" width="45%" title="Per-Sound Comparison">
</p>

<p align="center">
  <img src="public/phase2_feature_importance.png" width="45%" title="Random Forest Feature Importance">
  &nbsp;&nbsp;
  <img src="public/phase2_lopo_per_participant.png" width="45%" title="Per-Participant LOPO Accuracy">
</p>

### CNN Architecture Structure

The best-performing model extracts Mel-Spectrogram features through three connected spatial convolution blocks.

```mermaid
graph TD
    A["Log Mel-Spectrogram Image (1x64x128)"] --> B{"Conv2D Block 1"}
    B --> C{"Conv2D Block 2"}
    C --> D{"Conv2D Block 3"}
    D --> E("Classifier Head<br>Flatten & Dense layers")
    E --> F(("Output: 4 Sound Classes<br>Softmax Projection"))
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style F fill:#bbf,stroke:#333,stroke-width:2px
```

</div>

---

## Getting Started

1. **Audio Data**: Ensure the `audio_data/` directory is populated with the necessary recordings.
2. **Drum Hits**: If needed, run `extract_drum_hits.py` to populate `audio_data/drums` from the `ENST-drums-dataset-master` (ensure the raw dataset is present locally).
3. **Classification**: Run `phase2_classification.py` to perform supervised classification using SVM, Random Forest, and CNN models.
4. **Analysis**: Use `rms_analysis.py` and `phase3_realism_analysis.py` for further signal processing and comparative studies.
