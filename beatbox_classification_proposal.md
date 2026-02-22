# Beatbox Classification Test Proposal

## Overview

This proposal outlines a method for performing beatbox classification tests based on the SRIP Experimental Procedure documents and the provided audio data. The approach leverages audio feature extraction techniques similar to those used in the existing `sound_analysis.py` file, adapted for beatbox sound classification.

## Audio Data Structure

The audio data is organized in a hierarchical structure:

```
audio_data/
  ├── [participant_id]/
  │   ├── Phase 1/
  │   │   ├── [sound_type]-[instance].wav
  │   │   └── ...
  │   ├── Phase 2/
  │   │   ├── [pattern].wav
  │   │   └── ...
  │   └── Phase 3/
  │       ├── [sound_type]-imitate-[instance].wav
  │       └── ...
  └── ...
```

Where:
- `[participant_id]` is a numerical identifier for each participant (1, 2, 3, ...)
- `[sound_type]` is a code for the beatbox sound (b, k, tsp, tss, etc.)
- `[instance]` is a numerical identifier for each recording of the same sound

## Phase Definitions

### Phase 1: Original Beatbox Sounds

**Purpose**: This phase contains the original, isolated beatbox sounds performed by each participant.

**Content**: 
- Individual beatbox sounds (e.g., "b", "k", "tsp", "tss")
- Multiple instances of each sound type for consistency analysis
- Sounds are typically isolated and performed in a controlled manner

**Usage in Classification**: 
- Primary training dataset for the classification model
- Used to establish baseline features for each sound type
- Provides ground truth labels for model training

### Phase 2: Beatbox Patterns

**Purpose**: This phase contains sequences or patterns of beatbox sounds performed by each participant.

**Content**: 
- Combined beatbox patterns (e.g., "PatternA", "PatternB")
- Sequences of multiple beatbox sounds performed in succession
- More complex sound structures than Phase 1

**Usage in Classification**: 
- Optional for advanced classification tasks
- Can be used to test pattern recognition capabilities
- Provides context for sound transitions and combinations

### Phase 3: Imitation Attempts

**Purpose**: This phase contains recordings of participants attempting to imitate specific beatbox sounds.

**Content**: 
- Imitation attempts of specific beatbox sounds (e.g., "b-imitate", "k-imitate")
- Multiple instances of each imitation attempt
- May contain variations in accuracy and style

**Usage in Classification**: 
- Primary test dataset for evaluating classification performance
- Used to assess how well the model can identify sounds across different performers
- Provides insights into imitation accuracy and sound perception

## Proposed Input Structure

### Input Format for Audio Analysis

To ensure consistent and accurate feature extraction, we propose the following input structure:

```python
# Input structure for audio analysis
audio_input = {
    "file_path": "path/to/audio/file.wav",
    "time_indices": {
        "start": 0.0,  # Start time in seconds
        "end": None    # End time in seconds (None for full duration)
    },
    "metadata": {
        "participant_id": "1",
        "phase": "1",  # 1, 2, or 3
        "sound_type": "b",  # Sound identifier
        "instance": "1"  # Recording instance
    }
}
```

### Time Indices Specification

**For Phase 1 (Isolated Sounds):**
- Default: Analyze the entire file duration
- Recommended: Use `start=0.0` and `end=None`
- Rationale: These files typically contain only the target sound with minimal silence

**For Phase 2 (Patterns):**
- Optional: Specify time indices to isolate individual sounds within patterns
- Example: `start=1.2` and `end=1.8` to extract a specific sound from a pattern
- Rationale: Allows for analysis of individual sounds within complex patterns

**For Phase 3 (Imitation Attempts):**
- Default: Analyze the entire file duration
- Recommended: Use `start=0.0` and `end=None`
- Rationale: These files typically contain only the imitation attempt with minimal silence

### Implementation of Time Indices

To incorporate time indices into the feature extraction process, we can modify the `extract_features` function as follows:

```python
def extract_features(file_path, start_time=0.0, end_time=None):
    """
    Extracts the 4 Peeters feature families from an audio file
    
    Args:
        file_path (str): Path to the .wav file
        start_time (float): Start time in seconds
        end_time (float): End time in seconds (None for full duration)
        
    Returns:
        dict: Extracted features
    """
    try:
        y, sr = librosa.load(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    
    # Apply time indices if specified
    if start_time > 0.0 or end_time is not None:
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr) if end_time is not None else len(y)
        y = y[start_sample:end_sample]
    
    # Extract time-series vectors
    frame_size = 2048
    hop_length = 512
    
    # Spectral Centroid
    centroid_series = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_size, hop_length=hop_length)[0]
    
    # RMS Energy
    rms_series = librosa.feature.rms(y=y, frame_length=frame_size, hop_length=hop_length)[0]
    
    # Harmonic-Percussive Separation (For Noisiness Calculation)
    y_harm, y_perc = librosa.effects.hpss(y)
    noise_rms_series = librosa.feature.rms(y=y_perc, frame_length=frame_size, hop_length=hop_length)[0]
    
    # Calculate the 4 feature families
    features = {
        "spectral_center": np.median(centroid_series),
        "spectral_variability": np.subtract(*np.percentile(centroid_series, [75, 25])),  # IQR
        "temporal_energy": np.median(rms_series),
        "periodicity": np.median(noise_rms_series / (rms_series + 1e-6))
    }
    
    return features
```

### Time Index Detection (Optional Enhancement)

For Phase 2 files containing multiple sounds, we can implement automatic sound segmentation:

```python
def detect_sound_boundaries(file_path, threshold=0.01):
    """
    Detects sound boundaries in an audio file based on energy levels
    
    Args:
        file_path (str): Path to the .wav file
        threshold (float): Energy threshold for sound detection
        
    Returns:
        list: List of (start, end) time tuples for detected sounds
    """
    y, sr = librosa.load(file_path)
    
    # Extract RMS energy
    frame_size = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_size, hop_length=hop_length)[0]
    
    # Convert to time domain
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    
    # Detect sound boundaries
    sound_boundaries = []
    in_sound = False
    start_time = 0.0
    
    for i, energy in enumerate(rms):
        if energy > threshold and not in_sound:
            # Start of sound
            start_time = times[i]
            in_sound = True
        elif energy <= threshold and in_sound:
            # End of sound
            end_time = times[i]
            sound_boundaries.append((start_time, end_time))
            in_sound = False
    
    # Handle sound at the end of file
    if in_sound:
        end_time = times[-1]
        sound_boundaries.append((start_time, end_time))
    
    return sound_boundaries
```

This enhancement would allow for automatic segmentation of Phase 2 pattern files into individual sounds for separate analysis.

## Classification Approach

### 1. Feature Extraction

We'll extract the following features from each audio file, based on the Peeters et al. (2011) feature families:

1. **Spectral Center (Median)** - Measures the "brightness" of the sound
2. **Spectral Variability (IQR)** - Measures the stability/consistency of the sound
3. **Temporal Energy (Median)** - Measures the "volume" or energy of the sound
4. **Periodicity Noisiness (Median)** - Measures the noisiness ratio of the sound

These features have been shown to be truly independent and effective for sound classification tasks.

### 2. Model Training

We'll use the Phase 1 data as our training set, since it contains multiple instances of each sound type for each participant. The training process will involve:

1. Extracting features from all Phase 1 audio files
2. Labeling each feature set with its corresponding sound type
3. Training a classification model on the labeled feature data

### 3. Model Testing

We'll test the trained model on two different datasets:

1. **Intra-participant testing**: Using Phase 1 data from participants not included in the training set
2. **Imitation testing**: Using Phase 3 data, which contains participants' attempts to imitate various beatbox sounds

### 4. Evaluation Metrics

We'll evaluate the classification performance using the following metrics:

1. **Accuracy** - Overall percentage of correct classifications
2. **Precision** - Percentage of positive predictions that are correct
3. **Recall** - Percentage of actual positives that are correctly identified
4. **F1 Score** - Harmonic mean of precision and recall
5. **Confusion Matrix** - Visual representation of classification performance across all sound types

## Implementation Plan

### Step 1: Data Preparation

1. **Audit the dataset** - Verify the completeness of the audio data
2. **Organize metadata** - Create a structured representation of the audio files and their labels
3. **Split the data** - Divide the Phase 1 data into training, validation, and test sets

### Step 2: Feature Extraction

1. **Extend `sound_analysis.py`** - Modify the existing code to process multiple files and output structured feature data
2. **Extract features** - Process all audio files to extract the four feature families
3. **Normalize features** - Scale features to a consistent range for better model performance

### Step 3: Model Development

1. **Select models** - Test multiple classification algorithms (e.g., SVM, Random Forest, KNN)
2. **Train models** - Train each model on the training dataset
3. **Tune hyperparameters** - Optimize model parameters using the validation dataset
4. **Evaluate models** - Compare model performance on the test dataset

### Step 4: Testing and Analysis

1. **Test on intra-participant data** - Evaluate model performance on unseen participants
2. **Test on imitation data** - Evaluate model performance on Phase 3 imitation data
3. **Analyze results** - Identify patterns in misclassifications and areas for improvement
4. **Generate reports** - Create visualizations of classification performance

## Proposed Code Structure

```python
# beatbox_classification.py

import librosa
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Feature extraction function (adapted from sound_analysis.py)
def extract_features(file_path):
    """
    Extracts the 4 Peeters feature families from an audio file
    """
    try:
        y, sr = librosa.load(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    
    # Extract time-series vectors
    frame_size = 2048
    hop_length = 512
    
    # Spectral Centroid
    centroid_series = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_size, hop_length=hop_length)[0]
    
    # RMS Energy
    rms_series = librosa.feature.rms(y=y, frame_length=frame_size, hop_length=hop_length)[0]
    
    # Harmonic-Percussive Separation (For Noisiness Calculation)
    y_harm, y_perc = librosa.effects.hpss(y)
    noise_rms_series = librosa.feature.rms(y=y_perc, frame_length=frame_size, hop_length=hop_length)[0]
    
    # Calculate the 4 feature families
    features = {
        "spectral_center": np.median(centroid_series),
        "spectral_variability": np.subtract(*np.percentile(centroid_series, [75, 25])),  # IQR
        "temporal_energy": np.median(rms_series),
        "periodicity": np.median(noise_rms_series / (rms_series + 1e-6))
    }
    
    return features

# Data loading function
def load_data(audio_dir):
    """
    Loads audio files and extracts features
    """
    data = []
    labels = []
    
    # Traverse the directory structure
    for participant in os.listdir(audio_dir):
        participant_path = os.path.join(audio_dir, participant)
        if not os.path.isdir(participant_path):
            continue
        
        # Process Phase 1 data (original sounds)
        phase1_path = os.path.join(participant_path, "Phase 1")
        if os.path.exists(phase1_path):
            for file in os.listdir(phase1_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(phase1_path, file)
                    
                    # Extract label from filename (e.g., "1-b-1.wav" → "b")
                    label = file.split("-")[1] if len(file.split("-")) > 1 else "unknown"
                    
                    # Extract features
                    features = extract_features(file_path)
                    if features:
                        data.append(list(features.values()))
                        labels.append(label)
    
    return np.array(data), np.array(labels)

# Main classification function
def classify_beatbox_sounds(audio_dir):
    """
    Main function for beatbox sound classification
    """
    # Load and prepare data
    X, y = load_data(audio_dir)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models to test
    models = {
        "SVM": SVC(),
        "Random Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier()
    }
    
    # Train and evaluate models
    results = {}
    for model_name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Test model
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        
        results[model_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }
        
        # Print results
        print(f"\n{model_name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    
    return results

# Main execution
if __name__ == "__main__":
    audio_dir = "audio_data"
    results = classify_beatbox_sounds(audio_dir)
```

## Expected Outcomes

1. **Classification Model** - A trained model capable of identifying beatbox sounds with high accuracy
2. **Performance Metrics** - Detailed evaluation of the model's performance across different sound types
3. **Insights** - Understanding of which beatbox sounds are most easily confused and why
4. **Imitation Analysis** - Assessment of how well participants can imitate different beatbox sounds

## Next Steps

1. **Implement the code** - Create the `beatbox_classification.py` file as outlined
2. **Run the analysis** - Execute the classification pipeline on the provided audio data
3. **Refine the model** - Optimize the model based on initial results
4. **Extend the analysis** - Consider additional features or classification approaches
5. **Document findings** - Create a comprehensive report of the classification results

## Conclusion

This approach provides a systematic method for classifying beatbox sounds using established audio feature extraction techniques. By leveraging the existing `sound_analysis.py` code and adapting it for classification tasks, we can efficiently analyze the provided audio data and gain insights into beatbox sound production and imitation.