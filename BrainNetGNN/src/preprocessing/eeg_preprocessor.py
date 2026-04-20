"""
EEG Preprocessing Pipeline for BrainNetGNN
==========================================
Handles both EEGMAT (cognitive workload) and ADHD datasets.
Pipeline: Raw EEG → Bandpass filter → Artifact rejection → Epoching → Clean segments

References:
- Prior work: mutual information on filtered EEG windows for FBN construction
- MNE-Python standard preprocessing: https://mne.tools/stable/auto_tutorials/preprocessing/
"""

import os
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from tqdm import tqdm
import warnings
import json

warnings.filterwarnings('ignore', category=RuntimeWarning)
mne.set_log_level('ERROR')

# Standard 10-20 electrode positions for 19 channels
STANDARD_19_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
    'Fz', 'Cz', 'Pz'
]

# Alternative naming (some datasets use T7/T8/P7/P8 instead of T3/T4/T5/T6)
CHANNEL_RENAME_MAP = {
    'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6',
    'EEG Fp1': 'Fp1', 'EEG Fp2': 'Fp2', 'EEG F3': 'F3', 'EEG F4': 'F4',
    'EEG C3': 'C3', 'EEG C4': 'C4', 'EEG P3': 'P3', 'EEG P4': 'P4',
    'EEG O1': 'O1', 'EEG O2': 'O2', 'EEG F7': 'F7', 'EEG F8': 'F8',
    'EEG T3': 'T3', 'EEG T4': 'T4', 'EEG T5': 'T5', 'EEG T6': 'T6',
    'EEG Fz': 'Fz', 'EEG Cz': 'Cz', 'EEG Pz': 'Pz',
    'EEG A2-A1': 'REF', 'ECG ECG': 'ECG'
}


def preprocess_eegmat(data_dir: str, output_dir: str, epoch_duration: float = 4.0):
    """
    Preprocess EEGMAT dataset (cognitive workload: rest vs. mental arithmetic).
    
    Dataset: 36 subjects, 2 conditions each (baseline _1, arithmetic _2)
    Format: EDF files, 500 Hz, 19 EEG + 1 ref + 1 ECG channels
    
    Args:
        data_dir: Path to raw EEGMAT EDF files
        output_dir: Path to save processed epochs
        epoch_duration: Duration of each epoch in seconds
    """
    print("=" * 60)
    print("PREPROCESSING EEGMAT (Cognitive Workload)")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    edf_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.edf')])
    subjects = sorted(set(f.split('_')[0] for f in edf_files))
    
    all_epochs = []
    all_labels = []
    all_subject_ids = []
    stats = {'total_subjects': 0, 'total_epochs': 0, 'baseline_epochs': 0, 'task_epochs': 0}
    
    target_sfreq = 128.0  # Downsample to match ADHD dataset
    
    for subj in tqdm(subjects, desc="Processing EEGMAT subjects"):
        for task_id, label in [('1', 0), ('2', 1)]:  # 0=baseline, 1=arithmetic
            fname = f"{subj}_{task_id}.edf"
            fpath = os.path.join(data_dir, fname)
            if not os.path.exists(fpath):
                continue
            
            try:
                # Load raw EEG
                raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
                
                # Rename channels to standard names
                rename = {ch: CHANNEL_RENAME_MAP.get(ch, ch) for ch in raw.ch_names}
                raw.rename_channels(rename)
                
                # Keep only standard 19 EEG channels
                eeg_channels = [ch for ch in raw.ch_names if ch in STANDARD_19_CHANNELS]
                if len(eeg_channels) < 19:
                    continue
                raw.pick(eeg_channels)
                
                # Reorder to standard order
                raw.reorder_channels(STANDARD_19_CHANNELS)
                
                # Set channel types and montage
                raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})
                montage = mne.channels.make_standard_montage('standard_1020')
                raw.set_montage(montage, on_missing='warn')
                
                # Bandpass filter (1-45 Hz) — standard for cognitive EEG
                raw.filter(l_freq=1.0, h_freq=45.0, verbose=False)
                
                # Notch filter at 50 Hz (power line noise)
                raw.notch_filter(freqs=50.0, verbose=False)
                
                # Resample to 128 Hz to match ADHD dataset
                raw.resample(target_sfreq, verbose=False)
                
                # Segment into fixed-length epochs
                data = raw.get_data()  # (n_channels, n_samples)
                n_samples_per_epoch = int(epoch_duration * target_sfreq)
                n_epochs = data.shape[1] // n_samples_per_epoch
                
                for i in range(n_epochs):
                    start = i * n_samples_per_epoch
                    end = start + n_samples_per_epoch
                    epoch = data[:, start:end]
                    
                    # Simple artifact rejection: reject if max amplitude > 100 µV
                    if np.max(np.abs(epoch)) < 100e-6:
                        all_epochs.append(epoch)
                        all_labels.append(label)
                        all_subject_ids.append(subj)
                        
                        if label == 0:
                            stats['baseline_epochs'] += 1
                        else:
                            stats['task_epochs'] += 1
                
            except Exception as e:
                print(f"  Error processing {fname}: {e}")
                continue
        
        stats['total_subjects'] += 1
    
    # Convert to arrays
    epochs_array = np.array(all_epochs)  # (n_epochs, n_channels, n_samples)
    labels_array = np.array(all_labels)
    subjects_array = np.array(all_subject_ids)
    
    stats['total_epochs'] = len(epochs_array)
    
    # Save
    np.save(os.path.join(output_dir, 'eegmat_epochs.npy'), epochs_array)
    np.save(os.path.join(output_dir, 'eegmat_labels.npy'), labels_array)
    np.save(os.path.join(output_dir, 'eegmat_subjects.npy'), subjects_array)
    
    with open(os.path.join(output_dir, 'eegmat_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n  Subjects processed: {stats['total_subjects']}")
    print(f"  Total epochs: {stats['total_epochs']}")
    print(f"  Baseline epochs: {stats['baseline_epochs']}")
    print(f"  Task epochs: {stats['task_epochs']}")
    print(f"  Epoch shape: {epochs_array.shape}")
    print(f"  Saved to: {output_dir}")
    
    return epochs_array, labels_array, subjects_array


def preprocess_adhd(data_path: str, output_dir: str, epoch_duration: float = 4.0):
    """
    Preprocess ADHD EEG dataset (ADHD vs. healthy controls).
    
    Dataset: 121 children (61 ADHD, 60 control), 19 channels, 128 Hz
    Format: CSV with columns Fp1..Pz, Class, ID
    
    Args:
        data_path: Path to adhdata.csv
        output_dir: Path to save processed epochs
        epoch_duration: Duration of each epoch in seconds
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING ADHD DATASET")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CSV
    print("  Loading CSV...")
    df = pd.read_csv(data_path)
    
    # Channel columns (first 19 columns)
    channel_cols = [c for c in df.columns if c not in ['Class', 'ID']]
    
    # Map T7->T3, T8->T4, P7->T5, P8->T6 for standard naming
    col_rename = {c: CHANNEL_RENAME_MAP.get(c, c) for c in channel_cols}
    df.rename(columns=col_rename, inplace=True)
    channel_cols = [col_rename.get(c, c) for c in channel_cols]
    
    sfreq = 128.0
    n_samples_per_epoch = int(epoch_duration * sfreq)
    
    all_epochs = []
    all_labels = []
    all_subject_ids = []
    stats = {'total_subjects': 0, 'total_epochs': 0, 'adhd_epochs': 0, 'control_epochs': 0}
    
    subjects = df['ID'].unique()
    
    # Build channel name mapping once
    ch_names_standard = []
    for ch in channel_cols:
        if ch in STANDARD_19_CHANNELS:
            ch_names_standard.append(ch)
        elif ch in CHANNEL_RENAME_MAP:
            ch_names_standard.append(CHANNEL_RENAME_MAP[ch])
        else:
            ch_names_standard.append(ch)
    
    for subj in tqdm(subjects, desc="Processing ADHD subjects"):
        subj_data = df[df['ID'] == subj]
        label = 1 if subj_data['Class'].iloc[0] == 'ADHD' else 0
        
        # Get EEG data for this subject
        eeg_data = subj_data[channel_cols].values.T.astype(np.float64)  # (n_channels, n_samples)
        
        # Data is in arbitrary ADC units — scale to volts
        # Typical EEG range: ±100 µV; dataset range: ~±13000 units
        # Scale factor: 13000 units ≈ 100 µV → 1 unit ≈ 0.0077 µV
        eeg_data = eeg_data * 0.0077e-6  # Convert to volts
        
        info = mne.create_info(ch_names=ch_names_standard, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(eeg_data, info, verbose=False)
        
        # Set montage
        try:
            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage, on_missing='warn')
        except:
            pass
        
        # Bandpass filter (1-45 Hz)
        raw.filter(l_freq=1.0, h_freq=45.0, verbose=False)
        
        # Notch filter at 50 Hz
        raw.notch_filter(freqs=50.0, verbose=False)
        
        # Get filtered data
        data = raw.get_data()  # (n_channels, n_samples)
        
        # Segment into epochs
        n_epochs = data.shape[1] // n_samples_per_epoch
        
        for i in range(n_epochs):
            start = i * n_samples_per_epoch
            end = start + n_samples_per_epoch
            epoch = data[:, start:end]
            
            # Z-score based artifact rejection: reject if any channel has
            # extreme values (> 5 std from epoch mean)
            epoch_zscore = np.abs((epoch - epoch.mean()) / (epoch.std() + 1e-10))
            if epoch_zscore.max() < 8.0:  # lenient threshold for children's EEG
                all_epochs.append(epoch)
                all_labels.append(label)
                all_subject_ids.append(subj)
                
                if label == 1:
                    stats['adhd_epochs'] += 1
                else:
                    stats['control_epochs'] += 1
        
        stats['total_subjects'] += 1
    
    # Convert to arrays
    epochs_array = np.array(all_epochs)
    labels_array = np.array(all_labels)
    subjects_array = np.array(all_subject_ids)
    
    stats['total_epochs'] = len(epochs_array)
    
    # Save
    np.save(os.path.join(output_dir, 'adhd_epochs.npy'), epochs_array)
    np.save(os.path.join(output_dir, 'adhd_labels.npy'), labels_array)
    np.save(os.path.join(output_dir, 'adhd_subjects.npy'), subjects_array)
    
    with open(os.path.join(output_dir, 'adhd_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n  Subjects processed: {stats['total_subjects']}")
    print(f"  Total epochs: {stats['total_epochs']}")
    print(f"  ADHD epochs: {stats['adhd_epochs']}")
    print(f"  Control epochs: {stats['control_epochs']}")
    print(f"  Epoch shape: {epochs_array.shape}")
    print(f"  Saved to: {output_dir}")
    
    return epochs_array, labels_array, subjects_array


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parents[2]
    
    # Preprocess EEGMAT
    eegmat_epochs, eegmat_labels, eegmat_subjects = preprocess_eegmat(
        data_dir=str(base_dir / 'data' / 'raw' / 'eegmat'),
        output_dir=str(base_dir / 'data' / 'processed'),
        epoch_duration=4.0
    )
    
    # Preprocess ADHD
    adhd_epochs, adhd_labels, adhd_subjects = preprocess_adhd(
        data_path=str(base_dir / 'data' / 'raw' / 'adhd' / 'adhdata.csv'),
        output_dir=str(base_dir / 'data' / 'processed'),
        epoch_duration=4.0
    )
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"EEGMAT: {eegmat_epochs.shape[0]} epochs, shape {eegmat_epochs.shape}")
    print(f"ADHD:   {adhd_epochs.shape[0]} epochs, shape {adhd_epochs.shape}")
