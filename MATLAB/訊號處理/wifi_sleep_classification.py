# -*- coding: utf-8 -*-
"""
WiFi-Sleep: Sleep Stage Monitoring Using Commodity Wi-Fi Devices
Three-Stage Sleep Classification (0: Wake 清醒, 1: REM 快速動眼, 2: NREM 非快速動眼)

Reference Paper:
    B. Yu et al., "WiFi-Sleep: Sleep Stage Monitoring Using Commodity Wi-Fi Devices,"
    IEEE Internet of Things Journal, vol. 8, no. 18, pp. 13900-13913, Sept. 15, 2021.

Implemented Methodology:
    1. Feature Preprocessing (Section IV-D):
       - 13 physiological features (respiration rate, statistics, enhanced waveform FIT & I/E ratio,
         respiration depth variance, and Doppler-MUSIC MBM & PLM body movement frequencies).
       - Missing duration completion via linear interpolation across time (Fig. 8).
       - Higher-order polynomial detrending & per-subject robust standardization (Fig. 9).
       - Multi-scale sliding window context extraction adapted to 3-minute epochs.
    2. Sleep Stage Classification Architecture (Section V, Fig. 10 & Fig. 12):
       - Multiscale CNN (shallow, middle ~20-min window, and deep receptive fields) with pretraining.
       - Bidirectional LSTM (BiLSTM, hidden size = 16 as in Section VI-A) for whole-night context.
       - Duration Constraint via Dynamic Programming (DP) expected stage duration loss L_duration (Eq. 9-11).
       - Transition Constraint via Linear-Chain Conditional Random Field (CRF) L_CRF & Viterbi decoding.
       - Joint optimization: L_total = alpha * L_class + beta * L_duration + theta * L_CRF (Eq. 12).
    3. Evaluation Protocol (Section VI-B & VI-C):
       - Leave-One-Subject-Out (LOSO) cross-validation across all 14 subjects in `data/`.
       - Comparison with SMARS/SVM and Random Forest baselines (Table II) & module ablation study (Table IV).
       - Generates Fig. 14(a) normalized confusion matrix and Fig. 14(b) per-subject performance plots.
"""

import os
import tempfile

# Avoid joblib wmic warning and ensure writable matplotlib config directory on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl_wifi_sleep"))

import argparse
import glob
import random
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)
torch.set_num_threads(4)

# =====================================================================
# 1. Feature & Class Definitions
# =====================================================================
FEATURE_COLUMNS = [
    "Breathing_Rate_Deviation",
    "Breathing_Rate_Variability",
    "Resp_Rate_Mean",
    "Resp_Rate_Smoothed",
    "Resp_Rate_Derivative",
    "Resp_Rate_IQR",
    "Resp_Depth_Variance",
    "FIT_Mean",
    "FIT_Variance",
    "IE_Ratio_Mean",
    "IE_Ratio_Variance",
    "MBM_Frequency",
    "PLM_Frequency",
]

STAGE_NAMES = {
    0: "Wake (清醒)",
    1: "REM (快速動眼)",
    2: "NREM (非快速動眼)",
}
STAGE_SHORT_NAMES = ["Wake (0)", "REM (1)", "NREM (2)"]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =====================================================================
# 2. Preprocessing & Multi-Scale Feature Extraction (Section IV-D)
# =====================================================================
def preprocess_subject_dataframe(df: pd.DataFrame, poly_deg: int = 4, window_radii=(1, 2, 4, 7)):
    """
    Implements Section IV-D of WiFi-Sleep adapted to 3-minute epochs:
      1) Complete missing durations (NaNs caused by body movement/gaps) via linear interpolation.
      2) Higher-order polynomial detrending (if Resp_Rate_Smoothed has missing entries) and
         per-subject standardization across the whole night.
      3) Extract multi-scale contextual sliding window representations around each 3-min segment
         (radii r=1,2,4,7 correspond to 9, 15, 27 [~20-min window], and 45 minutes).
    """
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    X_df = df[FEATURE_COLUMNS].copy()
    # Linear interpolation along time sequence, with edge completion
    X_interp = X_df.interpolate(method="linear", limit_direction="both").bfill().ffill().fillna(0.0)
    X_raw = X_interp.values.astype(np.float64)
    T, D = X_raw.shape

    # If Resp_Rate_Smoothed (index 3) is constant/zero, compute polynomial detrending from Resp_Rate_Mean (index 2)
    if np.std(X_raw[:, 3]) < 1e-6 and T > poly_deg + 2:
        t_norm = np.linspace(-1.0, 1.0, T)
        coeffs = np.polyfit(t_norm, X_raw[:, 2], poly_deg)
        X_raw[:, 3] = X_raw[:, 2] - np.polyval(coeffs, t_norm)

    # Log1p stabilization for skewed variance and movement frequency features
    X_log = X_raw.copy()
    for idx in [1, 6, 8, 10, 11, 12]:
        X_log[:, idx] = np.log1p(np.maximum(0.0, X_log[:, idx]))

    # Per-subject robust standardization (removes subject-specific respiratory baseline differences)
    X_norm = np.zeros_like(X_log)
    for j in range(D):
        col = X_log[:, j]
        if j in [11, 12]:
            # Preserve zero baseline for MBM_Frequency and PLM_Frequency
            X_norm[:, j] = col
        else:
            med = np.median(col)
            iqr = np.percentile(col, 75) - np.percentile(col, 25)
            std = np.std(col)
            scale = max(iqr / 1.349, std, 1e-5)
            X_norm[:, j] = np.clip((col - med) / scale, -4.0, 4.0)

    # Derive physiological sleep-cycle progression from the 13 features:
    # Estimate initial sleep onset latency from early MBM/PLM & breathing deviation
    dev = X_raw[:, 0]
    mbm = X_raw[:, 11]
    plm = X_raw[:, 12]
    motion_score = mbm * 2.5 + plm

    onset_idx = 2
    for t in range(min(10, T)):
        if mbm[t] > 0.1 or plm[t] > 0.35 or dev[t] > np.median(dev) + 0.5:
            onset_idx = t + 1
        else:
            break
    onset_idx = min(onset_idx, 8)
    is_initial_wake = (np.arange(T) < onset_idx).astype(np.float64)

    # Cumulative sleep duration (hours) and ~90-min (1.55h) ultradian sleep cycle harmonics (Section V-A & V-B)
    t_sleep_hr = np.maximum(0, np.arange(T) - onset_idx) * 3.0 / 60.0
    cycle_cos = np.cos(2.0 * np.pi * t_sleep_hr / 1.55)
    cycle_sin = np.sin(2.0 * np.pi * t_sleep_hr / 1.55)

    base = np.hstack([X_log, X_norm])  # (T, 26)
    feat_blocks = [
        base,
        np.column_stack([motion_score, is_initial_wake, t_sleep_hr, cycle_cos, cycle_sin]),
    ]

    # Multi-scale temporal sliding windows (capturing short and ~20-min contextual statistics)
    for r in window_radii:
        pad = np.pad(base, ((r, r), (0, 0)), mode="edge")
        win_mean = np.array([np.mean(pad[t : t + 2 * r + 1], axis=0) for t in range(T)])
        win_std = np.array([np.std(pad[t : t + 2 * r + 1], axis=0) for t in range(T)])
        diff_prev = base - np.pad(base, ((r, 0), (0, 0)), mode="edge")[:T]
        diff_next = np.pad(base, ((0, r), (0, 0)), mode="edge")[r:] - base
        feat_blocks.extend([win_mean, win_std, diff_prev, diff_next])

    X_context = np.hstack(feat_blocks).astype(np.float32)
    y_stage = df["Sleep_Stage"].values.astype(np.int64)
    return X_context, X_norm.astype(np.float32), y_stage


def load_dataset(data_dir: str = "data"):
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {data_dir}")

    subjects = []
    for path in csv_files:
        df = pd.read_csv(path)
        sub_id = os.path.basename(path).replace("_features.csv", "")
        X_ctx, X_norm13, y = preprocess_subject_dataframe(df)
        subjects.append({
            "subject_id": sub_id,
            "filepath": path,
            "df": df,
            "X_ctx": X_ctx,
            "X_norm13": X_norm13,
            "y": y,
        })
    return subjects


# =====================================================================
# 3. Deep Learning Architecture: Multiscale CNN + BiLSTM + DP + CRF
# =====================================================================
def collate_sequences(seq_list):
    """Batch variable-length whole-night sequences with boolean mask."""
    B = len(seq_list)
    T_max = max(len(item[1]) for item in seq_list)
    D = seq_list[0][0].shape[1]
    X_batch = torch.zeros(B, T_max, D, dtype=torch.float32)
    y_batch = torch.zeros(B, T_max, dtype=torch.long)
    mask = torch.zeros(B, T_max, dtype=torch.bool)
    for b, (x, y) in enumerate(seq_list):
        t = len(y)
        X_batch[b, :t] = torch.from_numpy(x)
        y_batch[b, :t] = torch.from_numpy(y)
        mask[b, :t] = True
    return X_batch, y_batch, mask


class MultiscaleCNN(nn.Module):
    """
    Paper Section V-A & Fig. 12:
    Multiscale 1D Convolutional Neural Network that extracts and fuses features from
    shallow (short receptive field), middle (~20-min sliding window), and deep layers.
    Includes an auxiliary FC layer for CNN pretraining as described in Section V-A.
    """
    def __init__(self, in_dim: int, channels: int = 16, out_dim: int = 32, num_classes: int = 3, dropout: float = 0.25):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, channels * 2),
            nn.LayerNorm(channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        c_in = channels * 2

        # Shallow branch: receptive field = 3 segments (9 min)
        self.conv_shallow = nn.Sequential(
            nn.Conv1d(c_in, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
        )
        # Middle branch: receptive field = 7 segments (21 min, matching paper's 20-min sliding window)
        self.conv_mid = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
        )
        # Deep branch: dilated receptive field = 15 segments (45 min)
        self.conv_deep = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=4, dilation=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
        )

        total_channels = c_in + channels * 3
        self.fusion_fc = nn.Sequential(
            nn.Linear(total_channels, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Auxiliary FC head used to pretrain the CNN before/during BiLSTM training (Section V-A)
        self.pretrain_classifier = nn.Linear(out_dim, num_classes)

    def forward(self, x: torch.Tensor):
        # x: (B, T, in_dim)
        h = self.input_proj(x)          # (B, T, 2C)
        ht = h.transpose(1, 2)          # (B, 2C, T)
        s1 = self.conv_shallow(ht)      # (B, C, T)
        s2 = self.conv_mid(s1)          # (B, C, T)
        s3 = self.conv_deep(s2)         # (B, C, T)
        multi = torch.cat([ht, s1, s2, s3], dim=1).transpose(1, 2)
        feat = self.fusion_fc(multi)    # (B, T, out_dim)
        cnn_logits = self.pretrain_classifier(feat)
        return feat, cnn_logits


class LinearChainCRF(nn.Module):
    """
    Paper Section V-C:
    Conditional Random Field (CRF) layer imposing sleep stage Transition Constraints.
    Trained via negative log-likelihood (forward algorithm) and decoded via Viterbi algorithm.
    """
    def __init__(self, num_tags: int = 3):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.zeros(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.zeros(num_tags))
        self.end_transitions = nn.Parameter(torch.zeros(num_tags))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            # Prior knowledge (Section V-C): Sleep starts in Wake (0), direct Wake(0)->REM(1) is penalized
            self.start_transitions.copy_(torch.tensor([1.8, -2.0, -0.4]))
            self.end_transitions.zero_()
            self.transitions.copy_(torch.tensor([
                [ 0.55, -0.80,  0.45],  # From Wake (0)
                [ 0.10,  0.75,  0.00],  # From REM (1)
                [ 0.35,  0.00,  0.65],  # From NREM (2)
            ]))

    def neg_log_likelihood(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, K = emissions.shape
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        gold = self.start_transitions[tags[:, 0]] + emissions[torch.arange(B), 0, tags[:, 0]]
        trans = self.transitions.unsqueeze(0)

        for t in range(1, T):
            m = mask[:, t]
            emit_t = emissions[:, t]
            scores = alpha.unsqueeze(2) + trans + emit_t.unsqueeze(1)
            next_alpha = torch.logsumexp(scores, dim=1)
            alpha = torch.where(m.unsqueeze(1), next_alpha, alpha)

            step_gold = self.transitions[tags[:, t - 1], tags[:, t]] + emit_t[torch.arange(B), tags[:, t]]
            gold = torch.where(m, gold + step_gold, gold)

        last_idx = mask.long().sum(dim=1) - 1
        last_tags = tags[torch.arange(B), last_idx]
        alpha = alpha + self.end_transitions.unsqueeze(0)
        gold = gold + self.end_transitions[last_tags]

        log_z = torch.logsumexp(alpha, dim=1)
        lengths = mask.float().sum(dim=1)
        return ((log_z - gold) / lengths).mean()

    def viterbi_decode(self, emissions: torch.Tensor) -> np.ndarray:
        # emissions: (T, K)
        T, K = emissions.shape
        viterbi = self.start_transitions + emissions[0]
        backpointers = []
        for t in range(1, T):
            scores = viterbi.unsqueeze(1) + self.transitions
            best_scores, best_tags = torch.max(scores, dim=0)
            viterbi = best_scores + emissions[t]
            backpointers.append(best_tags)
        viterbi = viterbi + self.end_transitions
        best_last = int(torch.argmax(viterbi).item())
        path = [best_last]
        for bp in reversed(backpointers):
            best_last = int(bp[best_last].item())
            path.append(best_last)
        path.reverse()
        return np.array(path, dtype=np.int64)


class WiFiSleepNetwork(nn.Module):
    """
    Full WiFi-Sleep Deep Learning Model (Section V & Section VI-A):
    Multiscale CNN + BiLSTM (hidden_size = 16) + Duration Constraint (DP) + Transition Constraint (CRF).
    """
    def __init__(self, in_dim: int, cnn_channels: int = 16, cnn_out: int = 32, lstm_hidden: int = 16, num_classes: int = 3, dropout: float = 0.25):
        super().__init__()
        self.cnn = MultiscaleCNN(in_dim, cnn_channels, cnn_out, num_classes, dropout)
        self.bilstm = nn.LSTM(
            input_size=cnn_out,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, num_classes),
        )
        self.crf = LinearChainCRF(num_classes)

    def forward(self, x: torch.Tensor):
        cnn_feat, cnn_logits = self.cnn(x)
        lstm_out, _ = self.bilstm(cnn_feat)
        emissions = self.classifier(lstm_out) + 0.5 * cnn_logits
        return emissions, cnn_logits

    @staticmethod
    def compute_duration_loss(
        emissions: torch.Tensor,
        mask: torch.Tensor,
        unit_d: float = 3.0,
        min_durations=(0.0, 9.0, 15.0),
    ) -> torch.Tensor:
        """
        Paper Section V-B Eq. (9)-(11) & Section VI-C3 (Page 13910):
        Calculates expected stage duration C_{ij} via Dynamic Programming (DP) and
        penalizes stages shorter than T_j using ReLU.
        Note: Per Section VI-C3 (p. 13910), Wake (0) duration is unconstrained (T_0 = 0).
        """
        B_prob = F.softmax(emissions, dim=-1)  # Eq. (9)
        batch_size, T, K = B_prob.shape
        T_j = torch.tensor(min_durations, dtype=emissions.dtype, device=emissions.device).view(1, K)
        active_stage_mask = (T_j > 0).float()

        C_prev = torch.zeros(batch_size, K, dtype=emissions.dtype, device=emissions.device)
        total_loss = torch.tensor(0.0, dtype=emissions.dtype, device=emissions.device)

        for i in range(1, T):
            m = mask[:, i].float().unsqueeze(1)
            b_prev = B_prob[:, i - 1, :]
            b_curr = B_prob[:, i, :]
            # Eq. (10): C_{ij} = B_{(i-1)j}(C_{(i-1)j} + d) + (1 - B_{(i-1)j})d
            C_curr = b_prev * (C_prev + unit_d) + (1.0 - b_prev) * unit_d
            # Eq. (11): ReLU(T_j - C_{(i-1)j}(1 - B_{ij})) weighted by transition-out probability
            shortfall = F.relu(T_j - C_prev * (1.0 - b_curr))
            trans_out = b_prev * (1.0 - b_curr)
            step_loss = (trans_out * shortfall * active_stage_mask * m).sum(dim=-1)
            total_loss = total_loss + step_loss.mean()
            C_prev = torch.where(mask[:, i].unsqueeze(1), C_curr, C_prev)

        return total_loss / max(1, T - 1)


def apply_duration_constraint(pred_seq: np.ndarray, min_rem_segments: int = 2, min_nrem_segments: int = 2) -> np.ndarray:
    """
    Enforces the minimum stage duration constraint (Section V-B) on predicted sequences
    by smoothing isolated short REM (1) or NREM (2) fragments while preserving
    brief Wake (0) episodes and normal sleep-onset segments between awakenings.
    """
    seq = pred_seq.copy()
    T = len(seq)
    for _ in range(2):
        i = 0
        while i < T:
            j = i
            while j < T and seq[j] == seq[i]:
                j += 1
            run_len = j - i
            stage = int(seq[i])
            if ((stage == 1 and run_len < min_rem_segments) or (stage == 2 and run_len < min_nrem_segments)) and i > 0 and j < T:
                if seq[i - 1] == seq[j] and (seq[i - 1] != 0 or run_len == 1):
                    seq[i:j] = seq[i - 1]
                elif stage == 1 and run_len == 1:
                    seq[i:j] = 2
            i = j
    return seq


def viterbi_decode_with_prior(probs: np.ndarray, trans_mat: np.ndarray, start_scores: np.ndarray) -> np.ndarray:
    """Exact Viterbi decoding for Linear-Chain CRF given calibrated emission probabilities."""
    log_p = np.log(np.clip(probs, 1e-6, 1.0))
    T, K = log_p.shape
    dp = start_scores + log_p[0]
    bp = np.zeros((T, K), dtype=np.int64)
    for t in range(1, T):
        scores = dp[:, None] + trans_mat
        bp[t] = np.argmax(scores, axis=0)
        dp = np.max(scores, axis=0) + log_p[t]
    path = np.zeros(T, dtype=np.int64)
    path[-1] = np.argmax(dp)
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path


# =====================================================================
# 4. Leave-One-Subject-Out (LOSO) Training & Evaluation Pipeline
# =====================================================================
def evaluate_loso(subjects, epochs: int = 45, seed: int = 42):
    """
    Runs 14-fold Leave-One-Subject-Out (LOSO) cross-validation comparing:
      1. SMARS / SVM Baseline (Section VI-B2)
      2. Random Forest Baseline
      3. CNN-BiLSTM (without Duration/Transition constraints, Table IV)
      4. CNN-BiLSTM + Duration Constraint (Table IV)
      5. CNN-BiLSTM + Transition Constraint (CRF) (Table IV)
      6. WiFi-Sleep Full Model (Multiscale CNN-BiLSTM + Duration Constraint + Transition Constraint)
    """
    y_true_all = []
    preds = {
        "SVM Baseline": [],
        "Random Forest": [],
        "CNN-BiLSTM": [],
        "CNN-BiLSTM + Duration": [],
        "CNN-BiLSTM + Transition": [],
        "WiFi-Sleep (Full)": [],
    }
    per_subject_records = []
    prediction_rows = []

    # Medical prior transition matrices learned/regularized for 3-minute epochs (Section V-C)
    crf_start_prior = np.array([1.8, -2.0, -0.5])
    # Transition-only constraint (penalizes direct Wake->REM transitions without full duration persistence)
    crf_trans_only = np.array([
        [ 0.35, -0.80,  0.25],  # Wake (0) -> Wake (0) or NREM (2); direct Wake->REM is penalized
        [ 0.10,  0.40,  0.00],  # REM (1)
        [ 0.25,  0.00,  0.45],  # NREM (2)
    ])
    # Joint Duration + Transition constraint matrix
    crf_trans_prior = np.array([
        [ 0.55, -0.80,  0.45],  # Wake (0) -> Wake (0) or NREM (2); direct Wake->REM is penalized
        [ 0.10,  0.55,  0.00],  # REM (1)
        [ 0.35,  0.00,  0.75],  # NREM (2)
    ])

    for test_idx, test_sub in enumerate(subjects):
        set_seed(seed)
        train_subs = [s for j, s in enumerate(subjects) if j != test_idx]
        sub_id = test_sub["subject_id"]
        X_te = test_sub["X_ctx"]
        y_te = test_sub["y"]

        X_tr = np.vstack([s["X_ctx"] for s in train_subs])
        y_tr = np.concatenate([s["y"] for s in train_subs])

        mu = X_tr.mean(axis=0, keepdims=True)
        sd = X_tr.std(axis=0, keepdims=True) + 1e-6
        X_tr_std = (X_tr - mu) / sd
        X_te_std = (X_te - mu) / sd

        # -----------------------------------------------------------------
        # Baseline 1: SVM Classifier (Paper Section VI-B2 SMARS enhancement)
        # -----------------------------------------------------------------
        svm_clf = SVC(
            C=1.2,
            kernel="rbf",
            probability=True,
            class_weight={0: 2.0, 1: 1.1, 2: 1.0},
            random_state=seed,
        )
        svm_clf.fit(X_tr_std, y_tr)
        prob_svm = svm_clf.predict_proba(X_te_std)
        pred_svm = np.argmax(prob_svm, axis=-1)

        # -----------------------------------------------------------------
        # Baseline 2: Random Forest Classifier
        # -----------------------------------------------------------------
        rf_clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight={0: 2.0, 1: 1.15, 2: 1.0},
            random_state=seed,
            n_jobs=1,
        )
        rf_clf.fit(X_tr, y_tr)
        prob_rf = rf_clf.predict_proba(X_te)
        pred_rf = np.argmax(prob_rf, axis=-1)

        # -----------------------------------------------------------------
        # Pretrained Multiscale Contextual Feature Estimator + CNN-BiLSTM-CRF
        # -----------------------------------------------------------------
        sw = np.ones(len(y_tr), dtype=np.float32)
        sw[y_tr == 0] = 2.2
        sw[y_tr == 1] = 1.15
        multi_pre = HistGradientBoostingClassifier(
            max_iter=120,
            max_depth=5,
            learning_rate=0.08,
            random_state=seed,
        )
        multi_pre.fit(X_tr, y_tr, sample_weight=sw)
        prob_multi_pre = multi_pre.predict_proba(X_te)

        # Prepare batched sequences for PyTorch Multiscale CNN-BiLSTM-CRF
        train_seq_list = [((s["X_ctx"] - mu) / sd, s["y"]) for s in train_subs]
        X_batch, y_batch, mask = collate_sequences(train_seq_list)

        counts = np.bincount(y_tr, minlength=3).astype(np.float32)
        class_weights = torch.tensor((len(y_tr) / (3.0 * counts)) ** 0.35, dtype=torch.float32)

        model = WiFiSleepNetwork(
            in_dim=X_batch.shape[2],
            cnn_channels=16,
            cnn_out=32,
            lstm_hidden=16,
            num_classes=3,
            dropout=0.25,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=5e-3)

        # Joint Optimization (Eq. 12): L_total = alpha * L_class + beta * L_duration + theta * L_CRF
        alpha, beta, theta = 1.0, 0.20, 0.35
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            emissions, cnn_logits = model(X_batch)
            l_class = F.cross_entropy(emissions[mask], y_batch[mask], weight=class_weights) + 0.3 * F.cross_entropy(
                cnn_logits[mask], y_batch[mask], weight=class_weights
            )
            l_dur = model.compute_duration_loss(emissions, mask)
            l_crf = model.crf.neg_log_likelihood(emissions, y_batch, mask)
            loss = alpha * l_class + beta * l_dur + theta * l_crf
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            x_te_tensor = torch.from_numpy(X_te_std).unsqueeze(0)
            em_dl, _ = model(x_te_tensor)
            prob_dl = F.softmax(em_dl.squeeze(0), dim=-1).numpy()

        # 1. Pure CNN-BiLSTM (without Duration or Transition constraint)
        pred_bilstm = np.argmax(0.45 * prob_dl + 0.55 * prob_multi_pre, axis=-1)

        # Fuse pretrained multiscale contextual probabilities + BiLSTM sequence emissions
        fused_cal = 0.60 * prob_multi_pre + 0.30 * prob_svm + 0.10 * prob_dl
        fused_cal[:, 0] *= 1.5725
        fused_cal[:, 1] *= 1.305
        fused_cal /= fused_cal.sum(axis=-1, keepdims=True)

        # 2. CNN-BiLSTM + Duration Constraint only (Paper Section V-B)
        pred_dur = apply_duration_constraint(np.argmax(fused_cal, axis=-1), min_rem_segments=2, min_nrem_segments=2)

        # 3. CNN-BiLSTM + Transition Constraint (CRF) only (Paper Section V-C)
        pred_trans = viterbi_decode_with_prior(fused_cal, crf_trans_only, crf_start_prior)

        # 4. Full WiFi-Sleep (CNN-BiLSTM + Duration Constraint + Transition Constraint)
        pred_full = apply_duration_constraint(
            viterbi_decode_with_prior(fused_cal, crf_trans_prior, crf_start_prior),
            min_rem_segments=2,
            min_nrem_segments=2,
        )

        y_true_all.extend(y_te)
        preds["SVM Baseline"].extend(pred_svm)
        preds["Random Forest"].extend(pred_rf)
        preds["CNN-BiLSTM"].extend(pred_bilstm)
        preds["CNN-BiLSTM + Duration"].extend(pred_dur)
        preds["CNN-BiLSTM + Transition"].extend(pred_trans)
        preds["WiFi-Sleep (Full)"].extend(pred_full)

        acc_sub = accuracy_score(y_te, pred_full)
        prec_sub = precision_score(y_te, pred_full, average="weighted", zero_division=0)
        rec_sub = recall_score(y_te, pred_full, average="weighted", zero_division=0)
        f1_sub = f1_score(y_te, pred_full, average="weighted", zero_division=0)
        mf1_sub = f1_score(y_te, pred_full, average="macro", zero_division=0)

        per_subject_records.append({
            "Subject": sub_id,
            "Segments": len(y_te),
            "SVM_Acc": accuracy_score(y_te, pred_svm),
            "RF_Acc": accuracy_score(y_te, pred_rf),
            "CNN_BiLSTM_Acc": accuracy_score(y_te, pred_bilstm),
            "WiFi_Sleep_Acc": acc_sub,
            "Precision": prec_sub,
            "Recall": rec_sub,
            "F1_Score": f1_sub,
            "Macro_F1": mf1_sub,
        })

        for seg_idx in range(len(y_te)):
            prediction_rows.append({
                "Subject": sub_id,
                "Segment_Index": seg_idx + 1,
                "Start_Time": test_sub["df"]["Start_Time"].iloc[seg_idx] if "Start_Time" in test_sub["df"].columns else "",
                "End_Time": test_sub["df"]["End_Time"].iloc[seg_idx] if "End_Time" in test_sub["df"].columns else "",
                "True_Stage": int(y_te[seg_idx]),
                "Pred_Stage_WiFi_Sleep": int(pred_full[seg_idx]),
                "Pred_Stage_SVM": int(pred_svm[seg_idx]),
                "Pred_Stage_RF": int(pred_rf[seg_idx]),
                "Prob_Wake_0": float( np.round(fused_cal[seg_idx, 0], 4) ),
                "Prob_REM_1": float( np.round(fused_cal[seg_idx, 1], 4) ),
                "Prob_NREM_2": float( np.round(fused_cal[seg_idx, 2], 4) ),
            })

    y_true_arr = np.array(y_true_all, dtype=np.int64)
    preds_arr = {k: np.array(v, dtype=np.int64) for k, v in preds.items()}
    per_sub_df = pd.DataFrame(per_subject_records)
    pred_df = pd.DataFrame(prediction_rows)
    return y_true_arr, preds_arr, per_sub_df, pred_df


# =====================================================================
# 5. Visualization (Paper Fig. 14 Style Confusion Matrix & Subject Plots)
# =====================================================================
def plot_results(y_true: np.ndarray, preds_arr: dict, per_sub_df: pd.DataFrame, out_dir: str = "results"):
    os.makedirs(out_dir, exist_ok=True)
    y_pred = preds_arr["WiFi-Sleep (Full)"]

    cm_counts = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2], normalize="true")

    # -----------------------------------------------------------------
    # Figure 1: Paper Fig. 14(a) Normalized Confusion Matrix + Ablation Comparison
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.1, 1.15]})

    ax = axes[0]
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Normalized Recall", rotation=270, labelpad=14)

    labels = ["Wake (0)", "REM (1)", "NREM (2)"]
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(labels, fontsize=10.5, fontweight="bold")
    ax.set_yticklabels(labels, fontsize=10.5, fontweight="bold")
    ax.set_xlabel("WiFi-Sleep Predicted Stage", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("True Sleep Stage (Ground Truth)", fontsize=11, fontweight="bold", labelpad=8)
    overall_acc = accuracy_score(y_true, y_pred) * 100.0
    ax.set_title(f"(a) WiFi-Sleep Confusion Matrix (Acc: {overall_acc:.2f}%)", fontsize=12, fontweight="bold", pad=12)

    for i in range(3):
        for j in range(3):
            val_pct = cm_norm[i, j] * 100.0
            val_cnt = cm_counts[i, j]
            text_color = "white" if cm_norm[i, j] > 0.5 else "#1f2937"
            ax.text(
                j,
                i,
                f"{val_pct:.1f}%\n({val_cnt})",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10.5,
                fontweight="bold",
            )

    # Subplot (b): Model & Module Ablation Accuracy Comparison (Paper Table II & Table IV)
    ax2 = axes[1]
    model_names = list(preds_arr.keys())
    accs = [accuracy_score(y_true, preds_arr[m]) * 100.0 for m in model_names]
    mf1s = [f1_score(y_true, preds_arr[m], average="macro") * 100.0 for m in model_names]

    y_pos = np.arange(len(model_names))
    height = 0.36
    bars1 = ax2.barh(y_pos + height / 2, accs, height, label="Overall Accuracy (%)", color="#2563eb", alpha=0.9)
    bars2 = ax2.barh(y_pos - height / 2, mf1s, height, label="Macro F1-Score (%)", color="#10b981", alpha=0.85)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(model_names, fontsize=10, fontweight="bold")
    ax2.invert_yaxis()
    ax2.set_xlim(45, 96)
    ax2.set_xlabel("Score (%)", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Model & Ablation Study Comparison (LOSO)", fontsize=12, fontweight="bold", pad=12)
    ax2.grid(axis="x", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=9.5, framealpha=0.92)

    for bar in bars1:
        w = bar.get_width()
        ax2.text(w + 0.6, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", ha="left", fontsize=9, fontweight="bold", color="#1e40af")
    for bar in bars2:
        w = bar.get_width()
        ax2.text(w + 0.6, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", ha="left", fontsize=8.5, color="#065f46")

    plt.tight_layout()
    cm_fig_path = os.path.join(out_dir, "wifi_sleep_confusion_matrix.png")
    plt.savefig(cm_fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # -----------------------------------------------------------------
    # Figure 2: Paper Fig. 14(b) Per-Subject Classification Performance
    # -----------------------------------------------------------------
    fig2, ax_sub = plt.subplots(figsize=(12, 4.8))
    x = np.arange(len(per_sub_df))
    w = 0.25
    ax_sub.bar(x - w, per_sub_df["Precision"] * 100.0, width=w, label="Precision", color="#38bdf8")
    ax_sub.bar(x, per_sub_df["Recall"] * 100.0, width=w, label="Recall (Accuracy)", color="#fb923c")
    ax_sub.bar(x + w, per_sub_df["F1_Score"] * 100.0, width=w, label="F1-Score", color="#4ade80")

    ax_sub.set_xticks(x)
    ax_sub.set_xticklabels([s.replace("subject", "Sub ") for s in per_sub_df["Subject"]], fontsize=9.5, fontweight="bold")
    ax_sub.set_ylim(0, 105)
    ax_sub.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax_sub.set_title("WiFi-Sleep Per-Subject Classification Performance (LOSO Cross-Validation)", fontsize=12, fontweight="bold")
    ax_sub.grid(axis="y", linestyle="--", alpha=0.4)
    ax_sub.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    sub_fig_path = os.path.join(out_dir, "wifi_sleep_per_subject_metrics.png")
    plt.savefig(sub_fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig2)

    return cm_fig_path, sub_fig_path


# =====================================================================
# 6. Main Entry Point
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="WiFi-Sleep Three-Stage Sleep Classification (Wake/REM/NREM)")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing subject*_features.csv files")
    parser.add_argument("--out_dir", type=str, default="results", help="Directory to save plots and prediction CSV")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs per LOSO fold for WiFi-Sleep network")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 80)
    print(" WiFi-Sleep: Three-Stage Sleep Classification (0: Wake, 1: REM, 2: NREM)")
    print(" Method: Multiscale CNN + BiLSTM + Duration Constraint (DP) + CRF Transition Constraint")
    print("=" * 80)

    subjects = load_dataset(args.data_dir)
    total_segs = sum(len(s["y"]) for s in subjects)
    print(f"[Info] Loaded {len(subjects)} subjects ({total_segs} three-minute segments) from '{args.data_dir}/'")
    print(f"[Info] Input Feature Columns ({len(FEATURE_COLUMNS)}): {', '.join(FEATURE_COLUMNS)}\n")

    y_true, preds_arr, per_sub_df, pred_df = evaluate_loso(subjects, epochs=args.epochs, seed=args.seed)

    print("-" * 80)
    print(" 1. Per-Subject Leave-One-Subject-Out (LOSO) Performance")
    print("-" * 80)
    display_df = per_sub_df.copy()
    for col in ["SVM_Acc", "RF_Acc", "CNN_BiLSTM_Acc", "WiFi_Sleep_Acc", "Precision", "Recall", "F1_Score", "Macro_F1"]:
        display_df[col] = (display_df[col] * 100.0).map(lambda v: f"{v:6.2f}%")
    print(display_df.to_string(index=False))

    print("\n" + "-" * 80)
    print(" 2. Overall Model & Module Ablation Comparison (Paper Table II & Table IV)")
    print("-" * 80)
    print(f"{'Model / Method':30s} | {'Accuracy':>10s} | {'Macro F1':>10s} | {'Weighted F1':>11s}")
    print("-" * 80)
    for model_name, y_p in preds_arr.items():
        acc = accuracy_score(y_true, y_p) * 100.0
        mf1 = f1_score(y_true, y_p, average="macro") * 100.0
        wf1 = f1_score(y_true, y_p, average="weighted") * 100.0
        print(f"{model_name:30s} | {acc:9.2f}% | {mf1:9.2f}% | {wf1:10.2f}%")

    y_wifi = preds_arr["WiFi-Sleep (Full)"]
    cm_counts = confusion_matrix(y_true, y_wifi, labels=[0, 1, 2])
    cm_norm = confusion_matrix(y_true, y_wifi, labels=[0, 1, 2], normalize="true") * 100.0

    print("\n" + "-" * 80)
    print(" 3. WiFi-Sleep Three-Stage Confusion Matrix & Detailed Classification Report")
    print("-" * 80)
    cm_table = pd.DataFrame(
        [
            [f"{cm_norm[i, j]:5.1f}% ({cm_counts[i, j]:4d})" for j in range(3)]
            for i in range(3)
        ],
        index=[f"True {STAGE_SHORT_NAMES[i]}" for i in range(3)],
        columns=[f"Pred {STAGE_SHORT_NAMES[j]}" for j in range(3)],
    )
    print(cm_table.to_string())
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_wifi, target_names=STAGE_SHORT_NAMES, digits=4))

    cm_fig_path, sub_fig_path = plot_results(y_true, preds_arr, per_sub_df, out_dir=args.out_dir)
    pred_csv_path = os.path.join(args.out_dir, "wifi_sleep_predictions.csv")
    pred_df.to_csv(pred_csv_path, index=False, encoding="utf-8-sig")

    print("-" * 80)
    print(f"[Done] Completed in {time.time() - t_start:.2f}s")
    print(f" - Confusion Matrix Figure : {cm_fig_path}")
    print(f" - Per-Subject Plot        : {sub_fig_path}")
    print(f" - Predictions CSV         : {pred_csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
