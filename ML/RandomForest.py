import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedGroupKFold,
    GridSearchCV,
    cross_val_predict,
)
from sklearn.metrics import f1_score, make_scorer, classification_report
from sklearn.inspection import permutation_importance

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

RANDOM_STATE = 42
FILE_PATH = "full_sleep_features_merged.csv"

# 【新增】把平滑化變成可切換開關，取代手動改程式碼重跑
APPLY_SMOOTHING = True
SMOOTHING_WINDOW = 3

# 【修正核心 Bug】門檻搜尋要優化的目標：改成可選，預設 macro，
THRESHOLD_SCORING = "macro"  # "macro" 或 "rem"

# ---------------------------------------------------------------------------
# 1. 資料載入 + 特徵工程（邏輯不變，僅統一縮排）
# ---------------------------------------------------------------------------
def load_and_engineer_features(file_path=FILE_PATH, drop_cols=None):
    """
    載入資料並進行特徵工程
    :param file_path: CSV 檔案路徑
    :param drop_cols: List[str]，手動指定要剔除的特徵欄位名稱清單
    """
    df = pd.read_csv(file_path)

    filtered_df = df[df["Sleep_Stage"].isin([1, 2]) | df["Sleep_Stage"].isna()].copy()

    has_events = "Num_Events" in filtered_df.columns
    has_progress = "Sleep_Progress" in filtered_df.columns

    sort_cols = ["Subject_ID"] + (["Sleep_Progress"] if has_progress else [])
    filtered_df = filtered_df.sort_values(sort_cols).reset_index(drop=True)

    g = filtered_df.groupby("Subject_ID")

    roll_windows = [5, 10, 15]
    for w in roll_windows:
        for col in ["Breathing_Rate_Deviation", "Breathing_Rate_Variability"]:
            filtered_df[f"{col}_roll_mean_{w}"] = g[col].transform(
                lambda s: s.rolling(w, min_periods=1).mean()
            )
            filtered_df[f"{col}_roll_std_{w}"] = g[col].transform(
                lambda s: s.rolling(w, min_periods=1).std()
            )
            filtered_df[f"{col}_roll_range_{w}"] = g[col].transform(
                lambda s: s.rolling(w, min_periods=1).max()
                - s.rolling(w, min_periods=1).min()
            )

        if has_events:
            filtered_df[f"Num_Events_roll_sum_{w}"] = g["Num_Events"].transform(
                lambda s: s.rolling(w, min_periods=1).sum()
            )

    for col in ["Breathing_Rate_Deviation", "Breathing_Rate_Variability"]:
        for step in [1, 2, -1, -2]:
            tag = f"lag_{step}" if step > 0 else f"lead_{abs(step)}"
            filtered_df[f"{col}_{tag}"] = g[col].shift(step)

    for col in ["Breathing_Rate_Deviation", "Breathing_Rate_Variability"]:
        filtered_df[f"{col}_diff1"] = g[col].diff()

    filtered_df["Dev_x_Var"] = (
        filtered_df["Breathing_Rate_Deviation"]
        * filtered_df["Breathing_Rate_Variability"]
    )

    if has_progress:
        filtered_df["Progress_x_Var"] = (
            filtered_df["Sleep_Progress"] * filtered_df["Breathing_Rate_Variability"]
        )
        filtered_df["Progress_sin"] = np.sin(
            2 * np.pi * filtered_df["Sleep_Progress"] * 5
        )
        filtered_df["Progress_cos"] = np.cos(
            2 * np.pi * filtered_df["Sleep_Progress"] * 5
        )

    valid_mask = (
        (filtered_df["Wake_Sleep"] == 1)
        & (filtered_df["Sleep_Stage"].isin([1, 2]))
        & (filtered_df["Breathing_Rate_Deviation"].notna())
    )
    final_df = filtered_df[valid_mask].copy()

    exclude_cols = ["Subject_ID", "Sleep_Stage", "Wake_Sleep", "Start_Time", "End_Time"]
    drop_cols = drop_cols or []
    ignore_set = set(exclude_cols + drop_cols)
    feature_cols = [c for c in final_df.columns if c not in ignore_set]

    final_df[feature_cols] = final_df[feature_cols].fillna(0)

    print(
        f"--> [特徵工程完成] 原始總欄位數: {final_df.shape[1]}, 最終使用特徵數: {len(feature_cols)}"
    )
    if drop_cols:
        print(f"--> 已手動剔除 {len(drop_cols)} 個特徵: {drop_cols}")

    return final_df, feature_cols


# ---------------------------------------------------------------------------
# 2. 後處理：依受試者單獨平滑化
# ---------------------------------------------------------------------------
def smooth_predictions_by_subject(df_val, preds, window_size=3):
    df_val = df_val.copy()
    df_val["pred"] = preds
    smoothed_preds = []

    for _, sub_df in df_val.groupby("Subject_ID", sort=False):
        sub_preds = sub_df["pred"].values
        sub_smoothed = np.copy(sub_preds)
        half_w = window_size // 2
        n = len(sub_preds)

        for i in range(n):
            start = max(0, i - half_w)
            end = min(n, i + half_w + 1)
            sub_smoothed[i] = mode(sub_preds[start:end], keepdims=False)[0]

        smoothed_preds.extend(sub_smoothed)

    return np.array(smoothed_preds)

# ---------------------------------------------------------------------------
# 3. Leakage-free 決策門檻搜尋
#    【修正核心 Bug】新增 scoring 參數：可選 "macro" 或 "rem"，
#    不再寫死只優化 REM F1。
# ---------------------------------------------------------------------------
def tune_threshold_leakage_free(
    fitted_pipe,
    X_train,
    y_train,
    groups_train,
    inner_cv,
    pos_label=1,
    scoring="macro",
    thresholds=None,
):
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.01)

    classes = fitted_pipe.classes_
    pos_idx = list(classes).index(pos_label)
    other_label = [c for c in classes if c != pos_label][0]

    oof_proba = cross_val_predict(
        fitted_pipe,
        X_train,
        y_train,
        groups=groups_train,
        cv=inner_cv,
        method="predict_proba",
        n_jobs=None,
    )[:, pos_idx]

    best_thr, best_score = 0.5, -1.0
    for thr in thresholds:
        y_pred_thr = np.where(oof_proba >= thr, pos_label, other_label)
        if scoring == "macro":
            score = f1_score(y_train, y_pred_thr, average="macro", zero_division=0)
        else:  # "rem"
            score = f1_score(
                y_train, y_pred_thr, pos_label=pos_label, zero_division=0
            )
        if score > best_score:
            best_score, best_thr = score, thr

    return best_thr, best_score

# ---------------------------------------------------------------------------
# 4. 特徵重要性篩選（診斷用途，非本折內選擇依據）
#    【修正】原本定義了但沒被呼叫；現在在 main() 裡實際使用它。
# ---------------------------------------------------------------------------
def report_top_features(best_model, X_val, y_val, feature_cols, scorer, top_k=20):
    result = permutation_importance(
        best_model,
        X_val,
        y_val,
        scoring=scorer,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    order = np.argsort(result.importances_mean)[::-1][:top_k]
    return [(feature_cols[i], result.importances_mean[i]) for i in order]


def plot_and_get_feature_importance(X_df, y, feature_names, top_k=20, lgb_params=None):
    """使用 LightGBM 計算全資料集特徵重要性並繪圖（純診斷用，不用於模型選擇）"""
    if lgb_params is None:
        lgb_params = {
            "random_state": RANDOM_STATE,
            "is_unbalance": True,
            "n_estimators": 300,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "verbosity": -1,
        }

    model = LGBMClassifier(**lgb_params)
    model.fit(X_df, y)

    importances = model.feature_importances_
    feat_imp = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 8))
    top_feats = feat_imp.head(top_k)[::-1]
    plt.barh(top_feats["Feature"], top_feats["Importance"], color="skyblue")
    plt.xlabel("Feature Importance (Split Count)")
    plt.title(f"Top {top_k} Important Features (LightGBM)")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    zero_imp_feats = feat_imp[feat_imp["Importance"] == 0]["Feature"].tolist()
    low_imp_feats = feat_imp[feat_imp["Importance"] < 5]["Feature"].tolist()

    print("\n" + "=" * 50)
    print(f"【特徵重要性分析】總特徵數: {len(feature_names)}")
    print(f"前 {top_k} 大關鍵特徵:")
    print(feat_imp.head(top_k).to_string(index=False))
    print(f"\n零貢獻度特徵 (Importance == 0) 數量: {len(zero_imp_feats)}")
    if zero_imp_feats:
        print(f"建議剔除的零貢獻欄位: {zero_imp_feats}")
    print(f"\n極低貢獻度特徵 (Importance < 5) 數量: {len(low_imp_feats)}")
    print("=" * 50 + "\n")

    return feat_imp


# ---------------------------------------------------------------------------
# 5. 主流程
# ---------------------------------------------------------------------------
def main():
    features_to_drop = []  # 【建議】特徵剔除跟門檻調校分開實驗，先留空、單獨測試
    final_df, feature_cols = load_and_engineer_features(
        file_path=FILE_PATH, drop_cols=features_to_drop
    )

    X = final_df[feature_cols].values
    y = final_df["Sleep_Stage"].values.astype(int)
    groups = final_df["Subject_ID"].values

    print(
        f"總樣本數: {len(y)}, 特徵數: {len(feature_cols)}, 受試者/夜數: {len(np.unique(groups))}"
    )
    print(f"類別分布 -> REM(1): {(y==1).sum()}, NREM(2): {(y==2).sum()}\n")
    print(f"[設定] 平滑化: {APPLY_SMOOTHING} | 門檻搜尋目標: {THRESHOLD_SCORING}\n")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    pipe = ImbPipeline([
        ("sampler", SMOTE(random_state=RANDOM_STATE)),
        ("rf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)),
    ])

    param_grid = [
        {
            "sampler": [
                None,  # 對照組：完全不採樣，讓 class_weight 消融自己說話
                SMOTE(random_state=RANDOM_STATE, k_neighbors=5),
                BorderlineSMOTE(random_state=RANDOM_STATE, k_neighbors=5, sampling_strategy=0.7),
            ],
            "rf__n_estimators": [150, 300],
            "rf__max_depth": [6, 10],
            "rf__min_samples_leaf": [2, 5],
            "rf__max_features": ["sqrt"],
            # 【修正核心 Bug】恢復消融，不再預設疊加 balanced
            "rf__class_weight": [None, "balanced"],
        }
    ]

    macro_f1_scorer = make_scorer(f1_score, average="macro")

    y_true_all, y_pred_all = [], []
    fold_thresholds = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        groups_train = groups[train_idx]
        df_val = final_df.iloc[val_idx]

        inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

        grid = GridSearchCV(
            pipe, param_grid, scoring=macro_f1_scorer, cv=inner_cv, n_jobs=-1
        )
        grid.fit(X_train, y_train, groups=groups_train)
        best_model = grid.best_estimator_

        best_thr, oof_score = tune_threshold_leakage_free(
            best_model,
            X_train,
            y_train,
            groups_train,
            inner_cv,
            pos_label=1,
            scoring=THRESHOLD_SCORING,
        )
        fold_thresholds.append(best_thr)

        classes = best_model.classes_
        pos_idx = list(classes).index(1)
        other_label = [c for c in classes if c != 1][0]
        val_proba = best_model.predict_proba(X_val)[:, pos_idx]
        preds = np.where(val_proba >= best_thr, 1, other_label)

        if APPLY_SMOOTHING:
            preds = smooth_predictions_by_subject(df_val, preds, window_size=SMOOTHING_WINDOW)

        y_true_all.extend(y_val)
        y_pred_all.extend(preds)

        fold_macro_f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        fold_rem_f1 = f1_score(y_val, preds, pos_label=1, zero_division=0)
        print(
            f"[Fold {fold}] Val 樣本數={len(y_val)} | "
            f"sampler={type(grid.best_params_.get('sampler')).__name__} | "
            f"class_weight={grid.best_params_.get('rf__class_weight')} | "
            f"門檻={best_thr:.2f}(OOF {THRESHOLD_SCORING} F1={oof_score:.4f}) | "
            f"Val Macro F1={fold_macro_f1:.4f} | Val REM F1={fold_rem_f1:.4f}"
        )

        # 【修正】原本 report_top_features 從未被呼叫；這裡實際使用它做診斷輸出
        top_feats = report_top_features(
            best_model, X_val, y_val, feature_cols, macro_f1_scorer, top_k=10
        )
        print(f"  該折 Top 10 特徵: {[f for f, _ in top_feats]}")

    print(f"\n===== 5-Fold 彙整分類報告（門檻目標={THRESHOLD_SCORING}, 平滑化={APPLY_SMOOTHING}）=====")
    print(classification_report(y_true_all, y_pred_all, digits=4))
    print(f"各折最佳門檻: {[round(t, 2) for t in fold_thresholds]}")
    print(
        f"門檻離散程度: min={min(fold_thresholds):.2f}, "
        f"median={np.median(fold_thresholds):.2f}, max={max(fold_thresholds):.2f}"
    )

    # ------------------------------------------------------------------
    # LightGBM 對照組：套用「同一套」leakage-free 門檻搜尋，
    # 【修正】不再讓 LightGBM 享有比 RF 更少的優化，比較才公平。
    # ------------------------------------------------------------------
    if HAS_LGBM:
        print("\n===== LightGBM 對照組（同一 CV 骨架 + 同一套門檻搜尋） =====")
        lgbm_pipe = ImbPipeline([
            ("sampler", BorderlineSMOTE(random_state=RANDOM_STATE, k_neighbors=5, sampling_strategy=0.7)),
            (
                "lgbm",
                LGBMClassifier(
                    random_state=RANDOM_STATE,
                    is_unbalance=True,
                    n_estimators=500,
                    num_leaves=31,
                    learning_rate=0.05,
                    verbosity=-1,
                ),
            ),
        ])
        y_true_lgbm, y_pred_lgbm = [], []
        lgbm_thresholds = []

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            groups_train = groups[train_idx]
            df_val = final_df.iloc[val_idx]

            lgbm_pipe.fit(X_train, y_train)

            inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
            best_thr, _ = tune_threshold_leakage_free(
                lgbm_pipe, X_train, y_train, groups_train, inner_cv,
                pos_label=1, scoring=THRESHOLD_SCORING,
            )
            lgbm_thresholds.append(best_thr)

            classes = lgbm_pipe.classes_
            pos_idx = list(classes).index(1)
            other_label = [c for c in classes if c != 1][0]
            val_proba = lgbm_pipe.predict_proba(X_val)[:, pos_idx]
            preds = np.where(val_proba >= best_thr, 1, other_label)

            if APPLY_SMOOTHING:
                preds = smooth_predictions_by_subject(df_val, preds, window_size=SMOOTHING_WINDOW)

            y_true_lgbm.extend(y_val)
            y_pred_lgbm.extend(preds)

        print(classification_report(y_true_lgbm, y_pred_lgbm, digits=4))
        print(f"LightGBM 各折門檻: {[round(t, 2) for t in lgbm_thresholds]}")

        plot_and_get_feature_importance(final_df[feature_cols], y, feature_cols, top_k=20)
    else:
        print("\n(未安裝 lightgbm，略過對照組；pip install lightgbm 後可啟用)")


if __name__ == "__main__":
    main()