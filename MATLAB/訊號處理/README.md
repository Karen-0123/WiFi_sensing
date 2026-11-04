# Wi-Fi CSI 呼吸率估測與《WiFi-Sleep》睡眠階段特徵分析 MATLAB 專案交接文件

## 1. 專案概述

本專案是一套基於 **Intel 5300 Wi-Fi CSI (Channel State Information)** 資料之**非接觸式生理監測與睡眠階段特徵提取分析系統**，完整實作了《**WiFi-Sleep: Sleep Stage Monitoring Using Commodity Wi-Fi Devices**》（*IEEE Internet of Things Journal*, Vol. 8, No. 18, Sept. 2021）論文之核心訊號處理與 13 維生理特徵工程架構。

系統讀取 Intel 5300 CSI Tool 產生的 `.dat` 檔案，萃取 `2 Tx × 3 Rx × 30 Subcarriers` 的 CSI 矩陣，透過以下核心技術完成單檔生理監測與整晚連續睡眠批次特徵匯出：

1. **CSI Ratio 相位與共模雜訊消除 (`process_csi_signal.m`)**：計算同一接收端相鄰天線對的複數商數（`Rx1/Rx2`、`Rx1/Rx3`、`Rx2/Rx3`），同時抵消隨機時變相位偏移（SFO/CFO/PBD）與網卡 AGC 共模振幅雜訊。
2. **MRC-PCA 子載波方向對齊與最大比合併 (`process_csi_signal.m`, `select_respiration_stream.m`)**：計算各子載波呼吸頻帶 SNR 作為 MRC 增益，經帶通濾波後以 PCA 第一主成分投影向量之正負號（`sign(coeff(:,1))`）修正不同 Fresnel Zone 子載波的反相問題，再加權合成高 SNR 呼吸波形，並結合振幅與相位互補資訊消除感測盲區（Blind Spots）。
3. **向量化 Doppler-MUSIC 都卜勒速度譜體動偵測 (`detect_rollover.m`)**：以 1 秒滑動視窗、0.1 秒步長建立空間–時間平滑相關矩陣，分解訊號與雜訊子空間以提取 `[-0.5, 0.5] m/s` 速度譜能量變異數，並以雙閾值（`ThreshPLM = 0.32`, `ThreshMBM = 4.7`）與持續時間規則區分 **MBM（Major Body Movement，翻身/大型體動）** 與 **PLM（Periodic Limb Movement，週期性肢體微動）**，再以高斯核加權移動平均（Gaussian WMA）計算發生頻率。
4. **呼吸波峰/波谷偵測與強化波形特徵提取 (`detect_respiration_peaks.m`)**：排除掉包與體動區間，使用穩健 MAD 自適應門檻與 0.75 秒半週期鄰域校驗偵測吸氣波峰（Peaks）與吐氣波谷（Valleys），提取論文定義之 **呼吸深度變異度（Respiration Depth Variance）**、**吸氣時間佔比（FIT）** 與 **吸吐氣時間比（I/E Ratio）** 及其變異數。
5. **自相關函數（ACF）呼吸率估測與體動切除線性插值 (`calculate_dynamic_bpm.m`)**：在 20 秒滑動視窗內使用 ACF 第一週期峰搭配拋物線次樣本內插計算動態呼吸率（BPM），並自動切除體動污染區間（Cut）後以線性插值（Linear Interpolation）補齊。
6. **整晚高階多項式去趨勢與 20 欄特徵 CSV 匯出 (`calculate_breathing_variability.m`, `calculate_nrem_baseline.m`, `export_sleep_features.m`)**：針對整晚跨區段呼吸率擬合高階多項式基線去趨勢，計算平滑呼吸率、一階導數、四分位距（IQR）、呼吸變異度與 NREM 基線偏差，匯出相容原有 9 欄並擴充論文完整特徵之 20 欄 CSV 檔案。

---

## 2. 程式入口

本專案提供兩個主要執行入口（皆為 MATLAB Script）：

### 入口一：單檔測試與視覺化分析 (`CSI_Single_Test.m`)

用於單一 `.dat` 檔案的完整演算法驗證與圖表檢視：

```matlab
filename = 'C:\Users\fupei\Desktop\csi\data\0515_test_video\test_video002.dat';
Fs_orig = 200;
Fs_target = 40;
```

### 入口二：整晚睡眠批次分析與特徵匯出 (`CSI_Sleep_Batch_Analysis.m`)

用於整晚連續 3 分鐘區段檔案（`*seg*.dat`）的批次特徵提取與 CSV 匯出：

```matlab
data_folder = 'C:\Users\fupei\Desktop\csi\data\sleep\sleep013_200hz_360min_0803';
Fs_orig = 200;
Fs_target = 40;
my_filename = 'subject013_features.csv'; % 匯出供 ML 使用之 CSV 檔名
```

---

## 3. 專案檔案清單與職責說明

| 檔案名稱 | 模組分層 | 角色與功能說明 |
|---|---|---|
| `CSI_Single_Test.m` | 主入口 (單檔) | 串接單一 `.dat` 檔之讀取、重採樣、CSI Ratio + MRC-PCA、Doppler-MUSIC 體動偵測、ACF 呼吸率計算與多圖表視覺化 |
| `CSI_Sleep_Batch_Analysis.m` | 主入口 (批次) | 批次讀取資料夾內所有 `*seg*.dat`，提取每段 (180s) 呼吸與體動特徵，最後執行整晚多項式去趨勢與基線偏差計算並匯出 20 欄 CSV |
| `read_intel5300_dat.m` | 資料讀取層 | 呼叫外部 `read_bf_file` 讀取 `.dat`，篩選 `2×3×30` 封包、展開 `timestamp_low` 溢位並轉為相對秒數 |
| `resample_csi_data.m` | 前處理層 | 剔除重複時間戳、對複數 CSI 實虛部進行抗混疊低通濾波、線性插值至 `40 Hz` 均勻時間軸，並建立掉包遮罩 `gap_mask` |
| `process_csi_signal.m` | 特徵提取層 (主用) | 執行接收天線對 CSI Ratio 相除、跨全子載波呼吸 SNR 估計、動態視窗 Savitzky-Golay 濾波 (`sg_window_length`)、MRC-PCA 方向對齊加權融合與最佳 Tx 選擇 |
| `process_csi_signal_1.m` | 特徵提取層 (歷史備份) | 舊版特徵提取備份檔，僅作歷史紀錄保留 |
| `detect_rollover.m` | 體動與睡眠分支 | 向量化 Doppler-MUSIC 速度譜 (`-0.5~0.5 m/s`) 能量變異數計算，以雙閾值區分 MBM（翻身）與 PLM（微體動）並計算 Gaussian WMA 頻率 |
| `detect_sleep_state.m` | 體動與睡眠分支 | 根據區段內翻身事件的總時間佔比 (`motion_ratio < 0.08`) 與發生次數 (`num_events < 2`)，判定該區段為 `Sleep (1)` 或 `Awake (0)` |
| `select_respiration_stream.m` | 呼吸率估測分支 | 在 `Amp-PC1`、`Phase-PC1` 與 `Combined-MRC-PCA` 串流中，排除體動區段後以正規化 PSD 方差 × Prominence × SNR 選出最佳呼吸訊號 |
| `detect_respiration_peaks.m` | 呼吸波形特徵分支 | 排除掉包與體動區間，偵測吸氣波峰與吐氣波谷，並計算呼吸深度變異度 (`resp_depth_var`)、吸氣時間佔比 (`FIT`) 與吸吐比 (`I/E Ratio`) |
| `calculate_dynamic_bpm.m` | 呼吸率估測分支 | 以 20 秒滑動視窗執行 ACF 自相關函數（含次樣本拋物線內插）與 P2P 驗證計算動態 BPM，切除體動污染區間並以線性插值補齊 |
| `calculate_breathing_variability.m` | 整晚統計匯出層 | 對各區段 BPM 執行 `pchip` 插值、整晚高階多項式基線擬合去趨勢，計算呼吸率變異度、平滑呼吸率、一階導數與四分位距 (IQR) |
| `calculate_nrem_baseline.m` | 整晚統計匯出層 | 對整晚所有有效 BPM (`5–40 bpm`) 建立 35-bin 直方圖，取眾數 (Mode) 作為整晚 NREM 呼吸基線 `baseline_bpm` |
| `export_sleep_features.m` | 整晚統計匯出層 | 匯出保留原有 9 欄並擴充《WiFi-Sleep》論文 11 項強化特徵（共 20 欄）之 CSV 檔案 |

---

## 4. 匯出 CSV 特徵欄位規格 (`export_sleep_features.m`)

匯出的 CSV 檔案共包含 **20 個欄位**（前 9 欄與既有流程完全相容，第 10～20 欄補齊《WiFi-Sleep》論文全部 13 項呼吸與體動生理特徵）：

| 欄位序號 | 欄位名稱 | 類別 | 說明 |
|---|---|---|---|
| 1 | `Start_Time` | 時間資訊 | 區段起始時間 (`yyyy-mm-dd HH:MM:ss`) |
| 2 | `End_Time` | 時間資訊 | 區段結束時間 (`yyyy-mm-dd HH:MM:ss`) |
| 3 | `Wake_Sleep` | 狀態判別 | 清醒/睡眠初步判別 (`1` = Sleep, `0` = Awake) |
| 4 | `Breathing_Rate_Deviation` | 基本呼吸特徵 | 呼吸率基線偏差 `|all_90th_percentile - baseline_bpm|` |
| 5 | `Breathing_Rate_Variability` | 基本呼吸特徵 | 去趨勢後之區段呼吸率變異數 ($\text{bpm}^2$) |
| 6 | `Num_Events` | 基本體動特徵 | 區段內大型體動/翻身 (MBM) 事件次數 |
| 7 | `all_90th_percentile` | 基本呼吸特徵 | 區段有效動態 BPM 之第 90 百分位數 |
| 8 | `variance` | 基本呼吸特徵 | 區段插值後 BPM 之原始未去趨勢變異數 |
| 9 | `Sleep_Stage` | 標註欄位 | 睡眠階段標籤（預設填 `0` 供後續標註或模型推論） |
| 10 | `Resp_Rate_Mean` | 論文呼吸特徵 | 區段平均呼吸率 (BPM) |
| 11 | `Resp_Rate_Smoothed` | 論文呼吸特徵 | 整晚高階多項式基線去趨勢後之平滑呼吸率 (BPM) |
| 12 | `Resp_Rate_Derivative` | 論文呼吸特徵 | 平滑呼吸率之一階時間導數 (BPM/min) |
| 13 | `Resp_Rate_IQR` | 論文呼吸特徵 | 區段呼吸率四分位距 (75th - 25th percentile, BPM) |
| 14 | `Resp_Depth_Variance` | 論文強化波形特徵 | 30 秒穩定子視窗正規化吸氣波峰深度變異數 |
| 15 | `FIT_Mean` | 論文強化波形特徵 | 吸氣時間佔比平均值 $T_{\text{in}} / (T_{\text{in}} + T_{\text{ex}})$ |
| 16 | `FIT_Variance` | 論文強化波形特徵 | 吸氣時間佔比變異數 |
| 17 | `IE_Ratio_Mean` | 論文強化波形特徵 | 吸吐氣時間比平均值 $T_{\text{in}} / T_{\text{ex}}$ |
| 18 | `IE_Ratio_Variance` | 論文強化波形特徵 | 吸吐氣時間比變異數 |
| 19 | `MBM_Frequency` | 論文體動特徵 | 大型體動/翻身 (MBM) 高斯加權移動平均頻率 (次/分鐘) |
| 20 | `PLM_Frequency` | 論文體動特徵 | 週期性肢體微動 (PLM) 高斯加權移動平均頻率 (次/分鐘) |
