import matplotlib.pyplot as plt
import pandas as pd
import csiread
import numpy as np
from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler

# 設定
fs = 50
window_len = 250
overlap_len = 125
threshold = 1.5   # 能量閾值

file_list = ["flip50hzR_20s_001.dat"]

class CSIPipeline:
    def __init__(self, fs=50, cutoff=5, window_len=250, overlap_len=125):
        self.fs = fs                # 採樣率 (50Hz)
        self.cutoff = cutoff        # 低通濾波截止頻率 (5Hz)
        self.window_len = window_len # 視窗長度 (5秒 = 250點)
        self.overlap_len = overlap_len # 重疊長度 (50% = 125點)
        self.scaler = StandardScaler()

    def _butter_lowpass(self, data):
        """低通濾波：去除高頻雜訊"""
        nyq = 0.5 * self.fs #!!
        normal_cutoff = self.cutoff / nyq
        b, a = butter(5, normal_cutoff, btype='low', analog=False)
        return lfilter(b, a, data, axis=0)

    def process(self, raw_csi):
        """
        輸入 raw_csi: 形狀為 (封包數, 30個子載波) 的振幅資料
        """
        # 濾波
        filtered_data = self._butter_lowpass(raw_csi)
        
        # 切片
        segments = []
        step = self.window_len - self.overlap_len
        for i in range(0, len(filtered_data) - self.window_len + 1, step): #!!
            seg = filtered_data[i : i + self.window_len]
            
            # Z-score 標準化 #!!
            standardized_seg = self.scaler.fit_transform(seg)
            segments.append(standardized_seg)
            
        return np.array(segments)

def split_csi_data(csi_matrix, window_size, overlap):
    """
    csi_matrix: 形狀為 (封包數, 天線對, 子載波) 的矩陣
    window_size: 每個片段的封包數量
    overlap: 重疊的封包數量
    """
    segments = []
    start = 0
    while start + window_size <= len(csi_matrix):
        segment = csi_matrix[start:start + window_size]
        segments.append(segment)
        start += (window_size - overlap)
    
    return np.array(segments)

for filename in file_list:
    # 讀取
    csidata = csiread.Intel("0127data/" + filename)
    csidata.read()
    csi = csidata.get_scaled_csi() # (N, 30, Tx, Rx)
    # print(csi.shape)

    # 計算振幅
    amp_raw = np.abs(csi).mean(axis=(1, 2, 3)) # 跨時間、子載波、天線對維度取平均振幅

    # 濾波
    pipeline = CSIPipeline(fs=50, window_len=250, overlap_len=125)
    amp_filtered = pipeline._butter_lowpass(amp_raw)

    # 切分
    segments = split_csi_data(amp_filtered, window_len, overlap_len)
    print(f"檔案 {filename} 切分完成：{len(segments)} 片段")
    
    current_labels = []
    
    for i, seg in enumerate(segments):
        # 標準差
        std_val = np.std(seg)

        # 波動極小自動標記
        if std_val < threshold:
            current_labels.append({'source_file': filename, 'segment_id': i, 'label': 0})
            continue

        plt.figure(figsize=(8, 4))
        plt.plot(seg)
        plt.title(f"Segment {i} - Std: {std_val:.2f}")
        plt.ylim(np.min(amp_filtered), np.max(amp_filtered)) # 固定比例尺才好判斷
        plt.show()
        
        ans = input(f"片段 {i} (1:翻身, 0:靜止, s:跳過): ")

        if ans == 'q': break
        if ans == 's': continue
        
        current_labels.append({
            'source_file': filename,
            'segment_id': i,
            'label': int(ans)
        })
    
    # 存檔
    df = pd.DataFrame(current_labels)
    df.to_csv(filename.replace(".dat", "_labels.csv"), index=False)