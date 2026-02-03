import numpy as np
import csiread
from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler

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
            
            # Z-score 標準化
            standardized_seg = self.scaler.fit_transform(seg)
            segments.append(standardized_seg)

            # print(f"平均值 (應接近 0): {standardized_seg.mean():.2f}")
            # print(f"標準差 (應為 1): {standardized_seg.std():.2f}")
            
        return np.array(segments)

file_list = ["csi_10to20.dat", "csi_20to30.dat", "csi_30to40.dat", "csi_40to50.dat"]

csifile = "Training/csi_10to20.dat"
csidata = csiread.Intel(csifile, nrxnum=3, ntxnum=2, pl_size=10)
csidata.read()
csi = csidata.get_scaled_csi() # (N, 天線對, 子載波)
print(csidata.csi.shape)

# 標記時雖然不一定要標準化，但訓練前一定要做
# pipeline = CSIPipeline(fs=50, window_len=250, overlap_len=125)
# x_data = pipeline.process(csi_amplitude) #!!

# print(f"處理後的資料形狀: {x_data.shape}") 
# 預期輸出: (片段數, 250, 30) -> 這就是直接餵給機器學習模型的格式！



# np.abs(csi)
# np.angle(csi)

# # 標準化
# from sklearn.preprocessing import StandardScaler
# import numpy as np

# def standardize_segments(segments):
#     """
#     segments shape: (N_segments, window_size, feature_dim)
#     例如: (100, 150, 30) -> 100個片段, 每片150點, 30個子載波
#     """
#     standardized_data = []
#     scaler = StandardScaler()
    
#     for seg in segments:
#         # seg 的形狀是 (150, 30)
#         # 對這個片段進行標準化
#         scaled_seg = scaler.fit_transform(seg)
#         standardized_data.append(scaled_seg)
        
#     return np.array(standardized_data)

# # 使用範例
# # x_train_scaled = standardize_segments(x_train)

# # 濾波
# def butter_lowpass_filter(data, cutoff=5, fs=50, order=5):
#     """
#     data: 原始 CSI 振幅
#     cutoff: 截止頻率 (5Hz)
#     fs: 採樣率 (50Hz)
#     order: 濾波器階數 (越高過濾越乾淨，但會延遲)
#     """
#     nyq = 0.5 * fs
#     normal_cutoff = cutoff / nyq
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     y = lfilter(b, a, data)
#     return y

# # 假設 amplitude 是你解析後的 (N_packets,) 振幅數據
# filtered_amplitude = butter_lowpass_filter(amplitude)

# # 或者是更簡單的移動平均 (Moving Average)
# def moving_average(data, window_size=10):
#     return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# # 繪圖對比
# import matplotlib.pyplot as plt
# plt.plot(amplitude, label='Original', alpha=0.5)
# plt.plot(filtered_amplitude, label='Filtered (Low-pass)', linewidth=2)
# plt.legend()
# plt.title("50Hz CSI Filtering for Roll Detection")
# plt.show()