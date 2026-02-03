import csiread
import matplotlib.pyplot as plt
import numpy as np

def process_intel5300_csi(file_path, sampling_rate=5):
    # 1. 使用 Intel 5300 專用解碼器
    csidata = csiread.Intel(file_path)
    csidata.read()
    
    # 2. 提取 CSI 並轉換為振幅 (Amplitude)
    # 原始數據是複數，np.abs 會幫我們算 sqrt(real^2 + imag^2)
    # csi 結構: (封包數, 天線數, 子載波數)
    csi_matrix = csidata.get_scaled_csi()
    amplitude = np.abs(csi_matrix)
    
    # 3. 降維：為了觀察，我們取「第一根天線」所有「子載波」的平均值
    # 這樣可以減少雜訊，看到整體的能量變化
    avg_amplitude = np.mean(amplitude[:, 0, :], axis=1)
    
    # 4. 建立時間軸
    time = np.arange(len(avg_amplitude)) / sampling_rate
    
    # 5. 繪圖
    plt.figure(figsize=(12, 6))
    plt.plot(time, avg_amplitude, label='Average Amplitude (Antenna 0)', color='#1f77b4')
    
    # 標註你提到的 10-20 秒
    plt.axvspan(10, 20, color='red', alpha=0.2, label='Movement Block (10s-20s)')
    
    plt.title("Intel 5300 CSI Visualization")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 6. (選做) 儲存成 Numpy 供後續 ML 使用
    # np.save("processed_csi.npy", amplitude)
    # print("資料已存為 processed_csi.npy，形狀為:", amplitude.shape)

# 執行
process_intel5300_csi('csi_10to20.dat', sampling_rate=5)