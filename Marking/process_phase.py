import os
import numpy as np
import csiread
import shutil
from scipy.signal import butter, filtfilt

# ================= 設定區域 =================
# 1. 檔案路徑
CSI_FILE_PATH = "data/flip50hzR_20s_001.dat"
OUTPUT_ROOT = "dataset_phase_output/50hz_001"  # 輸出資料夾名稱

def load_data(path):
    """讀取 CSI 數據"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到檔案: {path}")
    
    print(f"正在讀取: {path} ...")
    csi_data = csiread.Intel(path)
    csi_data.read()
    
    # 轉為幅度 (Amplitude)
    # 取第1個Tx, 第1個Rx (根據你的需求調整)
    amp = np.abs(csi_data.csi[:, :, 0, 0])
    
    print(f"原始數據形狀: {amp.shape}")
    return amp

def butter_lowpass_filter(data, cutoff=10, fs=100, order=5):
    """低通濾波器"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # axis=0 表示沿著時間軸濾波
    y = filtfilt(b, a, data, axis=0)
    return y

def normalize_data(data):
    """
    [新增] Min-Max 歸一化
    將數據縮放到 0 ~ 1 之間
    """
    # 加上 1e-8 是為了防止分母為 0
    d_min = data.min()
    d_max = data.max()
    
    print(f"執行歸一化: Min={d_min:.4f}, Max={d_max:.4f}")
    
    normalized = (data - d_min) / (d_max - d_min + 1e-8)
    return normalized

def save_slices(csi_matrix, intervals, window_size, stride, output_dir):
    """核心切片函數"""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    total_saved = 0
    
    for label, start_idx, end_idx in intervals:
        # 1. 建立資料夾
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        
        # 2. 邊界檢查
        if start_idx < 0 or end_idx > csi_matrix.shape[0]:
            print(f"[警告] 區間 {start_idx}-{end_idx} 超出範圍，跳過。")
            continue
            
        # 3. 提取數據
        action_data = csi_matrix[start_idx:end_idx]
        
        # 4. 滑動視窗
        num_packets = action_data.shape[0]
        
        if num_packets < window_size:
            print(f"[跳過] 動作 '{label}' 長度 ({num_packets}) 不足")
            continue

        count = 0
        for i in range(0, num_packets - window_size + 1, stride):
            segment = action_data[i : i + window_size]
            
            # 儲存
            filename = f"{label}_{start_idx}_{count:03d}.npy"
            save_path = os.path.join(label_dir, filename)
            np.save(save_path, segment)
            
            count += 1
            total_saved += 1
            
        print(f"動作 '{label}' ({start_idx}-{end_idx}): 已儲存 {count} 個切片。")

    print("-" * 30)
    print(f"處理完成！總共生成 {total_saved} 個樣本。")
    print(f"檔案存放在: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    try:
        # 1. 載入原始數據
        csi_amp = load_data(CSI_FILE_PATH)

        # 2. 濾波處理
        # 注意: cutoff=1 很低，通常人體動作建議在 10Hz 左右，除非你要濾掉非常高頻的雜訊
        print("執行濾波...")
        amp_filtered = butter_lowpass_filter(csi_amp, cutoff=1, fs=50) # 我將cutoff改為10Hz供參考，原本是1
        
        # 3. [新增] 歸一化處理
        # 這樣所有存下來的數據都會在 0~1 之間
        amp_normalized = normalize_data(amp_filtered)

        # 4. 執行切片與存檔 (注意這裡傳入的是 amp_normalized)
        save_slices(amp_normalized, LABEL_INTERVALS, WINDOW_SIZE, STRIDE, OUTPUT_ROOT)
        
    except Exception as e:
        print(f"發生錯誤: {e}")