import os
import numpy as np
import csiread
import shutil

# ================= 設定區域 =================
# 1. 檔案路徑
CSI_FILE_PATH = "0127data/flip50hzR_20s_009.dat"
OUTPUT_ROOT = "dataset_output/50hz_009"  # 輸出資料夾名稱

# 2. 切片參數 (最重要的超參數)
WINDOW_SIZE = 200   # 視窗大小 (例如 2秒 * 100Hz = 200封包)
STRIDE = 100        # 步長 (移動多少包切下一片，100代表重疊50%)


# ================= 設定區域 =================
# 3. 動作標籤定義 (格式: [標籤名, 開始封包Index, 結束封包Index])
# 根據 PCA 運動特徵圖精確對齊生成
LABEL_INTERVALS = [
    ("noise", 0, 200),          # 躺下
    ("flipping", 960, 1160),    # 右翻1 
    ("static", 1160, 1970),
    ("flipping", 1970, 2170),   # 翻回1 
    ("static", 2170, 2940),

    ("flipping", 2940, 3140),   # 右翻2 
    ("static", 3140, 4020),
    ("flipping", 4020, 4220),   # 翻回2 
    ("static", 4220, 4980),

    ("flipping", 4980, 5180),   # 右翻3 
    ("static", 5180, 6030),
    ("flipping", 6030, 6230),   # 翻回3 
    ("static", 6230, 6910),

    ("flipping", 6910, 7110),   # 右翻4 
    ("static", 7110, 7980),
    ("flipping", 7980, 8180),   # 翻回4 
    ("static", 8180, 8890),

    ("flipping", 8890, 9090),   # 右翻5 
    ("static", 9090, 9960),
    ("flipping", 9960, 10160),  # 翻回5 
    ("static", 10160, 10879),   # 靜止 
    ("noise", 10879, 11079),    # 起身 
 
]


# ===========================================

def load_data(path):
    """讀取 CSI 數據"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到檔案: {path}")
    
    print(f"正在讀取: {path} ...")
    csi_data = csiread.Intel(path)
    csi_data.read()
    
    # 轉為幅度 (Amplitude) 並處理形狀
    # 原始形狀通常是 (Packets, Nrx, Ntx, Subcarriers)
    # 我們這裡取絕對值轉為幅度，方便後續處理
    amp = np.abs(csi_data.csi[:, :, 0, 0])
    
    print(f"原始數據形狀: {amp.shape}")
    return amp

##
from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff=10, fs=100, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # data shape: (packets, subcarriers)
    y = filtfilt(b, a, data, axis=0)
    return y

def save_slices(csi_matrix, intervals, window_size, stride, output_dir):
    """
    核心切片函數
    """
    # 如果輸出目錄存在，先清空 (可選，避免舊資料混淆)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    total_saved = 0
    
    for label, start_idx, end_idx in intervals:
        # 1. 建立該動作的資料夾
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        
        # 2. 檢查索引邊界
        if start_idx < 0 or end_idx > csi_matrix.shape[0]:
            print(f"[警告] 區間 {start_idx}-{end_idx} 超出數據範圍，跳過。")
            continue
            
        # 3. 提取該區間的數據
        action_data = csi_matrix[start_idx:end_idx]
        
        # 4. 執行滑動視窗切片
        # 只有當區間長度大於視窗大小才切
        num_packets = action_data.shape[0]
        
        if num_packets < window_size:
            print(f"[跳過] 動作 '{label}' 長度 ({num_packets}) 小於視窗大小 ({window_size})")
            continue

        count = 0
        for i in range(0, num_packets - window_size + 1, stride):
            # 切出一個 window
            segment = action_data[i : i + window_size]
            
            # 檔名格式: label_原始起始點_切片序號.npy
            filename = f"{label}_{start_idx}_{count:03d}.npy"
            save_path = os.path.join(label_dir, filename)
            
            # 儲存
            np.save(save_path, segment)
            count += 1
            total_saved += 1
            
        print(f"動作 '{label}' ({start_idx}-{end_idx}): 已儲存 {count} 個切片。")

    print("-" * 30)
    print(f"處理完成！總共生成 {total_saved} 個樣本。")
    print(f"檔案存放在: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    try:
        # 1. 載入數據
        csi_amp = load_data(CSI_FILE_PATH)

        ##
        amp_filtered = butter_lowpass_filter(csi_amp, cutoff=1, fs=50)
        
        # 2. 執行切片與存檔
        save_slices(csi_amp, LABEL_INTERVALS, WINDOW_SIZE, STRIDE, OUTPUT_ROOT)
        
    except Exception as e:
        print(f"發生錯誤: {e}")