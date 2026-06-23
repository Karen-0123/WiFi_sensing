import os
import time

#  設定跟剛剛一模一樣的模擬 Samba 資料夾路徑
MOCK_SAMBA_DIR = r"C:\Users\Admin\OneDrive\桌面\wifi_sensing\WiFi_sensing\Mock_Samba"
test_file_path = os.path.join(MOCK_SAMBA_DIR, "mock_hardware_signal_001.dat")


print("  硬體傳入訊號模擬工具 (Mock Samba Trigger)")
print("系統即將模擬硬體透過 Samba 共享，將 Wi-Fi CSI 原始訊號寫入妳的資料夾...")
print("準備倒數 3 秒...")

for i in range(3, 0, -1):
    print(f"  {i}...")
    time.sleep(1)

# 如果資料夾不存在，自動建立
if not os.path.exists(MOCK_SAMBA_DIR):
    os.makedirs(MOCK_SAMBA_DIR)

# 執行 Action：製造一個假檔案塞進去！
print("\n[Action] 檔案已成功寫入 Samba 共享資料夾！")
with open(test_file_path, "w", encoding="utf-8") as f:
    f.write("RAW_CSI_HEX_DATA_7F8031_SAMPLE_FROM_HARDWARE_ANTENNA")

print(f" 成功建立模擬訊號源: {os.path.basename(test_file_path)}")
print("\n請立刻查看另一個跑 auto_pipeline.py 的終端機視窗，奇蹟應該正在發生！")
print("====================================================")