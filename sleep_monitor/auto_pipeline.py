import os
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


# 根據自己的電腦環境修改以下 5 個路徑
# 1. 真實 Samba 監聽目錄 (例如掛載的 Z 槽，或網路 UNC 路徑如 r"\\192.168.1.100\csi_share")
SAMBA_WATCH_DIR = r"C:\Users\Admin\OneDrive\桌面\wifi_sensing\WiFi_sensing\Mock_Samba" 

# 2. Python 專案根目錄 (存放 sleep_monitor 與 import_real_data.py 的地方)
PROJECT_ROOT_DIR = r"C:\Users\Admin\OneDrive\桌面\wifi_sensing\WiFi_sensing\sleep_monitor"

# 3. 本機 MATLAB 執行檔路徑 (R2015b)
MATLAB_EXE_PATH = r"C:\Program Files\MATLAB\R2015b\bin\matlab.exe"

# 4. MATLAB 睡眠演算法腳本所在的資料夾路徑
MATLAB_SCRIPT_DIR = r"C:\Users\Admin\OneDrive\Documents\MATLAB\WiFi_sensing\MATLAB\Frequency_Calculation_2015"

# 5. 負責把 CSV 資料上傳到 Aiven 雲端的 Python 檔名
PYTHON_IMPORT_SCRIPT = "import_real_data.py"
# ==============================================================================


# ──  自動化核心工作流 ────────────────────────────────────────────────────────
def trigger_pipeline(file_path):
    try:
        print("\n" + "="*65)
        print(f"[事件觸發] 偵測到硬體傳入新訊號檔: {os.path.basename(file_path)}")
        print("啟動全自動 Pipeline...")
        print("="*65)

        # 【步驟 A】：背景呼叫 MATLAB R2015b 跑睡眠演算法
        print("\n[Step A] 正在背景呼叫 MATLAB R2015b 進行演算法運算...")
        matlab_cmd = f'"{MATLAB_EXE_PATH}" -nosplash -nodesktop -r "cd(\'{MATLAB_SCRIPT_DIR}\'); process_csi_signals; exit;"'
        
        start_time = time.time()
        subprocess.run(matlab_cmd, shell=True, check=True)
        print(f"MATLAB 演算法運算完畢！成功輸出最新的 CSV 真實生理數據！(耗時: {round(time.time() - start_time, 1)}秒)")

        #【步驟 B】：自動觸發 Python 將真數據批次灌入 Aiven 雲端
        print("\n[Step B] MATLAB 輸出完畢！立刻觸發 Python 將真數據批次灌入 Aiven 雲端...")
        python_cmd = f"python \"{os.path.join(PROJECT_ROOT_DIR, PYTHON_IMPORT_SCRIPT)}\""
        
        subprocess.run(python_cmd, shell=True, check=True)
        print("\n" + "="*65)
        print("[Pipeline 完美落地] 真實硬體接收、MATLAB 計算到跨雲端同步全自動完成！")
        print("=" * 65)

    except subprocess.CalledProcessError as e:
        print(f"Pipeline 中途崩潰，命令執行失敗: {e}")
    except Exception as e:
        print(f"系統發生未知異常: {e}")


# ── Samba 資料夾監聽器 ──────────────────────────────────────────────────
class SambaFileHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_triggered = 0

    def on_created(self, event):
        # 排除資料夾，且只監聽特定硬體副檔名 (例如 .dat 或 .csv)，避免暫存檔干擾
        if not event.is_directory and event.src_path.endswith(('.dat', '.csv')):
            current_time = time.time()
            
            # 防彈跳機制：10秒內不重複觸發同一個檔案
            if current_time - self.last_triggered > 10:  
                self.last_triggered = current_time
                target_file = event.src_path
                
                print(f"\n 偵測到硬體建立檔案: {os.path.basename(target_file)}")
                print("【核心等待】開始監控硬體檔案寫入狀態，避免大檔案未傳輸完成即讀取...")

                # 核心防禦：每 2 秒檢查一次檔案大小，直到大小不再變動（代表硬體透過 Samba 傳完檔案了）才放行
                last_size = -1
                while True:
                    try:
                        current_size = os.path.getsize(target_file)
                        if current_size == last_size and current_size > 0:
                            print(f"檔案透過 Samba 寫入完成！最終大小: {current_size} bytes。")
                            break
                        last_size = current_size
                    except FileNotFoundError:
                        print("檔案在寫入過程中消失。")
                        return
                    time.sleep(2) 
                
                # 檔案確認完全寫入，正式啟動
                trigger_pipeline(target_file)


if __name__ == "__main__":
    if not os.path.exists(SAMBA_WATCH_DIR):
        os.makedirs(SAMBA_WATCH_DIR)

    print("=" * 65)
    print("  Wi-Fi CSI 睡眠監測 ── 真實 Samba 遠端自動化守護進程已啟動！")
    print(f"  目前正在實時監控路徑: {SAMBA_WATCH_DIR}")
    print("=" * 65)

    event_handler = SambaFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=SAMBA_WATCH_DIR, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n監聽服務安全關閉。")
        observer.stop()
    observer.join()