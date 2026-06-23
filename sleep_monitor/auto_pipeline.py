
import os
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ── 1. 模擬測試路徑設定（對齊本機桌面環境） ──────────────────
SAMBA_WATCH_DIR = r"C:\Users\Admin\OneDrive\桌面\wifi_sensing\WiFi_sensing\Mock_Samba"
PROJECT_ROOT_DIR = r"C:\Users\Admin\OneDrive\桌面\wifi_sensing\WiFi_sensing\sleep_monitor"
MATLAB_EXE_PATH = r"C:\Program Files\MATLAB\R2015b\bin\matlab.exe"
MATLAB_SCRIPT_DIR = r"C:\Users\Admin\OneDrive\Documents\MATLAB\WiFi_sensing\MATLAB\Frequency_Calculation_2015"
PYTHON_IMPORT_SCRIPT = "import_real_data.py"


# ── 2. 自動化核心工作流（事件驅動核心） ──────────────────────────
def trigger_pipeline():
    try:
        print("\n" + "="*65)
        print("[事件觸發] 偵測到模擬 Samba 丟入新訊號檔！啟動全自動 Pipeline...")
        print("="*65)

        # 【步驟 A】：背景呼叫 MATLAB 跑睡眠演算法
        print("\nStep A] 正在背景呼叫 MATLAB R2015b 進行演算法運算...")
        
        
        # 跑完自動 exit; 釋放妳的電腦記憶體！
        matlab_cmd = f'"{MATLAB_EXE_PATH}" -nosplash -nodesktop -r "cd(\'{MATLAB_SCRIPT_DIR}\'); process_csi_signals; exit;"'
        
        start_time = time.time()
        subprocess.run(matlab_cmd, shell=True, check=True)
        print(f"MATLAB 演算法運算完畢！成功輸出最新的 CSV 真實生理數據！(耗時: {round(time.time() - start_time, 1)}秒)")

        # 🏃 【步驟 B】：自動觸發  Python傳送到 Aiven 雲端
        print("\n[Step B] MATLAB 輸出完畢！立刻觸發 Python 將真數據批次灌入 Aiven 雲端...")
        python_cmd = f"python \"{os.path.join(PROJECT_ROOT_DIR, PYTHON_IMPORT_SCRIPT)}\""
        
        subprocess.run(python_cmd, shell=True, check=True)
        print("\n" + "="*65)
        print("[Pipeline 完美落地] 本機模擬、MATLAB 計算到跨雲端同步全自動完成！")
        print("=" * 65)
        print("提示：現在可以直接打開 Render 網頁重新整理看成果囉！")

    except subprocess.CalledProcessError as e:
        print(f"Pipeline 中途崩潰，命令執行失敗: {e}")
    except Exception as e:
        print(f"系統發生未知異常: {e}")


# ── 3. 模擬 Samba 資料夾監聽器 ──────────────────────────────────────
class SambaFileHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_triggered = 0

    def on_created(self, event):
        if not event.is_directory:
            current_time = time.time()
            if current_time - self.last_triggered > 10:  
                self.last_triggered = current_time
                print(f"\n🔍 偵測到硬體傳入新檔案: {os.path.basename(event.src_path)}")
                print("等待硬體檔案寫入穩定 (3秒)...")
                time.sleep(3) 
                trigger_pipeline()


if __name__ == "__main__":
    if not os.path.exists(SAMBA_WATCH_DIR):
        os.makedirs(SAMBA_WATCH_DIR)

    print("=" * 65)
    print("  Wi-Fi CSI 睡眠監測 ── 模擬 Samba 自動化守護進程已啟動！")
    print(f"  目前正在實時監控測試路徑:\n    {SAMBA_WATCH_DIR}")
    print("=" * 65)
    print("提示：請保持此視窗存活，並開啟另一個終端機跑 mock_samba_trigger.py...")

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