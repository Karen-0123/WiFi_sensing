import paramiko
import time
import os
import subprocess

# --- 設定區 ---
HOSTNAME = '172.20.10.4'
USERNAME = 'lab114'
KEY_PATH = os.path.expanduser("C:/Users/user/.ssh/id_ed25519")
REMOTE_PATH = '/home/lab114/data'

# MATLAB 與 Python 環境路徑
MATLAB_EXE_PATH = r"C:\Program Files\MATLAB\R2015b\bin\matlab.exe"
MATLAB_SCRIPT_DIR = r"D:\大學資料\WiFi_sensing\MATLAB\頻率計算2015"
PROJECT_ROOT_DIR = r"D:\大學資料\WiFi_sensing"
PYTHON_IMPORT_SCRIPT = "import_real_data.py"

processed_files = set()

def trigger_pipeline(filename):
    """執行 MATLAB 與雲端同步 Pipeline"""
    try:
        print(f"\n[Step A] 呼叫 MATLAB 進行運算: {filename}")
        matlab_cmd = f'"{MATLAB_EXE_PATH}" -nosplash -nodesktop -r "cd(\'{MATLAB_SCRIPT_DIR}\'); process_csi_signals; exit;"'
        subprocess.run(matlab_cmd, shell=True, check=True)
        
        print(f"[Step B] 上傳至 Aiven 雲端...")
        python_cmd = f"python \"{os.path.join(PROJECT_ROOT_DIR, PYTHON_IMPORT_SCRIPT)}\""
        subprocess.run(python_cmd, shell=True, check=True)
        print("[完成] Pipeline 執行完畢。")
    except Exception as e:
        print(f"Pipeline 執行錯誤: {e}")

def monitor_sftp():
    transport = paramiko.Transport((HOSTNAME, 22))
    my_key = paramiko.Ed25519Key.from_private_key_file(KEY_PATH)
    transport.connect(username=USERNAME, pkey=my_key)
    sftp = paramiko.SFTPClient.from_transport(transport)
    
    print(f"✅ SSH 監控已啟動，目標路徑: {REMOTE_PATH}")
    
    try:
        while True:
            files = sftp.listdir(REMOTE_PATH)
            for file in files:
                if file.endswith(('.dat', '.csv')) and file not in processed_files:
                    full_remote_path = f"{REMOTE_PATH}/{file}"
                    print(f"\n偵測到新檔案: {file}，檢查傳輸狀態...")
                    
                    # 檔案完整性檢查機制 (防呆)
                    last_size = -1
                    while True:
                        attr = sftp.stat(full_remote_path)
                        current_size = attr.st_size
                        if current_size == last_size and current_size > 0:
                            print(f"檔案傳輸完成 (大小: {current_size} bytes)。")
                            break
                        last_size = current_size
                        time.sleep(2) # 每 2 秒檢查一次檔案是否還在增加
                    
                    # 觸發邏輯
                    trigger_pipeline(file)
                    processed_files.add(file)
                    
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n監控服務已關閉。")
    finally:
        sftp.close()
        transport.close()

if __name__ == "__main__":
    monitor_sftp()