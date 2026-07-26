@ -1,37 +0,0 @@
function rrv_results = calculate_rrv_metrics(true_peak_idx, Fs_target)
    % 行動通訊與無線感知標準：計算微觀呼吸間隔變異性 (RRV)
    % 輸入:
    %   true_peak_idx - 偵測到的呼吸波峰在重採樣矩陣中的索引值 (向量)
    %   Fs_target     - 訊號的目標採樣率 (例如 20 Hz)
    % 輸出:
    %   rrv_results   - 包含 SDNN, RMSSD 等標準生理學指標的結構體
    
    % 初始化輸出結構體
    rrv_results = struct('SDNN', NaN, 'RMSSD', NaN, 'Mean_BB', NaN, 'Total_Breaths', 0);
    
    % 檢查有效波峰數量，至少需要 3 個波峰才能計算相鄰差值
    if length(true_peak_idx) < 3
        warning('【系統提示】有效呼吸頂點過少，無法計算相鄰呼吸變異性指標。');
        return;
    end
    
    % 1. 將紅點頂點的矩陣索引轉換為「實際時間戳記（秒）」
    peak_times_sec = true_peak_idx(:) / Fs_target;
    
    % 2. 計算相鄰兩次呼吸的時間間隔 (Breath-to-Breath / BB Intervals)
    %    這就是學術論文中定義的 BB intervals
    bb_intervals = diff(peak_times_sec); % 單位：秒
    
    % 3. 計算相鄰呼吸間隔的差值（用於計算 RMSSD）
    successive_diffs = diff(bb_intervals); % 單位：秒
    
    % 4. 封裝標準時域生理指標 (比照 BioSPPy 與 NeuroKit2 數學定義)
    rrv_results.Total_Breaths = length(peak_times_sec);
    rrv_results.Mean_BB = mean(bb_intervals);
    
    % 時域指標 1: SDNN (所有呼吸間隔時間的標準差，反應總體變異性)
    rrv_results.SDNN = std(bb_intervals);
    
    % 時域指標 2: RMSSD (相鄰呼吸間隔差值的均方根，反應短期快速變異)
    rrv_results.RMSSD = sqrt(mean(successive_diffs.^2));
end