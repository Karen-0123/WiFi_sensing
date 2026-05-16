function [stages, epoch_time] = predict_sleep_stages(all_bpm, all_time, var_h, var_t, baseline_bpm, all_motion, all_m_time)
% PREDICT_SLEEP_STAGES 依據心率、變異數與體動資料預測睡眠階段
%
% 輸入參數：
%   all_bpm      - 原始心率資料 (BPM)
%   all_time     - 原始心率資料對應的時間軸 (秒)
%   var_h        - 呼吸或心率變異數數值 (Variance)
%   var_t        - 變異數對應的時間軸 (分鐘)
%   baseline_bpm - 基準心率 (BPM)
%   all_motion   - 體動感測資料 (1 表示有動作，0 表示靜止)
%   all_m_time   - 體動資料對應的時間軸 (秒)
%
% 輸出參數：
%   stages       - 平滑處理後的最終睡眠階段序列 (0:Deep, 1:Core, 2:REM, 3:Awake)
%   epoch_time   - 每個 Epoch 對應的時間軸 (單位：分鐘)

    % --- 時間軸與空間初始化 ---
    epoch_sec = 30;                         % 定義每個睡眠分期區間 (Epoch) 為 30 秒
    num_epochs = floor(max(all_time) / epoch_sec); % 計算整段資料總共包含幾個 Epoch
    epoch_time = (1:num_epochs) * epoch_sec / 60;  % 將每個 Epoch 的時間點轉換為「分鐘」，方便後續繪圖或對齊
    
    raw_stages = zeros(1, num_epochs);      % 初始化用來儲存原始預測結果的陣列
    
    % --- 門檻參數設定 (對齊BPM 基準) ---
    threshold_awake = 4.0;       % 判定為清醒 (Awake) 的心率偏差門檻
    threshold_rem_low = 2.5;     % 判定為快速動眼期 (REM) 的心率偏差下限門檻
    threshold_deep_var = 6.0;    % 調高門檻，讓藍色虛線上升，多抓一點深眠？ (深眠變異數上限門檻)
    threshold_deep_dev = 2.8;    % 判定為深眠 (Deep) 的心率偏差上限門檻

    % --- 逐個 Epoch 進行狀態判定 ---
    for i = 1:num_epochs
        t_start = (i-1) * epoch_sec; % 當前 Epoch 的起始時間 (秒)
        
        % 尋找與當前 Epoch 正中間 (t_start + 15秒) 最接近的變異數時間點索引
        [~, v_idx] = min(abs(var_t - (t_start + 15)/60));
        curr_var = var_h(v_idx);     % 取得當前 Epoch 對應的變異數數值
        
        % 找出落在當前 30 秒 Epoch 區間內的所有原始心率資料
        idx = (all_time >= t_start) & (all_time < t_start + epoch_sec);
        if any(idx)
            % 計算當前區間內的心率與基準心率的平均絕對偏差 (移除 NaN 值)
            curr_dev = mean(abs(all_bpm(idx) - baseline_bpm), 'omitnan');
        else
            curr_dev = 0; % 若該區間無心率資料，則偏差計為 0
        end
        
        % --- 優先權判定邏輯 ---
        
        % 條件 1. 清醒 (Awake) 判定
        % 當心率偏差大於門檻值，或者該 30 秒內有任何體動紀錄 (all_motion 含有 1)
        if curr_dev > threshold_awake || any(all_motion((all_m_time >= t_start) & (all_m_time < t_start + epoch_sec)) == 1)
            obs = 3; % 3 代表 Awake
            
        % 條件 2. 快速動眼期 (REM) 判定
        % 排除清醒後，若心率偏差仍大於 REM 下限 (代表呼吸/心率偏差處於中高位)
        elseif curr_dev > threshold_rem_low
            obs = 2; % 2 代表 REM
            
        % 條件 3. 深層睡眠 (Deep) 判定
        % 排除清醒與 REM 後，若變異數低於 6.0 且心率偏差也低於 2.8 (代表呼吸與心率極度穩定)
        elseif curr_var < threshold_deep_var && curr_dev < threshold_deep_dev
            obs = 0; % 0 代表 Deep
            
        % 條件 4. 淺層睡眠/核心睡眠 (Core) 判定：若不符合上述所有條件，則歸類為常態的 Core 睡眠
        else
            obs = 1; % 1 代表 Core
        end
        
        raw_stages(i) = obs; % 記錄該 Epoch 的原始判定結果
    end
    
    % --- 塊狀化平滑濾波 ---
    % 使用 21 點的一維中值濾波器 (medfilt1) 消除過於短暫、細碎的訊號雜訊。
    % 21 個 Epoch 相當於 10.5 分鐘的視窗，這是一個很平衡的數字，能有效維持睡眠階段的連續區塊。
    stages = medfilt1(raw_stages, 21); 
end