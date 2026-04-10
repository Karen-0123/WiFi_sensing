function [stage_timeline, time_axis_epochs] = classify_sleep_timeline(bpm_timeline, time_axis_bpm)
    % 透過滑動視窗 (Sliding Window) 進行連續睡眠階段分類
    % 輸入:
    %   bpm_timeline   - 由 calculate_dynamic_bpm 輸出的動態呼吸率陣列
    %   time_axis_bpm  - 對應的時間軸 (秒)
    % 輸出:
    %   stage_timeline   - 每個 Epoch 的睡眠階段結果 (Cell Array)
    %   time_axis_epochs - 每個判斷點對應的時間中心點 (分鐘)

    % ======================================================================
    % 1. 視窗參數設定
    % 根據臨床 AASM 標準，我們每 30 秒輸出一個結果 (Epoch)
    % 但為了統計穩定性，判斷邏輯會參考過去 180 秒 (3 分鐘) 的數據
    % ======================================================================
    window_size_sec = 180;  % 窗口長度: 180 秒
    step_size_sec = 30;     % 步進長度: 30 秒
    fs_bpm = 1;             % 假設 bpm_timeline 的採樣率是每秒 1 點 (由 step_size 決定)

    % 計算總時間與總 Epoch 數量
    total_duration = time_axis_bpm(end) - time_axis_bpm(1);
    num_epochs = floor((total_duration - window_size_sec) / step_size_sec) + 1;

    if num_epochs <= 0
        error('資料總長度不足 180 秒，無法進行睡眠分期分析。');
    end

    % 預先配置記憶體
    stage_timeline = cell(1, num_epochs);
    time_axis_epochs = zeros(1, num_epochs);

    % ======================================================================
    % 2. 執行滑動視窗迴圈
    % ======================================================================
    fprintf('步驟 7: 開始執行連續睡眠分期 (窗口: %ds, 步進: %ds)...\n', window_size_sec, step_size_sec);

    for i = 1:num_epochs
        % 計算當前窗口的時間範圍
        t_start = (i-1) * step_size_sec;
        t_end = t_start + window_size_sec;
        
        % 記錄該 Epoch 的中心時間點 (轉換為分鐘，方便畫圖)
        time_axis_epochs(i) = (t_start + (window_size_sec / 2)) / 60;

        % 找出落在當前 3 分鐘窗口內的所有 BPM 數據索引
        logical_idx = (time_axis_bpm >= t_start & time_axis_bpm <= t_end);
        current_window_data = bpm_timeline(logical_idx);

        % 呼叫核心判斷函數
        % 這裡會把 3 分鐘的數據丟進去，回傳一個最可能的階段
        [stage, ~] = analyze_sleep_stage(current_window_data);
        
        % 儲存結果
        stage_timeline{i} = stage;
    end

    % ======================================================================
    % 3. 結果統計輸出
    % ======================================================================
    % 簡單統計各階段出現比例
    unique_stages = unique(stage_timeline);
    fprintf('--- 睡眠分期統計 ---\n');
    for k = 1:length(unique_stages)
        count = sum(strcmp(stage_timeline, unique_stages{k}));
        fprintf('%s: %.1f%%\n', unique_stages{k}, (count/num_epochs)*100);
    end
    fprintf('連續睡眠分期完成！\n');
end