function [bpm_timeline, time_axis_bpm] = calculate_dynamic_bpm(true_peak_idx, total_samples)
    % 透過滑動窗口與 P2P 間隔計算動態呼吸頻率 (BPM)
    % 輸入:
    %   true_peak_idx - 有效呼吸頂點的索引值陣列
    %   total_samples - 原始訊號的總採樣點數
    % 輸出:
    %   bpm_timeline  - 隨時間變化的 BPM 陣列
    %   time_axis_bpm - 對應的時間軸 (以窗口中心點為準)

    fs = 200; % 採樣頻率 200 Hz
    
    % 將峰值索引轉換為真實時間 (秒)
    peak_times = true_peak_idx / fs;
    total_time = total_samples / fs;

    % 1. 滑動窗口設定
    window_size = 20; % 20 秒
    step_size = 1;    % 1 秒
    
    % 計算所有窗口的起始時間
    t_starts = 0:step_size:(total_time - window_size);
    num_windows = length(t_starts);
    
    if num_windows <= 0
        error('資料長度不足以建立 20 秒的窗口！請提供更長的資料。');
    end

    % 預先配置記憶體
    bpm_timeline = zeros(1, num_windows);
    time_axis_bpm = zeros(1, num_windows);

    % 2 & 3. 迴圈處理每個窗口，計算 P2P 間隔與 BPM
    for i = 1:num_windows
        t_s = t_starts(i);
        t_e = t_s + window_size;
        
        % 記錄該窗口的中心時間，用於繪圖
        time_axis_bpm(i) = t_s + (window_size / 2);
        
        % 找出落在當前窗口內的峰值時間
        peaks_in_window = peak_times(peak_times >= t_s & peak_times <= t_e);
        
        % 異常處理：檢查峰值數量是否足夠計算間隔
        if length(peaks_in_window) >= 2
            % 計算連續峰值之間的時間差
            p2p_intervals = diff(peaks_in_window);
            
            % 求平均間隔時間 Tp2p
            Tp2p = mean(p2p_intervals);
            
            % 轉換為每分鐘呼吸次數 (BPM)
            bpm_timeline(i) = 60 / Tp2p;
        else
            % 峰值不足時填入 NaN，避免除以零，並在繪圖時自然斷開線條
            bpm_timeline(i) = NaN; 
        end
    end

    % 4. 繪製隨時間變化的 BPM 曲線圖
    figure('Name', '連續呼吸率 (BPM) 監測', 'Position', [250, 250, 900, 400]);
    
    % 繪製 BPM 曲線，遇到 NaN 點會自動留白，代表該時段訊號不佳或無人
    plot(time_axis_bpm, bpm_timeline, '-o', 'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerFaceColor', 'b');
    
    % 計算整體平均 BPM (排除 NaN)
    valid_bpms = bpm_timeline(~isnan(bpm_timeline));
    if ~isempty(valid_bpms)
        avg_bpm = mean(valid_bpms);
        title(sprintf('動態呼吸率變化曲線 | 整體平均: %.1f BPM', avg_bpm), 'FontSize', 14, 'FontWeight', 'bold');
        
        % 畫一條平均基準線作為參考
        %yline(avg_bpm, 'r--', '平均呼吸率', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left');
    else
        title('動態呼吸率變化曲線 (無有效資料)', 'FontSize', 14, 'FontWeight', 'bold');
    end
    
%     xlabel('時間 (秒) - 窗口中心點', 'FontSize', 12);
%     ylabel('呼吸率 (BPM)', 'FontSize', 12);
%     
%     % 設定 Y 軸合理的顯示範圍 (例如人類正常與劇烈呼吸約落在 5 到 45 之間)
%     ylim([5 45]);
%     grid on;
    
    fprintf('BPM 計算完成！成功產出隨時間變化的呼吸率曲線。\n');
end