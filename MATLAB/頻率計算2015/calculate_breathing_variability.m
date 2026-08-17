function [var_history, var_time] = calculate_breathing_variability(all_bpm_cell, all_time_cell, num_files)

    % 每個 seg 計算一個 breathing variability
    %
    % 缺失資料規則：
    %   - NaN、BPM < 5、BPM > 40 視為缺失
    %   - 缺失比例 > 20% -> 該 seg = NaN
    %   - 缺失比例 <= 20% -> pchip 插值後計算 variability
    %
    % 輸入：
    %   all_bpm_cell  - 每個 seg 的 BPM，cell array
    %   all_time_cell - 每個 seg 的時間，cell array
    %   num_files     - seg 數量
    %
    % 輸出：
    %   var_history   - 每個 seg 一個 variability
    %   var_time      - 每個 seg 的中心時間

    window_sec = 180;
    max_missing_ratio = 0.20;

    var_history = NaN(num_files, 1);
    var_time    = NaN(num_files, 1);

    for i = 1:num_files

        % ---------------------------------------------------------
        % 取得目前 seg
        % ---------------------------------------------------------
        if isempty(all_bpm_cell{i}) || isempty(all_time_cell{i})
            continue;
        end

        bpm  = all_bpm_cell{i}(:);
        time = all_time_cell{i}(:);

        % 時間與 BPM 長度不一致時，取共同長度
        n = min(length(bpm), length(time));

        if n < 2
            continue;
        end

        bpm  = bpm(1:n);
        time = time(1:n);

        % ---------------------------------------------------------
        % Epoch 中心時間
        % ---------------------------------------------------------
        var_time(i) = (time(1) + time(end)) / 2;

        % ---------------------------------------------------------
        % 判斷有效資料
        % ---------------------------------------------------------
        valid_idx = ~isnan(bpm) & bpm >= 5 & bpm <= 40;

        % 沒有足夠有效資料
        if sum(valid_idx) < 2
            continue;
        end

        % ---------------------------------------------------------
        % 計算缺失比例
        % ---------------------------------------------------------
        missing_ratio = 1 - sum(valid_idx) / length(valid_idx);

        % 缺失 > 20%，直接 NaN
        if missing_ratio > max_missing_ratio
            continue;
        end

        % ---------------------------------------------------------
        % 插值
        % ---------------------------------------------------------
        bpm_interp = interp1( ...
            time(valid_idx), ...
            bpm(valid_idx), ...
            time, ...
            'pchip', ...
            'extrap');

        bpm_interp = bpm_interp(:);

        % ---------------------------------------------------------
        % 計算採樣率
        % ---------------------------------------------------------
        dt = mean(diff(time));

        if isempty(dt) || dt <= 0 || ~isfinite(dt)
            continue;
        end

        fs_bpm = 1 / dt;

        % ---------------------------------------------------------
        % Butterworth 低通濾波
        % ---------------------------------------------------------
        Wn = 0.1 / (fs_bpm / 2);

        if Wn >= 1
            Wn = 0.99;
        elseif Wn <= 0
            Wn = 0.01;
        end

        % 邊界 padding
        pad_len = min(100, length(bpm_interp)-1);

        if pad_len > 0
            bpm_padded = padarray( ...
                bpm_interp, ...
                pad_len, ...
                'replicate', ...
                'both');
        else
            bpm_padded = bpm_interp;
        end

        % Butterworth
        [b, a] = butter(4, Wn, 'low');
        trend_padded = filtfilt(b, a, bpm_padded);

        % 還原長度
        if pad_len > 0
            trend = trend_padded( ...
                pad_len+1 : end-pad_len);
        else
            trend = trend_padded;
        end

        % ---------------------------------------------------------
        % 去趨勢
        % ---------------------------------------------------------
        detrended_bpm = bpm_interp - trend;

        % ---------------------------------------------------------
        % Variance
        % ---------------------------------------------------------
        if length(detrended_bpm) >= 2
            raw_variance = var(detrended_bpm);

            % 和原本相同，除以 180 秒
            var_history(i) = raw_variance / window_sec;
        else
            var_history(i) = NaN;
        end
    end
end