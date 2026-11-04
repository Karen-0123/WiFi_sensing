function [var_history, var_time, varience, rr_features] = calculate_breathing_variability(all_bpm_cell, all_time_cell, num_files)
% calculate_breathing_variability  基於《WiFi-Sleep》高階多項式去趨勢之呼吸變異度與呼吸率統計特徵計算
%
% 論文演算法原理與修正 (Section IV-D, Page 13905-13906, Fig. 9, WiFi-Sleep, IEEE IoT Journal 2021):
%   1. 缺失資料規則：
%      - NaN、BPM < 5、BPM > 40 視為缺失
%      - 缺失比例 > 20% -> 該 seg = NaN
%      - 缺失比例 <= 20% -> 內部 pchip 插值、邊界 nearest 補齊 (避免三次外插發散)
%   2. 修正舊版 Bug：
%      - 舊版使用 0.1 Hz 低通濾波器對 20 秒滑動視窗 (等效頻寬 ~0.022 Hz) 的 BPM 去趨勢，
%        導致真實呼吸率波動全部被當成 trend 減掉，且錯誤除以 window_sec = 180。
%      - 現改為論文 Section IV-D (Fig. 9) 之「整晚高階多項式基線擬合去趨勢 (Higher-Order Polynomial
%        Detrending)」，並結合區段線性去趨勢扣除緩慢漂移，完整保留 20 秒視窗反映之真實呼吸波動。
%   3. 同時計算論文定義之呼吸率延伸特徵：
%      - Resp_Rate_Mean       : 區段平均呼吸率 (bpm)
%      - Resp_Rate_Smoothed   : 高階多項式去趨勢後之平滑呼吸率 (bpm)
%      - Resp_Rate_Derivative : 平滑呼吸率之一階時間導數 (bpm/min)
%      - Resp_Rate_IQR        : 區段呼吸率四分位距 (Interquartile Range, 75th - 25th)
%
% 輸入:
%   all_bpm_cell  - 每個 seg 的 BPM，cell array
%   all_time_cell - 每個 seg 的時間，cell array
%   num_files     - seg 數量
%
% 輸出:
%   var_history   - 每個 seg 的去趨勢呼吸率變異度 (Breathing Rate Variability) [num_files x 1]
%   var_time      - 每個 seg 的中心時間 [num_files x 1]
%   varience      - 每個 seg 的原始插值後變異數 [num_files x 1]
%   rr_features   - 論文呼吸率特徵結構體 (.mean, .smoothed, .derivative, .iqr, .poly_baseline)

    max_missing_ratio = 0.20;

    var_history    = NaN(num_files, 1);
    var_time       = NaN(num_files, 1);
    varience       = NaN(num_files, 1);
    resp_mean      = NaN(num_files, 1);
    resp_smoothed  = NaN(num_files, 1);
    resp_deriv     = NaN(num_files, 1);
    resp_iqr       = NaN(num_files, 1);
    poly_baseline  = NaN(num_files, 1);

    interp_bpm_cell  = cell(num_files, 1);
    interp_time_cell = cell(num_files, 1);
    seg_valid_flag   = false(num_files, 1);

    % ---------------------------------------------------------------------
    % 第一階段：各區段缺失值檢查、安全 pchip 插值與基礎統計量計算
    % ---------------------------------------------------------------------
    for i = 1:num_files
        if isempty(all_bpm_cell{i}) || isempty(all_time_cell{i})
            continue;
        end

        bpm  = all_bpm_cell{i}(:);
        time = all_time_cell{i}(:);
        n = min(length(bpm), length(time));
        if n < 2
            continue;
        end

        bpm  = bpm(1:n);
        time = time(1:n);
        var_time(i) = (time(1) + time(end)) / 2;

        valid_idx = ~isnan(bpm) & bpm >= 5 & bpm <= 40;
        if sum(valid_idx) < 2
            continue;
        end

        missing_ratio = 1 - sum(valid_idx) / length(valid_idx);
        if missing_ratio > max_missing_ratio
            continue;
        end

        % 內部使用 pchip 保形插值，首尾邊界缺口使用 nearest 補齊防止外插發散
        t_v = time(valid_idx);
        b_v = bpm(valid_idx);
        bpm_interp = interp1(t_v, b_v, time, 'pchip');
        nan_edge = isnan(bpm_interp);
        if any(nan_edge)
            bpm_interp(nan_edge) = interp1(t_v, b_v, time(nan_edge), 'nearest', 'extrap');
        end
        bpm_interp = max(5, min(40, bpm_interp(:)));

        interp_bpm_cell{i}  = bpm_interp;
        interp_time_cell{i} = time;
        seg_valid_flag(i)   = true;

        resp_mean(i) = mean(bpm_interp);
        varience(i)  = var(bpm_interp);
        resp_iqr(i)  = prctile(bpm_interp, 75) - prctile(bpm_interp, 25);
    end

    % ---------------------------------------------------------------------
    % 第二階段：論文 Section IV-D (Fig. 9) 整晚高階多項式曲線基線擬合與去趨勢
    % ---------------------------------------------------------------------
    all_t_concat   = vertcat(interp_time_cell{seg_valid_flag});
    all_bpm_concat = vertcat(interp_bpm_cell{seg_valid_flag});
    num_valid_segs = sum(seg_valid_flag);

    has_global_poly = false;
    if length(all_t_concat) >= 6 && (max(all_t_concat) - min(all_t_concat)) > 0
        if num_valid_segs >= 10
            poly_deg = 4;
        elseif num_valid_segs >= 4
            poly_deg = 3;
        elseif num_valid_segs >= 2
            poly_deg = 2;
        else
            poly_deg = 1;
        end
        [poly_coeff, ~, poly_mu] = polyfit(all_t_concat, all_bpm_concat, poly_deg);
        has_global_poly = true;
    end

    for i = 1:num_files
        if ~seg_valid_flag(i)
            continue;
        end

        bpm_interp = interp_bpm_cell{i};
        time       = interp_time_cell{i};

        % 1. 計算整晚高階多項式基線
        if has_global_poly
            global_trend = polyval(poly_coeff, time, [], poly_mu);
        else
            global_trend = mean(bpm_interp) * ones(size(bpm_interp));
        end
        poly_baseline(i) = mean(global_trend);

        % 2. 區段內線性趨勢扣除 (正確搭配正規化參數 local_mu，保留 20s 視窗真實生理呼吸波動)
        dt = mean(diff(time));
        if ~isempty(dt) && dt > 0 && isfinite(dt) && length(bpm_interp) >= 3 && (max(time) > min(time))
            [local_p, ~, local_mu] = polyfit(time, bpm_interp, 1);
            local_trend = polyval(local_p, time, [], local_mu);
        else
            local_trend = global_trend;
        end

        detrended_local  = bpm_interp - local_trend;
        detrended_global = bpm_interp - global_trend;

        % 區段內呼吸率變異度 (Breathing Rate Variability)
        var_history(i)   = var(detrended_local);

        % 整晚高階多項式去趨勢後之平滑呼吸率 (對應論文 Fig. 9 下圖 Detrended Respiratory Rate)
        resp_smoothed(i) = mean(detrended_global);

        % 單一區段時之呼吸率一階導數
        if num_valid_segs == 1 && length(time) >= 2 && dt > 0
            resp_deriv(i) = mean(gradient(bpm_interp, dt)) * 60; % bpm / min
        end
    end

    % ---------------------------------------------------------------------
    % 第三階段：計算跨區段平滑呼吸率之一階時間導數 (First-Order Derivative)
    % ---------------------------------------------------------------------
    if num_valid_segs >= 2
        valid_indices = find(seg_valid_flag);
        t_valid_centers = var_time(valid_indices) / 60; % 轉為分鐘
        s_valid_vals    = resp_smoothed(valid_indices);

        if length(valid_indices) >= 2
            deriv_vals = gradient(s_valid_vals, t_valid_centers); % bpm / min
            resp_deriv(valid_indices) = deriv_vals;
        end
    end

    rr_features = struct( ...
        'mean', resp_mean, ...
        'smoothed', resp_smoothed, ...
        'derivative', resp_deriv, ...
        'iqr', resp_iqr, ...
        'poly_baseline', poly_baseline);
end
