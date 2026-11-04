function [true_peak_idx, true_peak_vals, valley_idx, valley_vals, waveform_features] = detect_respiration_peaks(signal, gap_mask, Fs_target, motion_mask, sg_signal)
% detect_respiration_peaks  呼吸波峰與波谷精細偵測 + 《WiFi-Sleep》強化呼吸波形特徵提取
%
% 輸入:
%   signal      - 最佳帶通呼吸波形 [N x 1]
%   gap_mask    - 掉包遮罩 [N x 1] (true 代表掉包區間)
%   Fs_target   - 採樣率 (Hz，預設 40 Hz)
%   motion_mask - (可選) 體動區間遮罩 [N x 1] (true 代表體動干擾區間)
%   sg_signal   - (可選) SG 平滑 MRC-PCA 呼吸波形 [N x 1] (供 FIT / I-E Ratio 使用)
%
% 輸出:
%   true_peak_idx     - 有效呼吸波峰索引 (1-based)
%   true_peak_vals    - 有效呼吸波峰幅值
%   valley_idx        - 有效呼吸波谷索引 (1-based)
%   valley_vals       - 有效呼吸波谷幅值
%   waveform_features - 論文強化呼吸波形特徵結構體 (.resp_depth_var, .fit_mean, .fit_var, .ie_ratio_mean, .ie_ratio_var)

    if nargin < 3 || isempty(Fs_target), Fs_target = 40; end
    signal = signal(:);
    N = length(signal);

    waveform_features = struct( ...
        'resp_depth_var', NaN, ...
        'fit_mean', NaN, ...
        'fit_var', NaN, ...
        'ie_ratio_mean', NaN, ...
        'ie_ratio_var', NaN);

    if N < round(2 * Fs_target) || all(abs(signal) < eps)
        true_peak_idx  = [];
        true_peak_vals = [];
        valley_idx     = [];
        valley_vals    = [];
        return;
    end

    if nargin < 2 || isempty(gap_mask)
        gap_mask = false(N, 1);
    else
        gap_mask = logical(gap_mask(:));
    end

    if nargin < 4 || isempty(motion_mask)
        motion_mask = false(N, 1);
    else
        motion_mask = logical(motion_mask(:));
        if length(motion_mask) ~= N
            motion_mask = false(N, 1);
        end
    end

    if nargin < 5 || isempty(sg_signal) || length(sg_signal) ~= N
        sg_signal = signal;
    else
        sg_signal = sg_signal(:);
    end

    % 合併無效區間遮罩 (掉包區段 + 體動區段)
    invalid_mask = gap_mask | motion_mask;
    clean_samples = signal(~invalid_mask);
    if length(clean_samples) < round(2 * Fs_target)
        clean_samples = signal(~gap_mask);
    end
    if isempty(clean_samples)
        clean_samples = signal;
    end

    % 1. 使用穩健 MAD 估計安靜呼吸標準差，並依據主導呼吸週期動態設定最小峰距與鄰域視窗
    sig_med = median(clean_samples);
    robust_std = 1.4826 * median(abs(clean_samples - sig_med));
    if robust_std < eps
        robust_std = std(clean_samples);
    end
    adaptive_prominence = max(0.25 * robust_std, 1e-4);

    % 估計訊號之主導呼吸週期 T_dom (介於 2.2 ~ 4.8 秒，對應 12.5 ~ 27 bpm)
    T_dom = estimate_dominant_resp_period(clean_samples, Fs_target);

    % 動態設定最小峰距 (約 0.58 * T_dom，既不漏抓 3 秒正常呼吸，亦能阻擋 T/2 二次諧波雙峰)
    min_peak_dist = max(round(1.5 * Fs_target), round(0.58 * T_dom * Fs_target));
    [candidate_pks, candidate_locs] = findpeaks(signal, ...
        'MinPeakProminence', adaptive_prominence, ...
        'MinPeakDistance', min_peak_dist);

    % 2. 局部鄰域最大值驗證視窗 (前後各約 0.42 * T_dom，避免因固定 2.5 秒過寬視窗吃掉相鄰正常呼吸峰)
    half_window = max(round(0.75 * Fs_target), round(0.42 * T_dom * Fs_target));

    true_peak_idx  = [];
    true_peak_vals = [];

    % 3. 剔除掉包區間與體動區間內之偽峰
    for i = 1:length(candidate_locs)
        curr_idx = candidate_locs(i);
        curr_val = candidate_pks(i);

        if invalid_mask(curr_idx)
            continue;
        end

        start_idx = max(1, curr_idx - half_window);
        end_idx   = min(N, curr_idx + half_window);

        if curr_val >= max(signal(start_idx:end_idx)) - 1e-12
            true_peak_idx(end+1)  = curr_idx;
            true_peak_vals(end+1) = curr_val;
        end
    end

    % 4. 偵測呼吸波谷 (Expiration Valleys)
    [cand_v_vals, cand_v_locs] = findpeaks(-signal, ...
        'MinPeakProminence', adaptive_prominence, ...
        'MinPeakDistance', min_peak_dist);
    cand_v_vals = -cand_v_vals;

    valley_idx  = [];
    valley_vals = [];
    for i = 1:length(cand_v_locs)
        curr_idx = cand_v_locs(i);
        curr_val = cand_v_vals(i);
        if invalid_mask(curr_idx)
            continue;
        end
        start_idx = max(1, curr_idx - half_window);
        end_idx   = min(N, curr_idx + half_window);
        if curr_val <= min(signal(start_idx:end_idx)) + 1e-12
            valley_idx(end+1)  = curr_idx;
            valley_vals(end+1) = curr_val;
        end
    end

    % 5. 提取《WiFi-Sleep》強化呼吸波形特徵 (Section IV-D, Page 13906, Fig. 6)
    waveform_features = compute_enhanced_waveform_features( ...
        signal, sg_signal, true_peak_idx, valley_idx, invalid_mask, Fs_target);

    fprintf('峰值偵測完成：候選峰值 %d 個 -> 校正後保留 %d 個有效呼吸頂點 (波谷 %d 個)。\n', ...
            length(candidate_locs), length(true_peak_idx), length(valley_idx));
end

% =========================================================================
% 輔助函式：計算呼吸深度變異度、吸氣時間佔比 (FIT) 與吸吐比 (I/E Ratio)
% =========================================================================
function wf = compute_enhanced_waveform_features(bp_sig, sg_sig, peak_idx, valley_idx, invalid_mask, Fs)
    wf = struct( ...
        'resp_depth_var', NaN, ...
        'fit_mean', NaN, ...
        'fit_var', NaN, ...
        'ie_ratio_mean', NaN, ...
        'ie_ratio_var', NaN);

    N = length(bp_sig);
    if length(peak_idx) < 2
        return;
    end

    % (A) 呼吸深度變異度 (Respiration Depth Variance)
    win_len = round(30 * Fs);
    num_subwins = max(1, floor(N / win_len));
    local_depth_vars = [];

    for w = 1:num_subwins
        s_i = (w - 1) * win_len + 1;
        e_i = min(N, w * win_len);
        if mean(invalid_mask(s_i:e_i)) > 0.30
            continue;
        end
        pks_in_w = peak_idx(peak_idx >= s_i & peak_idx <= e_i);
        if length(pks_in_w) >= 3
            sub_sig = bp_sig(s_i:e_i);
            clean_sub = sub_sig(~invalid_mask(s_i:e_i));
            sub_std = 1.4826 * median(abs(clean_sub - median(clean_sub)));
            if sub_std > eps
                norm_pks = (bp_sig(pks_in_w) - median(clean_sub)) / sub_std;
                local_depth_vars(end+1) = var(norm_pks);
            end
        end
    end

    if ~isempty(local_depth_vars)
        wf.resp_depth_var = mean(local_depth_vars);
    else
        clean_all = bp_sig(~invalid_mask);
        clean_std = 1.4826 * median(abs(clean_all - median(clean_all)));
        if clean_std > eps && length(peak_idx) >= 2
            norm_pks = (bp_sig(peak_idx) - median(clean_all)) / clean_std;
            wf.resp_depth_var = var(norm_pks);
        end
    end

    % (B) 吸氣時間佔比 (FIT) 與 吸吐氣時間比 (I/E Ratio)
    min_cycle_sec = 60 / 35; % 1.71s
    max_cycle_sec = 60 / 8;  % 7.5s
    refine_rad    = round(0.35 * Fs);

    fit_list = [];
    ie_list  = [];

    for k = 1:(length(valley_idx) - 1)
        v1 = valley_idx(k);
        v2 = valley_idx(k+1);
        cycle_sec = (v2 - v1) / Fs;

        if cycle_sec < min_cycle_sec || cycle_sec > max_cycle_sec || any(invalid_mask(v1:v2))
            continue;
        end

        p_between = peak_idx(peak_idx > v1 & peak_idx < v2);
        if length(p_between) ~= 1
            continue;
        end
        p_mid = p_between(1);

        v1_range = max(1, v1 - refine_rad) : min(N, v1 + refine_rad);
        [~, min1_rel] = min(sg_sig(v1_range));
        v1_ref = v1_range(min1_rel);

        v2_range = max(1, v2 - refine_rad) : min(N, v2 + refine_rad);
        [~, min2_rel] = min(sg_sig(v2_range));
        v2_ref = v2_range(min2_rel);

        p_range = max(v1_ref + 1, p_mid - refine_rad) : min(v2_ref - 1, p_mid + refine_rad);
        if isempty(p_range)
            p_ref = p_mid;
        else
            [~, max_rel] = max(sg_sig(p_range));
            p_ref = p_range(max_rel);
        end

        t_in  = (p_ref - v1_ref) / Fs;
        t_ex  = (v2_ref - p_ref) / Fs;
        t_tot = t_in + t_ex;

        if t_in > 0.25 && t_ex > 0.25 && t_tot >= min_cycle_sec && t_tot <= max_cycle_sec
            fit_val = t_in / t_tot;
            ie_val  = t_in / t_ex;
            if fit_val >= 0.20 && fit_val <= 0.80
                fit_list(end+1) = fit_val;
                ie_list(end+1)  = ie_val;
            end
        end
    end

    if ~isempty(fit_list)
        wf.fit_mean      = mean(fit_list);
        wf.fit_var       = var(fit_list);
        wf.ie_ratio_mean = mean(ie_list);
        wf.ie_ratio_var  = var(ie_list);
    end
end

function T_dom = estimate_dominant_resp_period(x, Fs)
% 利用頻譜主峰估計主導呼吸週期 (限制於 2.2 ~ 4.8 秒，對應 12.5 ~ 27 bpm)
    x = double(x(:));
    n = length(x);
    if n < round(4 * Fs)
        T_dom = 3.5;
        return;
    end
    x = x - mean(x);
    Y = fft(x);
    P = abs(Y(1:floor(n/2)+1)).^2;
    f = Fs * (0:floor(n/2))' / n;
    mask = (f >= 0.21) & (f <= 0.48); % 12.6 ~ 28.8 bpm
    if any(mask)
        P_band = P(mask);
        f_band = f(mask);
        [~, idx] = max(P_band);
        f_peak = f_band(idx);
        T_dom = max(2.2, min(4.8, 1.0 / f_peak));
    else
        T_dom = 3.5;
    end
end
