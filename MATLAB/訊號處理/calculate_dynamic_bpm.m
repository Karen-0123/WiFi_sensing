function [seg_90th, bpm_timeline, time_axis_bpm, bpm_stats] = calculate_dynamic_bpm(true_peak_idx, total_samples, gap_mask, Fs_target, best_sig, motion_mask)
% calculate_dynamic_bpm  基於《WiFi-Sleep》自相關函數 (ACF) 與 P2P 協同驗證、體動切除插值之動態呼吸率模組
%
% 關鍵演算法修正:
%   1. 解決先前 ACF 在基本週期 T (~12 bpm, 5s) 與二次諧波 T/2 (~24 bpm, 2.5s) 之間跳變產生方波的問題：
%      - 移除會隨延遲拉高 2T/3T 峰值的無偏權重放大，改用標準正規化自相關函數 R_yy(tau)。
%      - 當自相關函數在 tau_2 (~2*tau_1) 處之相關峰值高於或接近 tau_1 時，自動識別 tau_2 為真實基頻週期
%        (tau_1 僅為單次呼吸內的吸吐不對稱次波峰)，防止誤抓二倍頻 (24 bpm)。
%      - 結合視窗內已通過鄰域檢驗的時域波峰間距 (P2P) 導引 ACF 主峰鎖定，確保動態 BPM 連續平滑。
%   2. 論文 Section IV-D (Fig. 8) 體動污染修補：將發生體動 (motion_mask) 期間的異常呼吸率切除 (Cut)，
%      對內部區間使用線性插值 (Linear Interpolation) 補齊，邊界處使用最近鄰外插防止斜率發散。
%
% 輸入:
%   true_peak_idx - 檢測到的呼吸波峰索引
%   total_samples - 訊號總採樣點數
%   gap_mask      - 掉包遮罩陣列 [N x 1] (true 表示掉包)
%   Fs_target     - 採樣頻率 (Hz，預設 40 Hz)
%   best_sig      - (可選) 最佳呼吸波形 [N x 1] (供計算 ACF 自相關函數呼吸率)
%   motion_mask   - (可選) 體動遮罩陣列 [N x 1] (true 表示體動期間，將予以切除並線性插值)
%
% 輸出:
%   seg_90th      - 當前區段有效 BPM 的第 90 百分位數
%   bpm_timeline  - 隨時間變化的動態 BPM 陣列 (已完成體動區間切除與線性插值修補)
%   time_axis_bpm - BPM 時間軸 (以滑動視窗中心點為基準)
%   bpm_stats     - 呼吸率統計結構體 (.mean, .iqr, .std, .seg_90th, .raw_before_interp)

    if nargin < 4 || isempty(Fs_target), Fs_target = 40; end
    if nargin < 5, best_sig = []; end
    if nargin < 6 || isempty(motion_mask)
        motion_mask = false(total_samples, 1);
    else
        motion_mask = logical(motion_mask(:));
        if length(motion_mask) ~= total_samples
            motion_mask = false(total_samples, 1);
        end
    end

    gap_mask = logical(gap_mask(:));
    peak_times = true_peak_idx(:) / Fs_target;
    total_time = total_samples / Fs_target;

    % 1. 20 秒滑動視窗，1 秒步長
    window_size = 20;
    step_size   = 1;
    t_starts    = 0 : step_size : max(0, total_time - window_size);

    num_wins        = length(t_starts);
    bpm_timeline    = NaN(1, num_wins);
    time_axis_bpm   = t_starts + (window_size / 2);
    motion_win_flag = false(1, num_wins);

    % 合法呼吸頻率範圍: 9 ~ 35 bpm (週期 1.71s ~ 6.67s)
    min_lag_samples = round((60 / 35) * Fs_target);
    max_lag_samples = round((60 / 9)  * Fs_target);

    % 計算全段有效 P2P 週期中位數作為先驗基頻參考 (防止單一視窗諧波倍頻跳變)
    all_p2p = diff(peak_times);
    valid_all_p2p = all_p2p(all_p2p >= 1.8 & all_p2p <= 6.5);
    if ~isempty(valid_all_p2p)
        global_ref_period = median(valid_all_p2p);
    else
        global_ref_period = NaN;
    end

    % 2. 遍歷滑動視窗計算呼吸率
    for i = 1:num_wins
        t_s = t_starts(i);
        t_e = t_s + window_size;

        idx_start = max(1, round(t_s * Fs_target) + 1);
        idx_end   = min(total_samples, round(t_e * Fs_target));

        % 斷訊品質評估：若此視窗內掉包比例超過 25%，保持 NaN
        if mean(gap_mask(idx_start:idx_end)) > 0.25
            continue;
        end

        % 體動污染標記：若視窗內超過 20% 處於體動狀態，標記為體動污染待後續切除插值
        if mean(motion_mask(idx_start:idx_end)) > 0.20
            motion_win_flag(i) = true;
            continue;
        end

        % 先計算視窗內經過異常值過濾之 P2P 呼吸週期與 BPM
        bpm_p2p = NaN;
        local_ref_period = global_ref_period;
        p_in_w = peak_times(peak_times >= t_s & peak_times <= t_e);
        if length(p_in_w) >= 2
            p2p_intervals = diff(p_in_w);
            % 若個別間距恰為全域中位數週期的 ~2 倍 (代表中間漏抓 1 個峰)，將其折半還原單一呼吸週期
            if ~isnan(global_ref_period) && global_ref_period > 0
                double_cycle_mask = (p2p_intervals >= 1.65 * global_ref_period) & ...
                                    (p2p_intervals <= 2.35 * global_ref_period);
                p2p_intervals(double_cycle_mask) = 0.5 * p2p_intervals(double_cycle_mask);
            end
            valid_p2p = p2p_intervals(p2p_intervals >= 1.8 & p2p_intervals <= 6.5);
            if ~isempty(valid_p2p)
                Tp2p = median(valid_p2p);
                bpm_p2p = 60 / Tp2p;
                local_ref_period = Tp2p;
            end
        end

        % 論文 Section IV-D Eq. (6)-(7): 自相關函數 (ACF) 提取呼吸週期 (結合基頻防倍頻跳變機制)
        bpm_acf = NaN;
        if ~isempty(best_sig) && length(best_sig) >= idx_end
            win_sig = best_sig(idx_start:idx_end);
            win_inv = gap_mask(idx_start:idx_end) | motion_mask(idx_start:idx_end);
            if sum(~win_inv) > round(5 * Fs_target)
                win_sig = win_sig - mean(win_sig(~win_inv));
                win_sig(win_inv) = 0;
                if std(win_sig) > eps
                    bpm_acf = estimate_bpm_by_acf(win_sig, Fs_target, min_lag_samples, max_lag_samples, local_ref_period);
                end
            end
        end

        % 融合 ACF 與 P2P 估計值
        if ~isnan(bpm_acf) && ~isnan(bpm_p2p)
            if abs(bpm_acf - bpm_p2p) <= 3.5
                bpm_val = 0.6 * bpm_acf + 0.4 * bpm_p2p;
            else
                % 若 ACF 出現倍頻/半頻偏離，以穩健時域波峰中位數 bpm_p2p 為準
                bpm_val = bpm_p2p;
            end
        elseif ~isnan(bpm_p2p)
            bpm_val = bpm_p2p;
        else
            bpm_val = bpm_acf;
        end

        if ~isnan(bpm_val) && bpm_val >= 5 && bpm_val <= 40
            bpm_timeline(i) = bpm_val;
        end
    end

    raw_before_interp = bpm_timeline;

    % 3. 論文 Section IV-D (Fig. 8): 切除體動污染區間並使用線性插值補齊
    valid_pts = ~isnan(bpm_timeline) & ~motion_win_flag;
    if any(motion_win_flag) && sum(valid_pts) >= 2
        t_val = time_axis_bpm(valid_pts);
        b_val = bpm_timeline(valid_pts);
        t_qry = time_axis_bpm(motion_win_flag);

        b_interp = interp1(t_val, b_val, t_qry, 'linear');
        nan_edge = isnan(b_interp);
        if any(nan_edge)
            b_interp(nan_edge) = interp1(t_val, b_val, t_qry(nan_edge), 'nearest', 'extrap');
        end

        bpm_timeline(motion_win_flag) = max(5, min(40, b_interp));
        gap_nan_mask = isnan(raw_before_interp) & ~motion_win_flag;
        bpm_timeline(gap_nan_mask) = NaN;
    elseif any(motion_win_flag) && sum(valid_pts) == 1
        bpm_timeline(motion_win_flag) = bpm_timeline(valid_pts);
    end

    % 4. 計算區段統計特徵 (90th 百分位數、平均值、四分位距 IQR)
    valid_bpm_seg = bpm_timeline(~isnan(bpm_timeline) & bpm_timeline >= 5 & bpm_timeline <= 40);
    if ~isempty(valid_bpm_seg)
        seg_90th = prctile(valid_bpm_seg, 90);
        bpm_mean = mean(valid_bpm_seg);
        bpm_iqr  = prctile(valid_bpm_seg, 75) - prctile(valid_bpm_seg, 25);
        bpm_std  = std(valid_bpm_seg);
    else
        seg_90th = NaN;
        bpm_mean = NaN;
        bpm_iqr  = NaN;
        bpm_std  = NaN;
    end

    bpm_stats = struct( ...
        'seg_90th', seg_90th, ...
        'mean', bpm_mean, ...
        'iqr', bpm_iqr, ...
        'std', bpm_std, ...
        'raw_before_interp', raw_before_interp, ...
        'motion_win_flag', motion_win_flag);

    fprintf('動態 BPM (ACF + 體動線性插值修補) 計算完成！\n');
end

% =========================================================================
% 輔助函式：論文 Eq. (6) & (7) 自相關函數 (ACF) 呼吸率提取 (含防二次諧波倍頻跳變機制)
% =========================================================================
function bpm_acf = estimate_bpm_by_acf(y, Fs, min_lag, max_lag, ref_period_sec)
    bpm_acf = NaN;
    y = y(:);
    Lw = length(y);
    max_lag = min(max_lag, Lw - 2);
    if min_lag >= max_lag
        return;
    end

    % 標準正規化自相關函數 R_yy(tau) (不使用無偏放大，避免長延遲 2T 峰值被人工墊高)
    nfft = 2^nextpow2(2 * Lw - 1);
    Y = fft(y, nfft);
    acf_full = real(ifft(abs(Y).^2));
    if acf_full(1) <= eps
        return;
    end
    acf_pos = acf_full(1:(max_lag + 2)) / acf_full(1); % lag 0 對應索引 1

    [pks, locs, ~, prom] = findpeaks(acf_pos, 'MinPeakProminence', 0.05);
    valid_mask = (locs - 1 >= min_lag) & (locs - 1 <= max_lag) & (pks > 0.08);
    if ~any(valid_mask)
        return;
    end

    cand_locs = locs(valid_mask);
    cand_pks  = pks(valid_mask);
    cand_prom = prom(valid_mask);

    % 若有局部時域 P2P 參考週期，優先在參考週期 +/- 30% 範圍內尋找主峰，徹底消除 T 與 T/2 跳變
    best_idx = [];
    if ~isnan(ref_period_sec) && ref_period_sec > 0
        ref_lag = ref_period_sec * Fs;
        near_mask = abs((cand_locs - 1) - ref_lag) <= 0.30 * ref_lag;
        if any(near_mask)
            near_locs = cand_locs(near_mask);
            near_pks  = cand_pks(near_mask);
            [~, max_near] = max(near_pks);
            best_idx = near_locs(max_near);
        end
    end

    % 若無先驗參考週期，取自相關係數最高的主峰 (並檢查是否為 2T 倍週期)
    if isempty(best_idx)
        [~, max_rel] = max(cand_pks);
        best_idx = cand_locs(max_rel);
        best_lag = best_idx - 1;

        % 檢查半週期處 (best_lag / 2) 是否存在更高或相當的真實基頻峰
        half_mask = abs((cand_locs - 1) - 0.5 * best_lag) <= 0.15 * best_lag;
        if any(half_mask)
            half_locs = cand_locs(half_mask);
            half_pks  = cand_pks(half_mask);
            half_prom = cand_prom(half_mask);
            [max_hp, h_idx] = max(half_pks);
            if max_hp >= 0.85 * cand_pks(max_rel) && half_prom(h_idx) >= 0.70 * cand_prom(max_rel)
                best_idx = half_locs(h_idx);
            end
        end
    end

    % 三點拋物線次樣本內插 (Sub-sample Parabolic Interpolation)
    if best_idx > 1 && best_idx < length(acf_pos)
        y1 = acf_pos(best_idx - 1);
        y2 = acf_pos(best_idx);
        y3 = acf_pos(best_idx + 1);
        denom = (y1 - 2 * y2 + y3);
        if abs(denom) > eps
            delta = 0.5 * (y1 - y3) / denom;
            delta = max(-0.5, min(0.5, delta));
        else
            delta = 0;
        end
        true_lag = (best_idx - 1) + delta;
    else
        true_lag = best_idx - 1;
    end

    period_sec = true_lag / Fs;
    if period_sec > 0
        bpm_acf = 60 / period_sec;
    end
end
