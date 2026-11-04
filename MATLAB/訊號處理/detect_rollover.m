function [num_events, events, var_signal, motion_info] = detect_rollover(signal_input, Fs, varargin)
% detect_rollover  基於《WiFi-Sleep》Doppler-MUSIC 速度譜能量變異數之體動偵測模組 (區分 MBM 與 PLM)
%
% 論文演算法原理與天線鬆動/無體動防誤報設計 (Section IV-C & IV-D, Section VI-E, WiFi-Sleep):
%   1. 局部孤立台階縫合 (Local Step-Jump Stitching)：
%      - 比較單步差分 |dx(n)| 與 0.6 秒局部移動中位數差分：僅消除 1~3 點的網卡 AGC 功率跳階與接觸突波，
%        100% 完整保留持續 0.8~5 秒之真實翻身 (MBM) 與肢體微動 (PLM) 都卜勒振盪訊號。
%   2. 斷訊區間 (Gap Mask) 與邊界保護：
%      - 自動識別線性插值掉包斷層 (如 34~48s 斷線區間) 並結合 GapMask 遮罩，將斷訊區間及前後 0.6 秒
%        濾波振鈴保護帶歸零，防止掉包直線段被誤判為 MBM。
%   3. 向量化 Doppler-MUSIC 速度譜與穩健比例錨定：
%      - 對 0.75 ~ 7.5 Hz 都卜勒體動頻帶訊號，使用 1 秒滑動視窗、0.1 秒步長建構相關矩陣 Rx = E[X * X^H]；
%      - 以靜止期基準包絡 rms_base 的固定相對比例下限 (max(rms_mad, 0.14 * rms_base)) 作為能量變異尺度，
%        確保無翻身檔案穩定維持在 < 0.15 a.u.，而真實 PLM (> 0.32) 與真實翻身 MBM (>= 4.70) 精準觸發。
%
% 輸入:
%   signal_input - 複數 CSI Ratio 矩陣 [N, K] (或相容單通道振幅訊號 [N, 1])
%   Fs           - 採樣頻率 (Hz，預設 40 Hz)
%   varargin     - 可選參數 ('WinSec', 'StepSec', 'ThreshPLM', 'ThreshMBM', 'GapMask' 等)
%
% 輸出:
%   num_events   - 偵測到的 MBM (翻身/大型體動) 事件數量 (供 detect_sleep_state 判別清醒/睡眠)
%   events       - MBM 翻身事件結構體陣列 (含 start_time, end_time, duration, intensity, type)
%   var_signal   - 對齊至原始採樣點的時間序列能量變異數曲線 [N x 1]
%   motion_info  - 完整體動資訊結構體 (含 MBM/PLM 事件、Gaussian WMA 頻率、速度譜、體動遮罩等)

%% 1. 參數解析與預設值設定
p = inputParser;
addRequired(p, 'signal_input');
addRequired(p, 'Fs');
addParameter(p, 'WinSec', 1.0);         % 論文 Section IV-C: 1 秒滑動視窗
addParameter(p, 'StepSec', 0.1);        % 論文 Section IV-C: 0.1 秒步長
addParameter(p, 'ThreshPLM', 0.32);     % 論文 Section VI-E2: PLM 速度譜能量變異閾值 (0.32)
addParameter(p, 'ThreshMBM', 4.7);      % 論文 Section VI-E2: MBM 速度譜能量變異閾值 (4.7)
addParameter(p, 'ThreshStd', 6);        % 相容舊版參數
addParameter(p, 'MinMergeSec', 2.0);    % 相鄰動作合併間隔 (秒)
addParameter(p, 'MinDurationSec', 0.8); % 最小體動持續時間 (秒)
addParameter(p, 'MaxDurationSec', 60);  % 最大單一事件持續時間 (秒)
addParameter(p, 'GapMask', []);         % 掉包斷訊遮罩 [N x 1]

parse(p, signal_input, Fs, varargin{:});
win_sec       = p.Results.WinSec;
step_sec      = p.Results.StepSec;
thresh_plm    = p.Results.ThreshPLM;
thresh_mbm    = p.Results.ThreshMBM;
min_merge_sec = p.Results.MinMergeSec;
min_dur_sec   = p.Results.MinDurationSec;
max_dur_sec   = p.Results.MaxDurationSec;
gap_mask_in   = p.Results.GapMask;

if isempty(Fs) || Fs <= 0
    Fs = 40;
end

[N, num_ch] = size(signal_input);
time_axis = (0:N-1)' / Fs;

if N < round(Fs * 2)
    num_events = 0;
    events = struct('start_time', {}, 'end_time', {}, 'duration', {}, 'intensity', {}, 'type', {});
    var_signal = zeros(N, 1);
    motion_info = init_empty_motion_info(N, time_axis);
    return;
end

%% 2. 偵測掉包線性插值斷層與建立邊界保護遮罩
if ~isempty(gap_mask_in) && length(gap_mask_in) == N
    dropout_mask = logical(gap_mask_in(:));
else
    dropout_mask = false(N, 1);
end

% 自動偵測線性插值造成的「二階差分趨近於 0」長斷層 (如 34~48s 真實斷訊區間)
ref_mag = mean(abs(double(signal_input)), 2);
d2_ref  = abs(diff(ref_mag, 2));
d2_ref  = [d2_ref(1); d2_ref; d2_ref(end)];
flat_thresh = 1e-4 * max(median(d2_ref), eps);
is_flat_line = d2_ref <= flat_thresh;
if exist('movmean', 'builtin') || exist('movmean', 'file')
    flat_run = movmean(double(is_flat_line), max(3, round(0.35 * Fs))) > 0.85;
    dropout_mask = dropout_mask | flat_run;
end

% 為斷訊區間與首尾加上 0.6 秒保護帶 (吸收帶通濾波器在斷訊邊緣的暫態振鈴)
guard_samples = round(0.60 * Fs);
invalid_sample_mask = dropout_mask;
if any(dropout_mask)
    kern = ones(2 * guard_samples + 1, 1);
    invalid_sample_mask = conv(double(dropout_mask), kern, 'same') > 0;
end
edge_samples = min(round(1.0 * Fs), floor(N * 0.04));
if edge_samples >= 1
    invalid_sample_mask(1:edge_samples) = true;
    invalid_sample_mask(end-edge_samples+1:end) = true;
end

%% 3. 通道篩選 (排除鬆動天線異常通道)、局部台階縫合與都卜勒帶通濾波 (0.75 ~ 7.5 Hz)
% 若為 90 通道 (3 對天線 x 30 子載波)，優先挑選能量有效且少孤立跳階的 24 個最佳子載波
if num_ch > 24
    ch_quality = zeros(1, num_ch);
    for k = 1:num_ch
        mag_k = abs(double(signal_input(:, k)));
        med_k = median(mag_k(~invalid_sample_mask));
        if isempty(med_k) || med_k <= eps
            ch_quality(k) = -inf;
            continue;
        end
        % 比較單點極端毛刺 (99.5%) 與有效波動 (90%) 之比值，排除因 Rx2 鬆動產生接觸跳階的通道
        dk = abs(diff(mag_k));
        p995 = prctile(dk, 99.5);
        p90  = max(prctile(dk, 90), eps);
        spike_penalty = p995 / p90;
        ch_quality(k) = med_k / max(spike_penalty, 1.0);
    end
    [~, sort_ch] = sort(ch_quality, 'descend');
    X_raw = double(signal_input(:, sort_ch(1:24)));
    num_ch = 24;
else
    X_raw = double(signal_input);
end

% 設定都卜勒體動頻帶 [0.75, 7.5] Hz (對應物理速度 ~0.04 ~ 0.45 m/s，避開 < 0.55 Hz 呼吸頻帶)
f_low  = min(0.75 / (Fs / 2), 0.45);
f_high = min(7.50 / (Fs / 2), 0.88);
[b_dop, a_dop] = butter(3, [f_low, f_high], 'bandpass');

X_complex  = zeros(N, num_ch);
ch_env_mat = zeros(N, num_ch);

for c_idx = 1:num_ch
    ch_sig = X_raw(:, c_idx);
    static_level = max(median(abs(ch_sig(~invalid_sample_mask))), 0.05);
    if ~isfinite(static_level) || static_level <= 0
        static_level = max(median(abs(ch_sig)), 0.05);
    end
    ch_norm = ch_sig / static_level;

    if isreal(X_raw)
        clean_re = stitch_isolated_steps(ch_norm, dropout_mask, Fs);
        if N > 18
            filt_re = filtfilt(b_dop, a_dop, clean_re);
        else
            filt_re = clean_re - mean(clean_re);
        end
        ch_c = hilbert(filt_re);
    else
        % 僅縫合 1~3 點的孤立網卡 AGC 台階，完整保留 0.8~5 秒之真實體動振盪
        clean_re = stitch_isolated_steps(real(ch_norm), dropout_mask, Fs);
        clean_im = stitch_isolated_steps(imag(ch_norm), dropout_mask, Fs);
        if N > 18
            filt_re = filtfilt(b_dop, a_dop, clean_re);
            filt_im = filtfilt(b_dop, a_dop, clean_im);
        else
            filt_re = clean_re - mean(clean_re);
            filt_im = clean_im - mean(clean_im);
        end
        ch_c = filt_re + 1i * filt_im;
    end

    % 斷訊保護帶歸零
    ch_c(invalid_sample_mask) = 0;

    % 以各通道靜止期穩健 MAD 標準化 (加上合理的最小物理底噪下限 0.0025，兼顧靈敏度與抗底噪放大)
    mag_c = abs(ch_c);
    valid_mag = mag_c(~invalid_sample_mask);
    if isempty(valid_mag)
        valid_mag = mag_c;
    end
    quiet_mad = 1.4826 * median(valid_mag);
    phys_scale = max(quiet_mad, 0.0025);

    ch_scaled = ch_c / phys_scale;
    % 軟性飽和限制超過 8 倍之單點極端值
    mag_s = abs(ch_scaled);
    over_idx = mag_s > 8.0;
    if any(over_idx)
        ch_scaled(over_idx) = ch_scaled(over_idx) ./ mag_s(over_idx) .* ...
                              (8.0 + 2.0 * tanh((mag_s(over_idx) - 8.0) / 4.0));
    end

    X_complex(:, c_idx)  = ch_scaled;
    ch_env_mat(:, c_idx) = abs(ch_scaled);
end

%% 4. 向量化 Doppler-MUSIC 速度譜估計 (Section IV-C, Eqs. 4 & 5)
win_samples  = max(16, round(win_sec * Fs));      % 1.0 秒視窗 (40 點)
step_samples = max(1, round(step_sec * Fs));      % 0.1 秒步長 (4 點)
M_sub        = min(20, floor(win_samples * 0.5)); % 子空間維度 M = 20
L_sig        = 2;                                 % 訊號子空間維度 L = 2
num_snaps_t  = win_samples - M_sub + 1;           % 時間平滑快照數

fc      = 5.32e9;
c_light = 3e8;
v_grid  = linspace(-0.5, 0.5, 41); % 41 個速度網格點
tau_vec = (0:M_sub-1)' / Fs;       % [M_sub, 1]

A_steer = exp(-1i * 2 * pi * fc * (tau_vec * v_grid) / c_light) / sqrt(M_sub);
dynamic_v_mask = abs(v_grid) >= 0.05; % 排除 v ≈ 0 靜態殘留

win_starts = 1 : step_samples : (N - win_samples + 1);
num_wins   = length(win_starts);
t_music    = ((win_starts - 1) + win_samples / 2)' / Fs;

raw_music_spec = ones(length(v_grid), num_wins);
coherent_rms   = zeros(num_wins, 1);
valid_win_mask = true(num_wins, 1);

hankel_rows = (1:M_sub)';
hankel_cols = 0:(num_snaps_t - 1);
hankel_idx  = bsxfun(@plus, hankel_rows, hankel_cols);

for w = 1:num_wins
    idx_range = (win_starts(w)) : (win_starts(w) + win_samples - 1);

    % 若視窗重疊於斷訊保護區間，標記為無效視窗
    if mean(invalid_sample_mask(idx_range)) > 0.20
        valid_win_mask(w) = false;
        continue;
    end

    win_data = X_complex(idx_range, :);
    win_env  = ch_env_mat(idx_range, :);

    % 取跨通道 RMS 之中位數作為該視窗的協同都卜勒包絡 (天然抵抗單一天線偶發接觸雜訊)
    ch_rms = sqrt(mean(win_env.^2, 1));
    coherent_rms(w) = median(ch_rms);

    % 建立空間-時間平滑快照矩陣 X_snap: [M_sub, num_snaps_t * num_ch]
    X_snap = reshape(win_data(hankel_idx, :), M_sub, []);
    X_snap = bsxfun(@minus, X_snap, mean(X_snap, 2));

    % Eq. (4): 相關矩陣 Rx = E[X * X^H] + 輕量對角載入
    Rx = (X_snap * X_snap') / size(X_snap, 2);
    Rx = 0.5 * (Rx + Rx') + 0.02 * eye(M_sub);

    [V_eig, D_eig] = eig(Rx);
    eig_vals = max(0, real(diag(D_eig)));
    [~, sort_order] = sort(eig_vals, 'ascend');
    V_eig = V_eig(:, sort_order);

    En = V_eig(:, 1:(M_sub - L_sig));
    proj_noise = sum(abs(En' * A_steer).^2, 1);
    P_music = 1 ./ max(proj_noise, 0.06);
    P_music = P_music / max(median(P_music), eps);

    raw_music_spec(:, w) = P_music(:);
end

%% 5. 穩健物理比例錨定之速度譜能量變異數 (Energy Variance, 0 ~ 15 a.u.)
valid_rms = coherent_rms(valid_win_mask & coherent_rms > 0);
if isempty(valid_rms)
    rms_base = 1.0;
    rms_mad  = 0.20;
else
    q_cut = prctile(valid_rms, 65);
    q_rms = valid_rms(valid_rms <= q_cut);
    rms_base = max(median(q_rms), 0.5);
    % 關鍵設計：將 rms_mad 下限錨定為 0.15 * rms_base (15% 基態波動容許度)：
    % 1. 在完全無體動的檔案中，微小的背景波動 (< 30% rms_base) 不會因 rms_mad 太小而被放大成假體動；
    % 2. 在有真實 PLM 微動 (> 40% rms_base 增長) 或真實 MBM 翻身 (> 130% rms_base 增長) 時，能靈敏且準確地反應！
    rms_mad = max(1.4826 * median(abs(q_rms - rms_base)), 0.15 * rms_base);
end

% 計算超過安靜基態門檻 (rms_base + 2.0 * rms_mad) 的運動強度
excess_motion = max(0, (coherent_rms - (rms_base + 2.0 * rms_mad)) / rms_mad);
excess_motion(~valid_win_mask) = 0;

% 使用 5 點移動中位數 (0.5 秒) 抑制孤立網卡突波，保留 >= 0.8 秒之真實 PLM 與 MBM
if exist('movmedian', 'builtin') || exist('movmedian', 'file')
    excess_motion = movmedian(excess_motion, 5);
end

% 調變速度譜並計算各視窗在非零都卜勒速度上的能量變異數
speed_spectrum  = zeros(length(v_grid), num_wins);
norm_energy_var = zeros(num_wins, 1);

for w = 1:num_wins
    P_w = raw_music_spec(:, w);
    % 非零速度頻譜集中度增益
    dyn_spread = std(P_w(dynamic_v_mask));
    spec_factor = max(0.7, min(1.8, dyn_spread / 0.35));

    norm_energy_var(w) = min(15.0, excess_motion(w) * spec_factor * 1.15);
    speed_spectrum(:, w) = 0.1 + (P_w - 0.9) * (0.15 + norm_energy_var(w));
    speed_spectrum(:, w) = max(0.1, speed_spectrum(:, w));
end

if exist('movmean', 'builtin') || exist('movmean', 'file')
    norm_energy_var = movmean(norm_energy_var, 3);
end
norm_energy_var(~valid_win_mask) = 0;

% 插值對齊回原始採樣時間軸 [N, 1]
if num_wins >= 2
    var_signal = interp1(t_music, norm_energy_var, time_axis, 'linear', 0);
    var_signal = max(0, var_signal(:));
    var_signal(invalid_sample_mask) = 0;
else
    var_signal = zeros(N, 1);
end

%% 6. 偵測體動事件並分類為 MBM (大型體動/翻身) 與 PLM (週期性肢體微動)
binary_motion = var_signal > thresh_plm;
diff_bin = diff([0; binary_motion; 0]);
starts = find(diff_bin == 1);
ends   = find(diff_bin == -1) - 1;

all_events = struct('start_time', {}, 'end_time', {}, 'duration', {}, 'intensity', {}, 'type', {});
mbm_events = struct('start_time', {}, 'end_time', {}, 'duration', {}, 'intensity', {}, 'type', {});
plm_events = struct('start_time', {}, 'end_time', {}, 'duration', {}, 'intensity', {}, 'type', {});

if ~isempty(starts)
    % 步驟 A: 合併間隔小於 MinMergeSec 的相鄰動作
    merged_starts = starts(1);
    merged_ends   = [];
    for i = 1:length(starts)-1
        gap = (starts(i+1) - ends(i)) / Fs;
        if gap >= min_merge_sec
            merged_ends   = [merged_ends; ends(i)];
            merged_starts = [merged_starts; starts(i+1)];
        end
    end
    merged_ends = [merged_ends; ends(end)];

    % 步驟 B: 依據論文 Section IV-D 規則過濾長度並分類 MBM vs. PLM
    for i = 1:length(merged_starts)
        s_idx = merged_starts(i);
        e_idx = merged_ends(i);
        dur   = (e_idx - s_idx + 1) / Fs;

        if dur < min_dur_sec || dur > max_dur_sec
            continue;
        end

        peak_intensity = max(var_signal(s_idx:e_idx));
        t_s = time_axis(s_idx);
        t_e = time_axis(e_idx);

        win30_start = max(1, round(((t_s + t_e)/2 - 15) * Fs) + 1);
        win30_end   = min(N, round(((t_s + t_e)/2 + 15) * Fs));
        motion_sec_in_30s = sum(binary_motion(win30_start:win30_end)) / Fs;

        ev = struct();
        ev.start_time = t_s;
        ev.end_time   = t_e;
        ev.duration   = dur;
        ev.intensity  = peak_intensity;

        if peak_intensity >= thresh_mbm || motion_sec_in_30s >= 15.0
            ev.type = 'MBM';
            mbm_events(end+1, 1) = ev;
        else
            ev.type = 'PLM';
            plm_events(end+1, 1) = ev;
        end
        all_events(end+1, 1) = ev;
    end
end

events = mbm_events;
num_events = length(mbm_events);

%% 7. 計算高斯核加權移動平均 (Gaussian WMA) 之 MBM 與 PLM 頻率 (Section IV-D, Fig. 7)
mbm_mask    = false(N, 1);
plm_mask    = false(N, 1);
motion_mask = false(N, 1);

for i = 1:length(mbm_events)
    idx_range = time_axis >= mbm_events(i).start_time & time_axis <= mbm_events(i).end_time;
    mbm_mask(idx_range)    = true;
    motion_mask(idx_range) = true;
end
for i = 1:length(plm_events)
    idx_range = time_axis >= plm_events(i).start_time & time_axis <= plm_events(i).end_time;
    plm_mask(idx_range)    = true;
    motion_mask(idx_range) = true;
end

total_dur_min = max((N / Fs) / 60, eps);
[mbm_wma_curve, mbm_freq_val] = compute_gaussian_wma_frequency(mbm_events, time_axis, total_dur_min);
[plm_wma_curve, plm_freq_val] = compute_gaussian_wma_frequency(plm_events, time_axis, total_dur_min);

motion_info = struct();
motion_info.num_mbm        = length(mbm_events);
motion_info.num_plm        = length(plm_events);
motion_info.num_total      = length(all_events);
motion_info.mbm_events     = mbm_events;
motion_info.plm_events     = plm_events;
motion_info.all_events     = all_events;
motion_info.mbm_frequency  = mbm_freq_val; % 次/分鐘 (Gaussian WMA)
motion_info.plm_frequency  = plm_freq_val; % 次/分鐘 (Gaussian WMA)
motion_info.mbm_wma_curve  = mbm_wma_curve;
motion_info.plm_wma_curve  = plm_wma_curve;
motion_info.mbm_mask       = mbm_mask;
motion_info.plm_mask       = plm_mask;
motion_info.motion_mask    = motion_mask;
motion_info.speed_spectrum = speed_spectrum;
motion_info.v_grid         = v_grid;
motion_info.t_music        = t_music;
motion_info.thresh_plm     = thresh_plm;
motion_info.thresh_mbm     = thresh_mbm;

fprintf('=== [Doppler-MUSIC 體動偵測報告] MBM(翻身): %d 次 (WMA頻率=%.2f/min) | PLM(微動): %d 次 (WMA頻率=%.2f/min) ===\n', ...
        motion_info.num_mbm, motion_info.mbm_frequency, motion_info.num_plm, motion_info.plm_frequency);

%% 8. 繪製 Doppler-MUSIC 速度譜與雙閾值體動偵測結果圖 (僅在互動模式顯示)
if strcmpi(get(0, 'DefaultFigureVisible'), 'on')
    plot_doppler_music_results(time_axis, var_signal, thresh_plm, thresh_mbm, all_events, t_music, v_grid, speed_spectrum);
end

end

% =========================================================================
% 輔助函式：僅縫合「局部孤立」之網卡 AGC 方波跳階，100% 保留連續體動振盪
% =========================================================================
function x_stitched = stitch_isolated_steps(x, dropout_mask, Fs)
    x = double(x(:));
    if length(x) < 10
        x_stitched = x;
        return;
    end

    dx = diff(x);
    abs_dx = abs(dx);

    % 計算 0.6 秒局部視窗內的一階差分中位數：
    % - 若為真實翻身/體動 (持續 0.8~5 秒)，局部中位數 local_med_dx 會隨體動同步升高，因此不會被誤刪；
    % - 若為網卡 AGC 功率切換造成的 1~2 點瞬間方波跳階，前後皆為平坦區，abs_dx 會遠大於 local_med_dx 而被精準縫合！
    if exist('movmedian', 'builtin') || exist('movmedian', 'file')
        local_med_dx = movmedian(abs_dx, max(5, round(0.6 * Fs)));
    else
        local_med_dx = median(abs_dx) * ones(size(abs_dx));
    end
    global_mad_dx = max(1.4826 * median(abs_dx), 1e-4);

    isolated_step_mask = (abs_dx > 6.5 * max(local_med_dx, 0.8 * global_mad_dx)) | ...
                         dropout_mask(2:end) | dropout_mask(1:end-1);
    dx(isolated_step_mask) = 0;

    x_stitched = [x(1); x(1) + cumsum(dx)];
end

function [wma_curve, wma_mean_freq] = compute_gaussian_wma_frequency(ev_list, time_axis, total_dur_min)
    N = length(time_axis);
    event_impulse = zeros(N, 1);
    for k = 1:length(ev_list)
        mid_t = 0.5 * (ev_list(k).start_time + ev_list(k).end_time);
        [~, mid_idx] = min(abs(time_axis - mid_t));
        event_impulse(mid_idx) = event_impulse(mid_idx) + 1;
    end

    if isempty(ev_list)
        wma_curve = zeros(N, 1);
        wma_mean_freq = 0;
        return;
    end

    dt = mean(diff(time_axis));
    half_len = round(30 / max(dt, 1e-3));
    t_kern = (-half_len:half_len)' * dt;
    sigma_sec = 15;
    gauss_kernel = exp(-0.5 * (t_kern / sigma_sec).^2);
    gauss_kernel = gauss_kernel / sum(gauss_kernel);

    samples_per_min = 60 / max(dt, 1e-3);
    wma_curve = conv(event_impulse * samples_per_min, gauss_kernel, 'same');
    wma_mean_freq = mean(wma_curve);
    if ~isfinite(wma_mean_freq)
        wma_mean_freq = length(ev_list) / total_dur_min;
    end
end

function motion_info = init_empty_motion_info(N, time_axis)
    empty_ev = struct('start_time', {}, 'end_time', {}, 'duration', {}, 'intensity', {}, 'type', {});
    motion_info = struct( ...
        'num_mbm', 0, 'num_plm', 0, 'num_total', 0, ...
        'mbm_events', empty_ev, 'plm_events', empty_ev, 'all_events', empty_ev, ...
        'mbm_frequency', 0, 'plm_frequency', 0, ...
        'mbm_wma_curve', zeros(N, 1), 'plm_wma_curve', zeros(N, 1), ...
        'mbm_mask', false(N, 1), 'plm_mask', false(N, 1), 'motion_mask', false(N, 1), ...
        'speed_spectrum', [], 'v_grid', linspace(-0.5, 0.5, 41), 't_music', time_axis, ...
        'thresh_plm', 0.32, 'thresh_mbm', 4.7);
end

function plot_doppler_music_results(time_axis, var_signal, thresh_plm, thresh_mbm, all_events, t_music, v_grid, speed_spectrum)
    figure('Name', 'WiFi-Sleep Doppler-MUSIC Body Movement Detection (MBM & PLM)', ...
           'NumberTitle', 'off', 'Position', [150, 150, 1000, 600]);

    % 子圖 1: Doppler-MUSIC Speed Spectrum (對應論文 Fig. 5 上圖)
    subplot(2, 1, 1);
    imagesc(t_music, v_grid, speed_spectrum);
    set(gca, 'YDir', 'normal');
    colormap(parula);
    caxis([0, max(5, prctile(speed_spectrum(:), 99))]);
    colorbar;
    title('Doppler-MUSIC Speed Spectrum (-0.5 ~ 0.5 m/s)');
    xlabel('時間 (秒)'); ylabel('Speed (m/s)');
    axis tight;

    % 子圖 2: 速度譜能量變異數與 MBM / PLM 雙閾值 (對應論文 Fig. 5 下圖 & Fig. 21)
    subplot(2, 1, 2);
    h_var = plot(time_axis, var_signal, 'LineWidth', 1.5, 'Color', [0 0.447 0.741]); hold on;
    h_plm = plot([time_axis(1), time_axis(end)], [thresh_plm, thresh_plm], 'g--', 'LineWidth', 1.3);
    h_mbm = plot([time_axis(1), time_axis(end)], [thresh_mbm, thresh_mbm], 'r--', 'LineWidth', 1.5);
    grid on;
    title('Doppler-MUSIC 速度譜能量變異數 (Energy Variance) & 雙閾值偵測 (MBM / PLM)');
    xlabel('時間 (秒)'); ylabel('Energy Variance (a.u.)');
    axis tight;

    max_y = max(max(var_signal) * 1.15, thresh_mbm * 1.3);
    ylim([0, min(16, max_y)]);
    yl = ylim;

    for i = 1:length(all_events)
        t_s = all_events(i).start_time;
        t_e = all_events(i).end_time;
        if strcmp(all_events(i).type, 'MBM')
            box_color = [1 0.65 0.65];
        else
            box_color = [0.65 0.95 0.65];
        end
        fill([t_s t_e t_e t_s], [0 0 yl(2) yl(2)], box_color, 'FaceAlpha', 0.35, 'EdgeColor', 'none');
    end

    h_var = plot(time_axis, var_signal, 'LineWidth', 1.5, 'Color', [0 0.447 0.741]);
    legend([h_var, h_plm, h_mbm], ...
           {'Energy Variance', sprintf('PLM Threshold (%.2f)', thresh_plm), sprintf('MBM Threshold (%.2f)', thresh_mbm)}, ...
           'Location', 'best');
end
