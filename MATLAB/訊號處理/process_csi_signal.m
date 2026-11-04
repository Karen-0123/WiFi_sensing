function [amp_pc1_norm, phase_pc1_norm, csi_ratio_out, mrc_pca_details] = process_csi_signal(csi_resampled, Fs_target)
% process_csi_signal  基於《WiFi-Sleep》論文之 CSI Ratio + 天線品質自適應 MRC-PCA 呼吸波形增強模組
%
% 論文演算法流程與天線鬆動抗干擾設計 (Section IV-A & IV-B, WiFi-Sleep, IEEE IoT Journal 2021):
%   [N, 30, 2, 3] 重採樣後 CSI 矩陣
%       |
%       +-- Step 1: 計算接收天線對之穩健 CSI Ratio (相除商數) 與天線對健康度評估
%              H_ratio_12 = Rx1 ./ Rx2, H_ratio_13 = Rx1 ./ Rx3, H_ratio_23 = Rx2 ./ Rx3
%              自動檢測是否有鬆動或接觸不良之天線 (如 Rx2 鬆動造成 0.05~0.16 Hz 慢速阻抗漂移與跳階)，
%              自動提高穩定天線對 (如 Rx1/Rx3) 之權重並壓抑異常天線對之慢速漂移。
%       |
%       +-- Step 2: 估計各子載波之生理呼吸頻帶 SNR (MRC 權重計算)
%              計算「正常睡眠呼吸頻帶 (0.18~0.52 Hz, 10.8~31.2 bpm) 能量」對
%              「低頻天線漂移 (<0.18 Hz) + 高頻雜訊 (>0.52 Hz)」之比值，防止 5~7 秒慢速漂移主導輸出。
%       |
%       +-- Step 3: 動態視窗 Savitzky-Golay 平滑 (1.5 秒視窗，保留真實 3~4 秒呼吸週期不失真)
%       |
%       +-- Step 4: 利用 PCA 修正不同 Fresnel Zone 子載波之正負反相問題 (MRC-PCA)
%       |
%       +-- Step 5: 結合振幅與相位互補資訊消除盲區 (Blind Spots) 並選出最佳 Tx
%
% 輸入:
%   csi_resampled   - 重採樣後之 4D CSI 矩陣 [N, 30, 2, 3]
%   Fs_target       - 採樣率 (Hz，預設 40 Hz)
%
% 輸出:
%   amp_pc1_norm    - 選定 Tx 之 MRC-PCA 融合振幅呼吸訊號 (Z-score) [N, 1]
%   phase_pc1_norm  - 選定 Tx 之 MRC-PCA 融合相位呼吸訊號 (Z-score) [N, 1]
%   csi_ratio_out   - 選定 Tx 之經過天線健康度加權之複數 CSI Ratio 矩陣 [N, 90] (供 Doppler-MUSIC 使用)
%   mrc_pca_details - 包含關聯波形與融合資訊之結構體

    if nargin < 2 || isempty(Fs_target)
        Fs_target = 40;
    end

    if isempty(csi_resampled)
        error('輸入的 CSI 矩陣為空！');
    end

    [N, Nsc, Ntx, Nrx] = size(csi_resampled);
    if Ntx ~= 2 || Nrx ~= 3 || Nsc ~= 30
        error('輸入 CSI 維度必須為 [N, 30, 2, 3]');
    end

    if N < 8
        error('CSI 資料長度不足，無法進行 MRC-PCA 訊號處理');
    end

    % 正常人類睡眠呼吸頻率範圍: 0.18 Hz ~ 0.52 Hz (10.8 ~ 31.2 bpm，週期 1.92 ~ 5.55 秒)
    % 將下界由 0.10 Hz 提升至 0.18 Hz，徹底濾除天線鬆動 (如 Rx2 接觸不良) 產生的 5.5~10 秒慢速阻抗漂移
    resp_band = [0.18, 0.52];
    pair_defs = [1 2; 1 3; 2 3];
    pair_names = {'Rx1/Rx2', 'Rx1/Rx3', 'Rx2/Rx3'};
    num_pairs = size(pair_defs, 1);
    num_channels = Nsc * num_pairs; % 90 個 CSI Ratio 通道

    % 設計生理呼吸頻帶帶通濾波器 [0.18, 0.52] Hz
    [b_bp, a_bp] = butter(3, resp_band / (Fs_target / 2), 'bandpass');
    % 設計 0.15 Hz 高通濾波器 (供 SG 波形去除天線鬆動造成之 6~10 秒慢速基線漂移)
    [b_hp, a_hp] = butter(2, 0.15 / (Fs_target / 2), 'high');

    % 動態計算 Savitzky-Golay 視窗大小 (1.5 秒視窗，在 40Hz 下為 61 點，避免 3 秒大視窗把正常呼吸抹慢成 5~6 秒)
    window_length = sg_window_length(N, Fs_target, 3);

    amp_fused_by_tx   = zeros(N, Ntx);
    phase_fused_by_tx = zeros(N, Ntx);
    joint_fused_by_tx = zeros(N, Ntx);
    amp_sg_by_tx      = zeros(N, Ntx);
    phase_sg_by_tx    = zeros(N, Ntx);
    joint_sg_by_tx    = zeros(N, Ntx);
    csi_ratio_by_tx   = cell(1, Ntx);

    amp_explained     = zeros(Ntx, 1);
    phase_explained   = zeros(Ntx, 1);
    band_focus        = zeros(Ntx, 1);
    selection_score   = zeros(Ntx, 1);
    avg_weights       = zeros(num_pairs, Ntx);

    fprintf('====== [process_csi_signal] 開始：CSI Ratio + 天線健康度自適應 MRC-PCA 融合 ======\n');

    for tx = 1:Ntx
        % -----------------------------------------------------------------
        % Step 1: 計算接收天線對之穩健 CSI Ratio 並檢測鬆動天線對 (Rx1/Rx2, Rx1/Rx3, Rx2/Rx3)
        % -----------------------------------------------------------------
        ratio_complex = zeros(N, num_channels);
        pair_health   = ones(1, num_pairs);

        for p = 1:num_pairs
            rx_a = pair_defs(p, 1);
            rx_b = pair_defs(p, 2);
            sig_a = squeeze(csi_resampled(:, :, tx, rx_a)); % [N, 30]
            sig_b = squeeze(csi_resampled(:, :, tx, rx_b)); % [N, 30]

            % 穩健分母下界保護：防止分母天線 (如鬆動的 Rx2) 瞬時深衰落時產生除零尖峰
            denom_power = abs(sig_b).^2;
            med_power   = median(denom_power, 1); % [1, 30]
            safe_denom  = bsxfun(@max, denom_power, 0.25 * max(med_power, eps));

            col_range = (p - 1) * Nsc + (1:Nsc);
            pair_ratio = (sig_a .* conj(sig_b)) ./ safe_denom;
            ratio_complex(:, col_range) = pair_ratio;

            % 評估該天線對之穩定度 (檢測是否有如 Rx2 天線鬆動導致的跳階與超低頻漂移)
            pair_health(p) = evaluate_pair_health(pair_ratio, Fs_target, resp_band);
        end
        pair_health = pair_health / max(pair_health);

        % 若含 Rx2 的兩組天線對 (Rx1/Rx2, Rx2/Rx3) 健康度明顯低於 Rx1/Rx3，壓抑異常天線對在 csi_ratio_out 的干擾
        ratio_weighted = ratio_complex;
        for p = 1:num_pairs
            col_range = (p - 1) * Nsc + (1:Nsc);
            ratio_weighted(:, col_range) = ratio_complex(:, col_range) * pair_health(p);
        end
        csi_ratio_by_tx{tx} = ratio_weighted;

        % 提取 CSI Ratio 之振幅與連續展開相位
        raw_amp   = abs(ratio_complex);                  % [N, 90]
        raw_phase = unwrap(angle(ratio_complex), [], 1); % [N, 90]

        % 移除各通道階躍、低頻趨勢並進行穩健 Z-score 標準化
        norm_amp   = zeros(N, num_channels);
        norm_phase = zeros(N, num_channels);
        for ch = 1:num_channels
            norm_amp(:, ch)   = clean_and_zscore_channel(raw_amp(:, ch), Fs_target);
            norm_phase(:, ch) = clean_and_zscore_channel(raw_phase(:, ch), Fs_target);
        end

        % 動態視窗 Savitzky-Golay 濾波 (1.5 秒視窗，濾除高頻雜訊並保留完整 3~5 秒呼吸波形)
        if window_length > 3
            fprintf('[Tx%d] 動態視窗 Savitzky-Golay 濾波 (window=%d, poly=3)...\n', tx, window_length);
            sg_amp   = sgolayfilt(norm_amp, 3, window_length);
            sg_phase = sgolayfilt(norm_phase, 3, window_length);
        else
            sg_amp   = norm_amp;
            sg_phase = norm_phase;
            warning('[Tx%d] 資料長度不足以設定視窗大小，跳過 Savitzky-Golay 濾波', tx);
        end

        % -----------------------------------------------------------------
        % Step 2: 估計全 90 個子載波通道之呼吸頻帶 SNR (結合天線對健康度權重)
        % -----------------------------------------------------------------
        snr_amp   = zeros(1, num_channels);
        snr_phase = zeros(1, num_channels);
        for p = 1:num_pairs
            col_range = (p - 1) * Nsc + (1:Nsc);
            for ch = col_range
                snr_amp(ch)   = estimate_subcarrier_snr(sg_amp(:, ch), Fs_target, resp_band) * (pair_health(p)^2);
                snr_phase(ch) = estimate_subcarrier_snr(sg_phase(:, ch), Fs_target, resp_band) * (pair_health(p)^2);
            end
        end

        % 記錄各 Rx pair 的平均權重供診斷視覺化
        total_snr_amp   = max(sum(snr_amp), eps);
        total_snr_phase = max(sum(snr_phase), eps);
        for p = 1:num_pairs
            col_range = (p - 1) * Nsc + (1:Nsc);
            avg_weights(p, tx) = 0.5 * (sum(snr_amp(col_range)) / total_snr_amp + ...
                                        sum(snr_phase(col_range)) / total_snr_phase);
            fprintf('[Tx%d] 平均 MRC 權重佔比 %s = %.3f (天線對穩定度=%.2f)\n', ...
                    tx, pair_names{p}, avg_weights(p, tx), pair_health(p));
        end

        % -----------------------------------------------------------------
        % Step 3 & Step 4: 論文 MRC-PCA 演算法 (篩選高 SNR 子載波進行方向對齊與加權合成)
        % -----------------------------------------------------------------
        [amp_fused, amp_sg_fused, exp_amp] = run_mrc_pca_fusion( ...
            sg_amp, snr_amp, b_bp, a_bp, b_hp, a_hp);

        [phase_fused, phase_sg_fused, exp_phase] = run_mrc_pca_fusion( ...
            sg_phase, snr_phase, b_bp, a_bp, b_hp, a_hp);

        % 同時建立「振幅 + 相位」180 通道聯合 MRC-PCA 融合訊號 (消除單一模態盲區)
        joint_sg_channels = [sg_amp, sg_phase];
        joint_snr         = [snr_amp, snr_phase];
        [joint_fused, joint_sg_fused, ~] = run_mrc_pca_fusion( ...
            joint_sg_channels, joint_snr, b_bp, a_bp, b_hp, a_hp);

        amp_fused_by_tx(:, tx)   = amp_fused;
        phase_fused_by_tx(:, tx) = phase_fused;
        joint_fused_by_tx(:, tx) = joint_fused;
        amp_sg_by_tx(:, tx)      = amp_sg_fused;
        phase_sg_by_tx(:, tx)    = phase_sg_fused;
        joint_sg_by_tx(:, tx)    = joint_sg_fused;

        amp_explained(tx)   = exp_amp;
        phase_explained(tx) = exp_phase;

        % -----------------------------------------------------------------
        % Step 5: 評估各 Tx 之呼吸訊號品質 (解釋方差 + 呼吸頻帶能量集中度)
        % -----------------------------------------------------------------
        snr_fused_amp   = estimate_subcarrier_snr(amp_fused, Fs_target, resp_band);
        snr_fused_phase = estimate_subcarrier_snr(phase_fused, Fs_target, resp_band);
        focus_ratio = 0.5 * (snr_fused_amp / (1 + snr_fused_amp) + ...
                             snr_fused_phase / (1 + snr_fused_phase));
        band_focus(tx) = focus_ratio;

        explained_mean = 0.5 * (amp_explained(tx) + phase_explained(tx));
        selection_score(tx) = 0.5 * explained_mean + 0.5 * band_focus(tx);

        fprintf(['[Tx%d] MRC-PCA 解釋方差: amp=%.2f%% phase=%.2f%% | ', ...
                 'band-focus=%.3f | score=%.3f\n'], ...
                tx, amp_explained(tx) * 100, phase_explained(tx) * 100, ...
                band_focus(tx), selection_score(tx));
    end

    % 選擇呼吸特徵最顯著之 Tx
    [~, best_tx] = max(selection_score);
    amp_pc1_norm   = amp_fused_by_tx(:, best_tx);
    phase_pc1_norm = phase_fused_by_tx(:, best_tx);
    csi_ratio_out  = csi_ratio_by_tx{best_tx};

    % 封裝詳細輸出結構體供下游強化特徵提取使用
    mrc_pca_details = struct();
    mrc_pca_details.best_tx           = best_tx;
    mrc_pca_details.joint_mrc_pca     = joint_fused_by_tx(:, best_tx);
    mrc_pca_details.amp_sg_waveform   = amp_sg_by_tx(:, best_tx);
    mrc_pca_details.phase_sg_waveform = phase_sg_by_tx(:, best_tx);
    mrc_pca_details.joint_sg_waveform = joint_sg_by_tx(:, best_tx);
    mrc_pca_details.selection_score   = selection_score;

    fprintf('[結果] 最終選擇 Tx%d 作為輸出 (CSI Ratio + MRC-PCA 方向對齊加權合成)\n', best_tx);

    % =========================================================================
    % 視覺化：僅在互動/單檔模式 (DefaultFigureVisible == 'on') 時繪圖
    % =========================================================================
    if strcmpi(get(0, 'DefaultFigureVisible'), 'on')
        time_axis = (0:N-1) / Fs_target;

        figure('Name', 'CSI Ratio -> MRC-PCA Fusion & Tx Selection', 'NumberTitle', 'off', ...
               'Position', [100, 100, 1100, 600]);

        subplot(2, 2, 1);
        bar(avg_weights.');
        set(gca, 'XTick', 1:Ntx, 'XTickLabel', {'Tx1', 'Tx2'});
        legend(pair_names, 'Location', 'best');
        ylabel('Mean MRC Weight Share');
        title('Per-Tx 平均 CSI Ratio MRC 權重分佈');
        grid on;

        subplot(2, 2, 2);
        plot(time_axis, amp_fused_by_tx(:, 1), 'Color', [0 0.447 0.741], 'LineWidth', 1.1); hold on;
        plot(time_axis, amp_fused_by_tx(:, 2), 'Color', [0.85 0.325 0.098], 'LineWidth', 1.1);
        plot(time_axis, amp_pc1_norm, 'k', 'LineWidth', 1.8);
        title(sprintf('MRC-PCA Amplitude Waveform (Selected: Tx%d)', best_tx));
        xlabel('時間 (秒)'); ylabel('Z-score'); grid on; axis tight;
        legend({'Tx1', 'Tx2', 'Selected'}, 'Location', 'best');

        subplot(2, 2, 3);
        plot(time_axis, phase_fused_by_tx(:, 1), 'Color', [0 0.447 0.741], 'LineWidth', 1.1); hold on;
        plot(time_axis, phase_fused_by_tx(:, 2), 'Color', [0.85 0.325 0.098], 'LineWidth', 1.1);
        plot(time_axis, phase_pc1_norm, 'k', 'LineWidth', 1.8);
        title(sprintf('MRC-PCA Phase Waveform (Selected: Tx%d)', best_tx));
        xlabel('時間 (秒)'); ylabel('Z-score'); grid on; axis tight;
        legend({'Tx1', 'Tx2', 'Selected'}, 'Location', 'best');

        subplot(2, 2, 4);
        metric_matrix = [amp_explained, phase_explained, band_focus, selection_score];
        bar(metric_matrix);
        set(gca, 'XTick', 1:Ntx, 'XTickLabel', {'Tx1', 'Tx2'});
        legend({'Amp PC1 explained', 'Phase PC1 explained', 'Band focus', 'Selection score'}, ...
               'Location', 'best');
        ylabel('Score');
        title('Tx Selection Criterion (MRC-PCA)');
        grid on;
    end

    fprintf('[完成] 執行 process_csi_signal (CSI Ratio + MRC-PCA) 完成，輸出長度 %d\n', N);
end

% =========================================================================
% 內部輔助函式：評估天線對穩定度與論文 Section IV-B 之 MRC-PCA 融合
% =========================================================================
function health_score = evaluate_pair_health(pair_ratio, Fs, resp_band)
% 評估單一天線對 (30 子載波) 是否受天線接頭鬆動 (如 Rx2 鬆動) 影響：
% 鬆動天線會產生大量 0.05~0.16 Hz (週期 6~20 秒) 之慢速阻抗漂移與高頻接觸跳階
    mag = abs(pair_ratio); % [N, 30]
    [N, K] = size(mag);
    if N < 16
        health_score = 1.0;
        return;
    end

    % 1. 檢查一階差分跳階異常度
    med_mag = max(median(mag, 1), eps);
    norm_mag = bsxfun(@rdivide, mag, med_mag);
    d_mag = abs(diff(norm_mag, 1, 1));
    glitch_level = median(prctile(d_mag, 98, 1));

    % 2. 檢查生理呼吸頻帶 (0.19~0.48 Hz) 與超低頻慢速漂移 (0.05~0.16 Hz) 之能量比
    norm_mag_0 = bsxfun(@minus, norm_mag, mean(norm_mag, 1));
    Y = fft(norm_mag_0);
    P2 = abs(Y / N).^2;
    half_len = floor(N / 2) + 1;
    P1 = P2(1:half_len, :);
    f = Fs * (0:(half_len - 1))' / N;

    resp_idx  = (f >= resp_band(1)) & (f <= resp_band(2));
    drift_idx = (f >= 0.05) & (f < resp_band(1));

    resp_p  = median(sum(P1(resp_idx, :), 1));
    drift_p = median(sum(P1(drift_idx, :), 1));

    drift_ratio = resp_p / max(resp_p + 0.8 * drift_p, eps);
    stability   = 1.0 / (1.0 + 4.0 * glitch_level);

    health_score = max(0.20, drift_ratio * stability);
end

function [fused_bp_norm, fused_sg_norm, pc1_explained] = run_mrc_pca_fusion(sg_channels, snr_vals, b_bp, a_bp, b_hp, a_hp)
    [N, M] = size(sg_channels);

    % 僅保留高於第 55 百分位數 SNR 之高品質子載波參與 MRC 合成 (排除鬆動天線或盲區子載波)
    snr_thresh = prctile(snr_vals, 55);
    active_mask = snr_vals >= snr_thresh;
    if sum(active_mask) < 5
        active_mask = true(1, M);
    end

    mrc_gains = zeros(1, M);
    mrc_gains(active_mask) = snr_vals(active_mask);
    mrc_gains = mrc_gains / max(sum(mrc_gains), eps);

    % 1. 將各子載波乘以 MRC 增益並進行呼吸頻帶帶通濾波 [0.18, 0.52] Hz
    weighted_bp = zeros(N, M);
    bp_channels = zeros(N, M);
    for k = 1:M
        if N > 12
            bp_sig = filtfilt(b_bp, a_bp, sg_channels(:, k));
        else
            bp_sig = sg_channels(:, k);
        end
        bp_channels(:, k) = bp_sig;
        weighted_bp(:, k) = bp_sig * mrc_gains(k);
    end

    % 2. 排除高能量體動瞬間突波後執行 PCA，確保 PC1 投影方向對齊於安靜呼吸
    inst_energy = sum(weighted_bp.^2, 2);
    med_e = median(inst_energy);
    mad_e = 1.4826 * median(abs(inst_energy - med_e));
    quiet_rows = inst_energy <= (med_e + 3.5 * max(mad_e, eps));
    if sum(quiet_rows) < max(10, round(0.3 * N))
        quiet_rows = true(N, 1);
    end

    [coeff, ~, latent] = pca(weighted_bp(quiet_rows, :));
    if isempty(coeff)
        direction_signs = ones(1, M);
        pc1_explained = 0;
    else
        direction_signs = sign(coeff(:, 1))'; % [1, M] 方向修正因子 (+1 或 -1)
        direction_signs(direction_signs == 0) = 1;
        total_var = sum(latent);
        if total_var > eps
            pc1_explained = latent(1) / total_var;
        else
            pc1_explained = 0;
        end
    end

    % 3. 將 PCA 決定的方向正負號加回 MRC 增益，進行加權平均合成 (最大化 SNR)
    final_weights = (direction_signs .* mrc_gains(:)'); % [1, M]
    fused_bp = bp_channels * final_weights(:);
    fused_sg = sg_channels * final_weights(:);

    % 對 SG 融合波形進行 0.15 Hz 高通濾波以消除 6~10 秒慢速基線漂移
    if N > 12
        fused_sg = filtfilt(b_hp, a_hp, fused_sg);
    end

    % 4. 以安靜呼吸標準差進行穩健歸一化
    fused_bp_norm = robust_zscore(fused_bp, quiet_rows);
    fused_sg_norm = robust_zscore(fused_sg, quiet_rows);
end

function snr_val = estimate_subcarrier_snr(signal_vec, Fs_target, resp_band)
% 論文 Section IV-B + 慢速漂移抑制：
% 計算正常睡眠呼吸頻率 (0.18~0.52 Hz) 能量與「超低頻天線漂移 (0.05~0.18 Hz) + 高頻雜訊 (>0.52 Hz)」之比值
    signal_vec = signal_vec(:);
    n = length(signal_vec);
    if n < 8
        snr_val = eps;
        return;
    end

    signal_vec = signal_vec - mean(signal_vec);
    if all(abs(signal_vec) < eps)
        snr_val = eps;
        return;
    end

    Y = fft(signal_vec);
    P2 = abs(Y / n).^2;
    half_len = floor(n / 2) + 1;
    P1 = P2(1:half_len);
    if half_len > 2
        P1(2:end-1) = 2 * P1(2:end-1);
    end

    f = Fs_target * (0:(half_len - 1)) / n;
    sig_idx   = (f >= resp_band(1)) & (f <= resp_band(2));
    drift_idx = (f >= 0.05) & (f < resp_band(1));
    noise_idx = (f > resp_band(2)) & (f <= min(5.0, Fs_target / 2));

    signal_power = sum(P1(sig_idx));
    % 將 < 0.18 Hz 的慢速天線漂移一併計入雜訊分母，防止 5.5~8 秒的慢速漂移通道奪取高權重
    noise_power  = sum(P1(noise_idx)) + 0.65 * sum(P1(drift_idx));
    snr_val = max(signal_power / max(noise_power, eps), eps);
end

function y = clean_and_zscore_channel(x, Fs)
% 消除網卡 AGC 局部瞬間階躍、移除超過 5 秒之慢速基線漂移，並抑制超過 3.8 倍 MAD 之瞬態尖峰
    x = double(x(:));

    % 1. 僅消除「局部孤立」之方波台階跳變 (比較單步差分與 0.6 秒局部移動中位數差分，保留連續波動)
    if length(x) > 10
        dx = diff(x);
        abs_dx = abs(dx);
        if exist('movmedian', 'builtin') || exist('movmedian', 'file')
            local_med_dx = movmedian(abs_dx, max(5, round(0.6 * Fs)));
        else
            local_med_dx = median(abs_dx) * ones(size(abs_dx));
        end
        global_mad_dx = max(1.4826 * median(abs_dx), eps);
        step_mask = abs_dx > max(6.0 * local_med_dx, 5.0 * global_mad_dx);
        dx(step_mask) = 0;
        x = [x(1); x(1) + cumsum(dx)];
    end

    % 2. 使用 5.5 秒滑動中位數視窗扣除天線鬆動造成的慢速基線漂移 (> 5.5 秒週期)
    win_med = max(5, round(5.5 * Fs));
    if exist('movmedian', 'builtin') || exist('movmedian', 'file')
        baseline = movmedian(x, win_med);
    else
        baseline = detrend(x, 'linear') - x;
        baseline = -baseline;
    end
    x_det = x - baseline;

    med_x = median(x_det);
    mad_x = 1.4826 * median(abs(x_det - med_x));
    if mad_x > eps
        clip_lim = 3.8 * mad_x;
        x_det = max(med_x - clip_lim, min(med_x + clip_lim, x_det));
    end
    y = safe_zscore(x_det);
end

function y = robust_zscore(x, quiet_rows)
    x = x(:);
    x_q = x(quiet_rows);
    if isempty(x_q)
        x_q = x;
    end
    mu = median(x_q);
    sigma = 1.4826 * median(abs(x_q - mu));
    if sigma < eps
        sigma = std(x);
    end
    if sigma < eps
        y = zeros(size(x));
    else
        y = (x - mu) / sigma;
    end
end

function y = safe_zscore(x)
    x = x(:);
    sigma = std(x);
    if isempty(x) || sigma < eps
        y = zeros(size(x));
    else
        y = (x - mean(x)) / sigma;
    end
end

function window_length = sg_window_length(n_samples, Fs_target, poly_order)
% 動態計算 Savitzky-Golay 奇數視窗大小 (對應 1.5 秒物理時間視窗，在 40Hz 下為 61 點，完整保留 3~4 秒正常呼吸週期)
    base_window = 2 * floor((1.5 * Fs_target) / 2) + 1;
    max_valid = n_samples;
    if mod(max_valid, 2) == 0
        max_valid = max_valid - 1;
    end

    window_length = min(base_window, max_valid);
    if window_length <= poly_order
        window_length = 0;
    end
end
