function [amp_pc1_norm, phase_pc1_norm] = process_csi_signal(csi_matrix, Fs_target)
% process_csi_signal   Rx pair conjugate multiplication + SNR weighting + per-Tx PCA
%
% Pipeline for MATLAB R2015b:
%   [N,30,2,3] CSI
%       |
%       +-- Step 1: Rx pair conjugate multiplication per Tx
%              Rx1*conj(Rx2), Rx1*conj(Rx3), Rx2*conj(Rx3)
%              (complex output preserved)
%       |
%       +-- Step 2: Per-subcarrier SNR estimation
%              SNR used only as weights; no antenna combining is performed
%       |
%       +-- Step 3: Savitzky-Golay filtering on amplitude and phase
%       |
%       +-- Step 4: PCA on each Tx independently
%              keep PC1 for amplitude and phase
%       |
%       +-- Step 5: Z-score normalization and Tx selection
%              select the Tx with the stronger breathing signature
%
% Inputs:
%   csi_matrix  - CSI tensor [N, 30, 2, 3]
%   Fs_target   - resampled sampling rate (Hz)
%
% Outputs:
%   amp_pc1_norm   - selected Tx amplitude PC1, z-scored [N, 1]
%   phase_pc1_norm - selected Tx phase PC1, z-scored [N, 1]

    if nargin < 2
        Fs_target = 40;
    end

    if isempty(csi_matrix)
        error('輸入的 CSI 矩陣為空！');
    end

    [N, Nsc, Ntx, Nrx] = size(csi_matrix);
    if Ntx ~= 2 || Nrx ~= 3 || Nsc ~= 30
        error('輸入 CSI 維度必須為 [N, 30, 2, 3]');
    end

    if N < 4
        error('CSI 資料長度不足，無法進行 PCA');
    end

    resp_band = [0.1, 0.6];
    pair_defs = [1 2; 1 3; 2 3];
    pair_names = {'Rx1*conj(Rx2)', 'Rx1*conj(Rx3)', 'Rx2*conj(Rx3)'};
    num_pairs = size(pair_defs, 1);

    amp_pc1_by_tx = zeros(N, Ntx);
    phase_pc1_by_tx = zeros(N, Ntx);
    amp_explained = zeros(Ntx, 1);
    phase_explained = zeros(Ntx, 1);
    band_focus = zeros(Ntx, 1);
    selection_score = zeros(Ntx, 1);
    avg_weights = zeros(num_pairs, Ntx);

    fprintf('====== [process_csi_signal] 開始：Rx pair 特徵 + SNR 權重 + Tx PCA ======\n');

    for tx = 1:Ntx
        fprintf('[Tx%d] 建立 Rx pair 的共軛特徵...\n', tx);

        pair_complex = zeros(N, Nsc, num_pairs);
        for p = 1:num_pairs
            rx_a = pair_defs(p, 1);
            rx_b = pair_defs(p, 2);
            rx_a_sig = squeeze(csi_matrix(:, :, tx, rx_a));
            rx_b_sig = squeeze(csi_matrix(:, :, tx, rx_b));
            pair_complex(:, :, p) = rx_a_sig .* conj(rx_b_sig);
        end

        fprintf('[Tx%d] 估計每個 subcarrier 的 SNR，並以其作為權重 (不進行天線合併)...\n', tx);
        pair_weights = zeros(num_pairs, Nsc);
        weighted_complex = zeros(N, Nsc * num_pairs);

        for sc = 1:Nsc
            snr_vals = zeros(num_pairs, 1);
            for p = 1:num_pairs
                snr_vals(p) = estimate_respiration_snr(pair_complex(:, sc, p), Fs_target, resp_band);
            end

            snr_sum = sum(snr_vals);

            if snr_sum < eps
                w = ones(num_pairs,1)/num_pairs;
                warning('[Tx%d] 子載波 #%d 的 SNR 估計為 0，改用平均權重', tx, sc);
            else
                w = snr_vals / snr_sum;
            end

            pair_weights(:, sc) = w;

            for p = 1:num_pairs
                col_idx = (p-1)*Nsc + sc;
                weighted_complex(:, col_idx) = pair_complex(:, sc, p) * w(p);
            end
        end

        for p = 1:num_pairs
            avg_weights(p, tx) = mean(pair_weights(p, :));
            fprintf('[Tx%d] 平均權重 %s = %.3f\n', tx, pair_names{p}, avg_weights(p, tx));
        end

        amp_matrix = abs(weighted_complex);
        phase_matrix = unwrap(angle(weighted_complex));

        %window_length = sg_window_length(N, Fs_target, 3);
        window_length = 121;
        if window_length > 3
            fprintf('[Tx%d] Savitzky-Golay 濾波 (window=%d, poly=3)...\n', tx, window_length);
            amp_filtered = sgolayfilt(amp_matrix, 3, window_length);
            phase_filtered = sgolayfilt(phase_matrix, 3, window_length);
        else
            amp_filtered = amp_matrix;
            phase_filtered = phase_matrix;
            warning('[Tx%d] 資料長度不足以設定視窗大小，跳過 Savitzky-Golay 濾波', tx);
        end

        fprintf('[Tx%d] 進行 PCA，提取振幅與相位的 PC1...\n', tx);
        [~, score_amp, latent_amp] = pca(amp_filtered);
        [~, score_phase, latent_phase] = pca(phase_filtered);

        amp_pc1_raw = score_amp(:, 1);
        phase_pc1_raw = score_phase(:, 1);
        amp_pc1_by_tx(:, tx) = safe_zscore(amp_pc1_raw);
        phase_pc1_by_tx(:, tx) = safe_zscore(phase_pc1_raw);

        total_var_amp = sum(latent_amp);
        total_var_phase = sum(latent_phase);
        if total_var_amp < eps
            amp_explained(tx) = 0;
        else
            amp_explained(tx) = latent_amp(1) / total_var_amp;
        end
        if total_var_phase < eps
            phase_explained(tx) = 0;
        else
            phase_explained(tx) = latent_phase(1) / total_var_phase;
        end

        band_focus(tx) = 0.5 * ( ...
            band_energy_ratio(amp_pc1_by_tx(:, tx), Fs_target, resp_band) + ...
            band_energy_ratio(phase_pc1_by_tx(:, tx), Fs_target, resp_band));

        % Tx selection metric:
        %   1) mean of amplitude/phase PC1 explained variance
        %   2) mean respiration-band energy concentration after z-score
        % The two terms are blended into one objective score.
        explained_mean = 0.5 * (amp_explained(tx) + phase_explained(tx));
        selection_score(tx) = 0.5 * explained_mean + 0.5 * band_focus(tx);

        fprintf(['[Tx%d] PC1 解釋方差: amp=%.2f%% phase=%.2f%% | ', ...
                 'band-focus=%.3f | score=%.3f\n'], ...
                tx, amp_explained(tx) * 100, phase_explained(tx) * 100, ...
                band_focus(tx), selection_score(tx));
    end

    [~, best_tx] = max(selection_score);
    amp_pc1_norm = amp_pc1_by_tx(:, best_tx);
    phase_pc1_norm = phase_pc1_by_tx(:, best_tx);

    fprintf('[結果] 最終選擇 Tx%d 作為輸出，為該 Tx 的振幅與相位 PC1（已分別 Z-score）\n', best_tx);
    fprintf('[結果] 選擇依據：PC1 解釋方差 + 呼吸頻帶能量集中度\n');

    % =========================================================================
    % 視覺化：SNR 權重分佈 + Tx PC1 對照 + Tx 選擇依據
    % =========================================================================
    time_axis = (0:N-1) / Fs_target;

    figure('Name', 'CSI Rx Pair -> Tx PCA Selection', 'NumberTitle', 'off', ...
           'Position', [100, 100, 1100, 600]);

    subplot(2, 2, 1);
    bar(avg_weights.');
    set(gca, 'XTick', 1:Ntx, 'XTickLabel', {'Tx1', 'Tx2'});
    legend(pair_names, 'Location', 'best');
    ylabel('Mean SNR Weight');
    title('Per-Tx 平均 Rx pair 權重');
    grid on;

    subplot(2, 2, 2);
    plot(time_axis, amp_pc1_by_tx(:, 1), 'Color', [0 0.447 0.741], 'LineWidth', 1.1); hold on;
    plot(time_axis, amp_pc1_by_tx(:, 2), 'Color', [0.85 0.325 0.098], 'LineWidth', 1.1);
    plot(time_axis, amp_pc1_norm, 'k', 'LineWidth', 1.8);
    title(sprintf('Amplitude PC1 (selected: Tx%d)', best_tx));
    xlabel('時間 (秒)'); ylabel('Z-score'); grid on; axis tight;
    legend('Tx1', 'Tx2', 'Selected', 'Location', 'best');

    subplot(2, 2, 3);
    plot(time_axis, phase_pc1_by_tx(:, 1), 'Color', [0 0.447 0.741], 'LineWidth', 1.1); hold on;
    plot(time_axis, phase_pc1_by_tx(:, 2), 'Color', [0.85 0.325 0.098], 'LineWidth', 1.1);
    plot(time_axis, phase_pc1_norm, 'k', 'LineWidth', 1.8);
    title(sprintf('Phase PC1 (selected: Tx%d)', best_tx));
    xlabel('時間 (秒)'); ylabel('Z-score'); grid on; axis tight;
    legend('Tx1', 'Tx2', 'Selected', 'Location', 'best');

    subplot(2, 2, 4);
    metric_matrix = [amp_explained, phase_explained, band_focus, selection_score];
    bar(metric_matrix);
    set(gca, 'XTick', 1:Ntx, 'XTickLabel', {'Tx1', 'Tx2'});
    legend({'Amp PC1 explained', 'Phase PC1 explained', 'Band focus', 'Selection score'}, ...
           'Location', 'best');
    ylabel('Score');
    title('Tx Selection Criterion');
    grid on;

    fprintf('[完成] 執行 process_csi_signal 流程完成，輸出長度 %d\n', N);
end

function snr_linear = estimate_respiration_snr(feature_series, Fs_target, resp_band)
% Estimate respiration-band SNR for one complex feature/subcarrier.

    amp_series = abs(feature_series(:));
    phase_series = unwrap(angle(feature_series(:)));

    amp_snr = band_energy_ratio(amp_series, Fs_target, resp_band);
    phase_snr = band_energy_ratio(phase_series, Fs_target, resp_band);

    snr_linear = max(0.5 * (amp_snr + phase_snr), eps);
end

function ratio = band_energy_ratio(signal_vec, Fs_target, resp_band)
% Respiration-band energy ratio using a one-sided FFT power spectrum.

    signal_vec = signal_vec(:);
    n = length(signal_vec);
    if n < 4
        ratio = 0;
        return;
    end

    signal_vec = signal_vec - mean(signal_vec);
    if all(abs(signal_vec) < eps)
        ratio = 0;
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
    band_idx = (f >= resp_band(1)) & (f <= resp_band(2));

    signal_power = sum(P1(band_idx));
    noise_power = sum(P1(~band_idx));
    ratio = signal_power / max(noise_power, eps);
end

function y = safe_zscore(x)
% Robust z-score that returns zeros for a near-constant input.

    x = x(:);
    sigma = std(x);
    if isempty(x) || sigma < eps
        y = zeros(size(x));
    else
        y = (x - mean(x)) / sigma;
    end
end

function window_length = sg_window_length(n_samples, Fs_target, poly_order)
% Choose an odd Savitzky-Golay window that is valid for the current input.

    base_window = 2 * floor((0.5 * Fs_target) / 2) + 1;
    max_valid = n_samples;
    if mod(max_valid, 2) == 0
        max_valid = max_valid - 1;
    end

    window_length = min(base_window, max_valid);
    if window_length <= poly_order
        window_length = 0;
    end
end