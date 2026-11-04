function [best_stream_name, best_signal, best_fpsd, best_sg_signal] = select_respiration_stream(amp_pcs_norm, phase_pcs_norm, Fs_target, motion_mask, mrc_pca_details)
    % select_respiration_stream  基於《WiFi-Sleep》振幅與相位互補之 MRC-PCA 呼吸串流選擇模組
    %
    % 輸入:
    %   amp_pcs_norm    - MRC-PCA 振幅呼吸訊號 [N x 1]
    %   phase_pcs_norm  - MRC-PCA 相位呼吸訊號 [N x 1]
    %   Fs_target       - 目標採樣率 (預設 40 Hz)
    %   motion_mask     - (可選) 體動遮罩 [N x 1]，計算頻譜評分時排除體動干擾區段
    %   mrc_pca_details - (可選) process_csi_signal 輸出的細節結構體 (含聯合融合與 SG 波形)
    %
    % 輸出:
    %   best_stream_name - 選定的最佳串流名稱 ('Amp-PC1', 'Phase-PC1', 或 'Combined-MRC-PCA')
    %   best_signal      - 最佳帶通呼吸訊號 [N x 1]
    %   best_fpsd        - 頻譜主峰頻率 (Hz)
    %   best_sg_signal   - 保留吸吐氣不對稱波形特徵的 SG 平滑訊號 [N x 1] (供 FIT / I-E Ratio 使用)

    if nargin < 3 || isempty(Fs_target), Fs_target = 40; end
    N = size(amp_pcs_norm, 1);

    if nargin < 4 || isempty(motion_mask)
        motion_mask = false(N, 1);
    else
        motion_mask = logical(motion_mask(:));
        if length(motion_mask) ~= N
            motion_mask = false(N, 1);
        end
    end

    % 1. 準備候選串流：若提供聯合 MRC-PCA 串流則一併納入評估以消除盲區
    if nargin >= 5 && isstruct(mrc_pca_details) && isfield(mrc_pca_details, 'joint_mrc_pca')
        all_streams = [amp_pcs_norm(:, 1), phase_pcs_norm(:, 1), mrc_pca_details.joint_mrc_pca(:, 1)];
        sg_streams  = [mrc_pca_details.amp_sg_waveform(:, 1), ...
                       mrc_pca_details.phase_sg_waveform(:, 1), ...
                       mrc_pca_details.joint_sg_waveform(:, 1)];
        labels = {'Amp-PC1', 'Phase-PC1', 'Combined-MRC-PCA'};
    else
        all_streams = [amp_pcs_norm(:, 1), phase_pcs_norm(:, 1)];
        sg_streams  = all_streams;
        labels = {'Amp-PC1', 'Phase-PC1'};
    end
    num_streams = size(all_streams, 2);

    % 帶通濾波器設定：0.19 Hz ~ 0.55 Hz (對應 11.4 ~ 33 bpm，週期 1.82 ~ 5.26 秒)
    % 濾除因 Rx2 天線鬆動或掉包插值產生的 0.10 ~ 0.18 Hz (週期 5.5 ~ 10 秒) 慢速基線漂移
    [b, a] = butter(3, [0.19, 0.55] / (Fs_target / 2), 'bandpass');

    % 正常人類睡眠呼吸頻率判定範圍 (11.5 bpm - 33 bpm)
    freq_min = 11.5 / 60; % 0.192 Hz (週期 <= 5.2 秒)
    freq_max = 33.0 / 60; % 0.550 Hz (週期 >= 1.82 秒)

    scores = -ones(1, num_streams);
    peak_freqs = zeros(1, num_streams);
    filtered_signals = zeros(N, num_streams);

    % 排除體動區段後的安靜樣本索引 (若安靜樣本超過 5 秒則只用安靜期評估 PSD)
    quiet_idx = ~motion_mask;
    if sum(quiet_idx) < round(5 * Fs_target)
        quiet_idx = true(N, 1);
    end
    N_eval = sum(quiet_idx);
    f = Fs_target * (0:floor(N_eval/2)) / N_eval;
    band_idx = (f >= freq_min) & (f <= freq_max);

    % 2. 遍歷候選串流進行呼吸顯著度與 SNR 評估
    for i = 1:num_streams
        if N > 12
            sig_filt = filtfilt(b, a, all_streams(:, i));
        else
            sig_filt = all_streams(:, i);
        end
        filtered_signals(:, i) = sig_filt;

        % 取安靜區段並加漢明窗 (Hamming Taper) 計算功率譜密度 (PSD)，抑制掉包邊界頻譜洩漏
        sig_eval = sig_filt(quiet_idx);
        sig_eval = sig_eval - mean(sig_eval);
        n_c = (0:(N_eval-1))';
        win_hamming = 0.54 - 0.46 * cos(2 * pi * n_c / max(N_eval - 1, 1));
        sig_eval = sig_eval .* win_hamming;

        Y = fft(sig_eval);
        P2 = abs(Y / N_eval);
        P1 = P2(1:floor(N_eval/2)+1);
        if length(P1) > 2
            P1(2:end-1) = 2 * P1(2:end-1);
        end
        psd_data = P1.^2;

        % 正規化 PSD 使不同串流可在相同基準比較
        psd_norm = psd_data / max(sum(psd_data), eps);

        psd_in_band = psd_norm;
        psd_in_band(~band_idx) = 0;
        [~, max_idx] = max(psd_in_band);
        peak_f = f(max_idx);
        peak_freqs(i) = peak_f;

        if peak_f >= freq_min && peak_f <= freq_max
            v = var(psd_norm(band_idx));
            [~, locs, ~, prom] = findpeaks(psd_norm);
            peak_match_idx = find(locs == max_idx, 1);

            if ~isempty(peak_match_idx)
                p_val = prom(peak_match_idx);
            else
                p_val = psd_norm(max_idx);
            end

            % 結合呼吸頻帶 SNR 作為綜合評分
            in_band_power  = sum(psd_norm(band_idx));
            out_band_power = sum(psd_norm(~band_idx));
            snr_ratio      = in_band_power / max(out_band_power, eps);

            scores(i) = v * p_val * snr_ratio;
        end
    end

    % 3. 最佳化選擇
    [best_score, best_idx] = max(scores);

    if best_score > 0
        best_stream_name = labels{best_idx};
        best_signal      = filtered_signals(:, best_idx);
        best_sg_signal   = sg_streams(:, best_idx);
        best_fpsd        = peak_freqs(best_idx);
    else
        best_stream_name = 'None (No Valid Respiration Found)';
        best_signal      = zeros(N, 1);
        best_sg_signal   = zeros(N, 1);
        best_fpsd        = 0;
        best_idx         = 0;
    end

    % 4. 視覺化 (僅在圖表可見時繪製)
    if strcmpi(get(0, 'DefaultFigureVisible'), 'on')
        time_axis = (0:N-1) / Fs_target;
        figure('Name', 'MRC-PCA 多路串流自動選擇系統', 'Position', [100, 100, 1000, 600]);

        subplot(2,1,1);
        plot(time_axis, filtered_signals, 'Color', [0.8 0.8 0.8], 'LineWidth', 0.5);
        hold on;

        if best_idx > 0
            plot(time_axis, best_signal, 'r', 'LineWidth', 1.5);
            title(['所有 MRC-PCA 串流對比 (最終選定: ', best_stream_name, ')'], 'FontSize', 12);
            h = legend([labels, {'Selected'}], 'Location', 'southoutside', 'Orientation', 'horizontal');
        else
            title(['所有 MRC-PCA 串流對比 (狀態: ', best_stream_name, ')'], 'FontSize', 12, 'Color', 'r');
            h = legend(labels, 'Location', 'southoutside', 'Orientation', 'horizontal');
        end
        xlabel('時間 (秒)'); ylabel('幅值');
        set(h, 'FontSize', 8);
        grid on;

        subplot(2,1,2);
        if best_idx > 0
            sig_eval = best_signal(quiet_idx) - mean(best_signal(quiet_idx));
            best_Y = fft(sig_eval);
            best_P2 = abs(best_Y / N_eval);
            best_P1 = best_P2(1:floor(N_eval/2)+1);
            if length(best_P1) > 2
                best_P1(2:end-1) = 2 * best_P1(2:end-1);
            end
            plot(f, best_P1.^2, 'LineWidth', 1.5);
            xlim([0 1.5]);
            yl = ylim;
            line([freq_min freq_min], yl, 'Color', 'g', 'LineStyle', '--');
            line([freq_max freq_max], yl, 'Color', 'g', 'LineStyle', '--');
            text(freq_min, yl(2)*0.9, ' 呼吸下限 (11.5 bpm)', 'Color', 'g');
            text(freq_max, yl(2)*0.9, ' 呼吸上限 (33 bpm)', 'Color', 'g');
            title(['最佳串流 PSD 分析 - 估計呼吸率: ', num2str(best_fpsd*60, '%.2f'), ' BPM']);
            xlabel('頻率 (Hz)'); ylabel('功率譜密度 (PSD)');
            grid on;
        else
            text(0.4, 0.5, '無有效呼吸頻譜可供顯示', 'FontSize', 14, 'Color', 'r');
        end
    end

    fprintf('自動選擇完成：最終選定 [%s]\n', best_stream_name);
end
