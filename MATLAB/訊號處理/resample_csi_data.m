function [csi_resampled, t_uniform, gap_mask] = resample_csi_data(csi_matrix, timestamp_sec, Fs_target, Fs_orig)
    % resample_csi_data  相位同步抗混疊低通濾波與均勻重採樣模組
    %
    % 關鍵訊號處理修正:
    %   1. Intel 5300 原始 CSI 每個封包皆帶有隨機硬體相位偏移 exp(-j*phi(t))。
    %      本模組先以每包 Rx1 相位作為同網卡共同相位基準進行對齊 (保留天線間相對相位差與各天線真實振幅)，
    %      並將「振幅」與「相對相位單位相量」分離處理，避免隨機相位在低通濾波時互相抵消。
    %   2. 先於時間軸進行線性對齊再施以低通抗混疊濾波，避免在長段掉包斷層 (如 34~48 秒) 兩端把
    %      間隔數秒的封包當成相鄰 5ms 樣本濾波而引發邊界振鈴 (Filter Ringing)。
    %   3. 區分「微秒/毫秒級封包抖動 (< 0.35 秒)」與「真正訊號中斷掉包 (>= 0.35 秒)」：
    %      避免 10~50ms 的正常 Wi-Fi 排程抖動將完好的呼吸波峰誤標為掉包無效點。
    %
    % 輸入:
    %   csi_matrix    - 原始 4D CSI 矩陣 [N_orig, 30, 2, 3]
    %   timestamp_sec - 原始相對時間向量 (秒) [N_orig, 1]
    %   Fs_target     - 目標重採樣率 (預設 40 Hz)
    %   Fs_orig       - 原始採樣率 (預設 200 Hz)
    %
    % 輸出:
    %   csi_resampled - 重採樣與插值後的 4D CSI 矩陣 [N_uniform, 30, 2, 3]
    %   t_uniform     - 均勻的時間軸向量 [N_uniform, 1]
    %   gap_mask      - 有效訊號斷層遮罩 [N_uniform, 1] (true 代表連續 >= 0.35 秒無封包之真實斷訊區間)

    % 1. 輸入參數防呆與預設值設定
    if nargin < 4, Fs_orig = 200; end
    if nargin < 3, Fs_target = 40; end

    [N_orig, Nsc, Ntx, Nrx] = size(csi_matrix);
    if N_orig < 4 || isempty(timestamp_sec)
        error('輸入的 CSI 矩陣為空或封包數過少！無法進行重採樣。');
    end

    fprintf('====== [重採樣模組] 啟動 ======\n');
    fprintf('原始設定採樣率: %.2f Hz -> 目標均勻重採樣率: %d Hz\n', Fs_orig, Fs_target);

    % 確保時間軸嚴格遞增 (剔除時間戳完全重複的異常封包，防止 interp1 報錯)
    timestamp_sec = double(timestamp_sec(:));
    [timestamp_sec, unique_idx] = unique(timestamp_sec, 'stable');
    if length(unique_idx) < N_orig
        csi_matrix = csi_matrix(unique_idx, :, :, :);
        N_orig = length(unique_idx);
    end

    % 2. 消除每包隨機共同硬體相位偏移 (以 Rx1 為相位參考，保留精確天線間相對相位與各天線真實振幅)
    csi_sync = zeros(size(csi_matrix));
    for tx = 1:Ntx
        ref_rx = squeeze(csi_matrix(:, :, tx, 1)); % [N_orig, 30]
        ref_phasor = exp(-1i * angle(ref_rx));     % 單位模長之相位共軛基準
        for rx = 1:Nrx
            csi_sync(:, :, tx, rx) = squeeze(csi_matrix(:, :, tx, rx)) .* ref_phasor;
        end
    end

    % 分離振幅與對齊後之單位相角向量
    amp_2d    = reshape(abs(csi_sync), N_orig, []);
    phasor_2d = reshape(exp(1i * angle(csi_sync)), N_orig, []);

    % 3. 產生均勻時間向量並先於均勻網格上插值，避免非均勻斷層直接濾波造成嚴重邊界振鈴
    t_uniform = (timestamp_sec(1) : 1/Fs_target : timestamp_sec(end))';
    N_uniform = length(t_uniform);

    % 在原始時間軸先做輕量 5 點移動中位數抑制單包毛刺，再插值至均勻時間軸
    if exist('movmedian', 'builtin') || exist('movmedian', 'file')
        amp_clean = movmedian(amp_2d, 5, 1);
    else
        amp_clean = amp_2d;
    end

    amp_res_2d    = interp1(timestamp_sec, amp_clean, t_uniform, 'linear', 'extrap');
    phasor_res_2d = interp1(timestamp_sec, phasor_2d, t_uniform, 'linear', 'extrap');

    % 4. 在均勻採樣時間軸上進行平滑抗混疊低通濾波 (截止頻率 12 Hz，完整涵蓋呼吸與 <= 0.5 m/s 體動都卜勒頻移)
    lp_Java_cutoff = min(12.0, 0.85 * (Fs_target / 2));
    [b_lp, a_lp] = butter(4, lp_Java_cutoff / (Fs_target / 2), 'low');
    if N_uniform > 18
        amp_res_2d    = filtfilt(b_lp, a_lp, amp_res_2d);
        phasor_res_2d = filtfilt(b_lp, a_lp, real(phasor_res_2d)) + ...
                        1i * filtfilt(b_lp, a_lp, imag(phasor_res_2d));
    end

    % 重新歸一化單位相角向量，使振幅與相位完全解耦
    phasor_res_mag = abs(phasor_res_2d);
    phasor_res_2d(phasor_res_mag > eps) = phasor_res_2d(phasor_res_mag > eps) ./ phasor_res_mag(phasor_res_mag > eps);

    csi_resampled_2d = max(0, amp_res_2d) .* phasor_res_2d;

    % =========================================================================
    % 5. 偵測真實訊號斷訊區間 (gap_mask)
    %    - 微觀封包抖動 (> 1.5 / Fs_orig) 僅作為封包統計參考；
    %    - 真正會影響 40 Hz 呼吸與都卜勒分析的斷訊門檻設為 >= 0.35 秒無封包 (相當於連續遺失 >= 70 個 200Hz 封包)，
    %      避免 10~50ms 的正常 Wi-Fi 排程抖動將完好的呼吸波峰誤判為掉包而剔除。
    % =========================================================================
    gap_mask = false(N_uniform, 1);
    diff_t = diff(timestamp_sec);

    micro_gap_thresh = (1 / Fs_orig) * 1.5;
    micro_gap_indices = find(diff_t > micro_gap_thresh);

    macro_gap_thresh = max(0.35, 10 / Fs_target); % >= 0.35 秒無任何封包視為真實斷訊斷層
    macro_gap_indices = find(diff_t >= macro_gap_thresh);

    for g = 1:length(macro_gap_indices)
        t_start = timestamp_sec(macro_gap_indices(g));
        t_end   = timestamp_sec(macro_gap_indices(g) + 1);
        gap_mask(t_uniform >= t_start & t_uniform <= t_end) = true;
    end

    lost_ratio = (sum(gap_mask) / N_uniform) * 100;

    % 6. 還原為 4D 矩陣 [N_uniform, 30, 2, 3]
    csi_resampled = reshape(csi_resampled_2d, N_uniform, Nsc, Ntx, Nrx);

    % 7. 繪製重採樣後之 CSI 波形圖 (僅在允許顯示圖表時繪製)
    if strcmpi(get(0, 'DefaultFigureVisible'), 'on')
        figure('Name', 'CSI Resampled Waveform', 'Color', 'w');
        sc_idx = 1; tx_idx = 1; rx_idx = 1;
        sample_signal = abs(csi_resampled(:, sc_idx, tx_idx, rx_idx));

        plot(t_uniform, sample_signal, 'LineWidth', 1.2, 'Color', [0 0.447 0.741]);
        grid on;
        xlabel('Time (seconds)', 'FontSize', 11);
        ylabel('Amplitude', 'FontSize', 11);
        title(sprintf('CSI Resampled Waveform (Subcarrier %d, TX %d, RX %d)', sc_idx, tx_idx, rx_idx), 'FontSize', 12);
        legend({'Resampled CSI'}, 'Location', 'best');
    end

    fprintf('重採樣完成！均勻點數: %d | 真實斷訊區段(>=%.2fs): %d 處 (微抖動 %d 處) | 有效斷訊佔比: %.2f%%\n', ...
            N_uniform, macro_gap_thresh, length(macro_gap_indices), length(micro_gap_indices), lost_ratio);
    fprintf('====== [重採樣模組] 處理完畢 ======\n\n');
end
