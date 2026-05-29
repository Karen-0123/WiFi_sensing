function [best_stream_name, best_signal, best_fpsd] = select_respiration_stream(amp_pcs_norm, phase_pcs_norm, Fs_target)
    % 基於 PCA 6路串流的呼吸特徵自動選擇邏輯
    % 輸入: amp_pcs_norm, phase_pcs_norm [N x 3] 矩陣

    
    if nargin < 2, Fs_target = 20; end  % 目標採樣率預設 20 Hz
    N = size(amp_pcs_norm, 1);
    
    % 1. 準備 2 個串流與標籤
    all_streams = [amp_pcs_norm, phase_pcs_norm];
    labels = {'Amp-PC1', 'Phase-PC1'};
    num_streams = 2;

    % 帶通濾波器自適應調整：0.1 Hz ~ 0.7 Hz (完美包覆 10~37 bpm 的 0.167~0.617 Hz)
    [b, a] = butter(3, [0.1, 0.7]/(Fs_target/2), 'bandpass');

    % 呼吸頻率判定範圍 (10bpm - 37bpm)
    freq_min = 10 / 60; % 0.167 Hz
    freq_max = 37 / 60; % 0.617 Hz

    % 頻率軸設定
    f = Fs_target * (0:(N/2)) / N;
    band_idx = (f >= freq_min) & (f <= freq_max); % 呼吸頻帶索引

    % 初始化紀錄變數
    scores = -ones(1, num_streams); 
    peak_freqs = zeros(1, num_streams);
    filtered_signals = zeros(N, num_streams);
    
    % 2. 遍歷 6 個串流進行評估
    for i = 1:num_streams
        % 帶通濾波
        sig_filt = filtfilt(b, a, all_streams(:, i));
        filtered_signals(:, i) = sig_filt;

        % 計算 PSD
        Y = fft(sig_filt);
        P2 = abs(Y/N);
        P1 = P2(1:floor(N/2)+1);
        P1(2:end-1) = 2 * P1(2:end-1);
        psd_data = P1.^2;

        % 找出呼吸頻帶內的最大峰值
        psd_in_band = psd_data;
        psd_in_band(~band_idx) = 0; 
        [max_val, max_idx] = max(psd_in_band);
        peak_f = f(max_idx);
        peak_freqs(i) = peak_f;

        % 頻率判定
        if peak_f >= freq_min && peak_f <= freq_max
            % 指標 A: 呼吸頻帶內的 PSD 方差
            v = var(psd_data(band_idx));
            
            % 【BUG 修正】核心優化：精確計算「該呼吸主峰」的 Prominence，而非全域最大值
            [~, locs, ~, prom] = findpeaks(psd_data);
            peak_match_idx = find(locs == max_idx, 1);
            
            if ~isempty(peak_match_idx)
                p_val = prom(peak_match_idx); % 確實拿取呼吸主峰的突起顯著度
            else
                p_val = 0; % 若非局部極大值則給予 0 分
            end
            
            % 綜合評分
            scores(i) = v * p_val;
        end
    end

    % 3. 最佳化選擇
    [best_score, best_idx] = max(scores);

    if best_score > 0
        best_stream_name = labels{best_idx};
        best_signal = filtered_signals(:, best_idx);
        best_fpsd = peak_freqs(best_idx);
    else
        best_stream_name = 'None (No Valid Respiration Found)';
        best_signal = zeros(N, 1);
        best_fpsd = 0;
        best_idx = 0; % 確保下方繪圖邏輯正確
    end

    % 4. 視覺化更新
    time_axis = (0:N-1) / Fs_target;
    figure('Name', 'PCA 多路串流自動選擇系統', 'Position', [100, 100, 1000, 600]);
    
    subplot(2,1,1);
    plot(time_axis, filtered_signals, 'Color', [0.8 0.8 0.8], 'LineWidth', 0.5);
    hold on;
    
    % 【BUG 修正】動態調整圖例，防止無訊號時報錯
    if best_idx > 0
        plot(time_axis, best_signal, 'r', 'LineWidth', 1.5);
        title(['所有 PCA 串流對比 (最終選定: ', best_stream_name, ')'], 'FontSize', 12);
        h = legend([labels, {'Selected'}], 'Location', 'southoutside', 'Orientation', 'horizontal');
    else
        title(['所有 PCA 串流對比 (狀態: ', best_stream_name, ')'], 'FontSize', 12, 'Color', 'r');
        h = legend(labels, 'Location', 'southoutside', 'Orientation', 'horizontal');
    end
    xlabel('時間 (秒)'); ylabel('幅值');
    set(h, 'FontSize', 8);
    grid on;

    subplot(2,1,2);
    if best_idx > 0
        best_Y = fft(best_signal);
        best_P2 = abs(best_Y/N);
        best_P1 = best_P2(1:floor(N/2)+1);
        best_P1(2:end-1) = 2 * best_P1(2:end-1);
        plot(f, best_P1.^2, 'LineWidth', 1.5);
        xlim([0 1.5]); 
        yl = ylim;
        line([freq_min freq_min], yl, 'Color', 'g', 'LineStyle', '--');
        line([freq_max freq_max], yl, 'Color', 'g', 'LineStyle', '--');
        text(freq_min, yl(2)*0.9, ' 呼吸下限 (10 bpm)', 'Color', 'g');
        text(freq_max, yl(2)*0.9, ' 呼吸上限 (37 bpm)', 'Color', 'g');
        title(['最佳串流 PSD 分析 - 估計呼吸率: ', num2str(best_fpsd*60, '%.2f'), ' BPM']);
        xlabel('頻率 (Hz)'); ylabel('功率譜密度 (PSD)');
        grid on;
    else
        text(0.4, 0.5, '無有效呼吸頻譜可供顯示', 'FontSize', 14, 'Color', 'r');
    end
    
    fprintf('自動選擇完成：最終選定 [%s]\n', best_stream_name);
end