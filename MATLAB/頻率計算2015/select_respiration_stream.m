function [best_stream_name, best_signal, best_fpsd] = select_respiration_stream(amp_pcs_norm, phase_pcs_norm)
    % 基於 PCA 6路串流的呼吸特徵自動選擇邏輯
    % 輸入: 
    %   amp_pcs_norm, phase_pcs_norm: 均為 [N x 3] 的矩陣 (PC1, PC2, PC3)

    fs = 200; 
    N = size(amp_pcs_norm, 1);
    
    % 1. 準備 6 個串流與標籤
    % 串流順序：Amp-PC1, Amp-PC2, Amp-PC3, Phase-PC1, Phase-PC2, Phase-PC3
    all_streams = [amp_pcs_norm, phase_pcs_norm];
    labels = {'Amp-PC1', 'Amp-PC2', 'Amp-PC3', 'Phase-PC1', 'Phase-PC2', 'Phase-PC3'};
    num_streams = 6;

%======================================================================
    % 帶通濾波器 (0.1 Hz ~ 0.5 Hz)
    [b, a] = butter(3, [0.1 0.5]/(fs/2), 'bandpass');

    % 呼吸頻率判定範圍 (10bpm - 37bpm)
    freq_min = 10 / 60; % 0.167 Hz
    freq_max = 37 / 60; % 0.617 Hz
%======================================================================    

    % 頻率軸設定
    f = fs * (0:(N/2)) / N;
    band_idx = (f >= freq_min) & (f <= freq_max); % 呼吸頻帶索引

    % 初始化紀錄變數
    scores = -ones(1, num_streams); % 儲存每個串流的分數
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
        psd_in_band(~band_idx) = 0; % 將非呼吸頻帶設為 0
        [max_val, max_idx] = max(psd_in_band);
        peak_f = f(max_idx);
        peak_freqs(i) = peak_f;

        % 頻率判定：檢查峰值是否落在 10-37 bpm
        if peak_f >= freq_min && peak_f <= freq_max
            % 計算指標 A: PSD 方差 (代表能量集中度)
            v = var(psd_data(band_idx));
            
            % 計算指標 B: 峰值顯著性 (Prominence)
            % 使用 findpeaks 找出主峰與其突起程度
            [pks, locs, w, prom] = findpeaks(psd_data, 'MinPeakHeight', max_val*0.8);
            if ~isempty(prom)
                p_val = max(prom); % 取最顯著的峰值突起程度
            else
                p_val = 0;
            end
            
            % 綜合評分 = 方差 * 顯著性 (兩者皆大者最優)
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
        % 若沒有任何串流符合
        best_stream_name = 'None (No Valid Respiration Found)';
        best_signal = zeros(N, 1);
        best_fpsd = 0;
    end

    % 4. 視覺化更新
    time_axis = (0:N-1) / fs;
    figure('Name', 'PCA 多路串流自動選擇系統', 'Position', [100, 100, 1000, 600]);
    
    % 繪製所有 6 路訊號供參考 (淺色)
    subplot(2,1,1);
    plot(time_axis, filtered_signals, 'Color', [0.8 0.8 0.8], 'LineWidth', 0.5);
    hold on;
    % 突出顯示選中的最佳路徑
    if best_idx > 0
        plot(time_axis, best_signal, 'r', 'LineWidth', 1.5);
    end
    title(['所有 PCA 串流對比 (選中: ', best_stream_name, ')'], 'FontSize', 12);
    xlabel('時間 (秒)'); ylabel('幅值');
    h = legend([labels, {'Selected'}], ...
           'Location', 'southoutside', ...
           'Orientation', 'horizontal');

    set(h, 'FontSize', 8);
    grid on;

    % 繪製最佳訊號的 PSD
    subplot(2,1,2);
    if best_idx > 0
        best_Y = fft(best_signal);
        best_P2 = abs(best_Y/N);
        best_P1 = best_P2(1:floor(N/2)+1);
        best_P1(2:end-1) = 2 * best_P1(2:end-1);
        plot(f, best_P1.^2, 'LineWidth', 1.5);
        xlim([0 1.5]); % 聚焦低頻區
        yl = ylim;
        line([freq_min freq_min], yl, ...
            'Color', 'g', ...
            'LineStyle', '--');
        line([freq_max freq_max], yl, ...
            'Color', 'r', ...
            'LineStyle', '--');
        text(freq_min, yl(2), '下限');
        text(freq_max, yl(2), '上限');
        title(['最佳串流 PSD 分析 - 估計呼吸率: ', num2str(best_fpsd*60, '%.2f'), ' BPM']);
        xlabel('頻率 (Hz)'); ylabel('功率譜密度 (PSD)');
        grid on;
    end

    fprintf('自動選擇完成：最終選定 [%s]\n', best_stream_name);
end