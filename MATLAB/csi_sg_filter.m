function respiration_final_integrated()
    %% Wi-Fi CSI 呼吸監測 - 全功能整合復刻版
    clear; clc; close all;

    % 1. 設定你的檔案路徑 (已更新為你的路徑)
    filename = 'C:\Users\Admin\OneDrive\Documents\MATLAB\WiFi_sensing\data\breathe\breathe_200hz_015.dat';
    fs = 200; 

    % 2. 執行處理流程
    fprintf('步驟 1: 正在讀取資料...\n');
    [csi_matrix] = read_intel5300_dat(filename);

    fprintf('步驟 2: 處理訊號 (消除相位偏移與 SG 濾波)...\n');
    [amp_filtered, phase_filtered] = process_csi_signal(csi_matrix);

    fprintf('步驟 3: 執行帶通濾波 (0.1-0.2 Hz) 與特徵選擇...\n');
    [best_name, best_sig] = select_respiration_stream(amp_filtered, phase_filtered, fs);

    fprintf('步驟 4: 執行精細峰值偵測 (True vs False Peaks)...\n');
    [p_idx, p_val, c_idx, c_val] = detect_respiration_peaks_classic(best_sig, fs);

    fprintf('步驟 5: 計算動態呼吸率 (BPM)...\n');
    [bpm_timeline, time_axis_bpm] = calculate_dynamic_bpm(p_idx, length(best_sig), fs);

    %% 3. 整合視覺化 (確保圖表與舊圖一模一樣)
    hFig = figure('Name', 'CSI Respiration Result', 'Color', 'w', 'Position', [100, 100, 900, 750]);
    t_sig = (0:length(best_sig)-1)/fs;

    % --- 子圖 1: 呼吸波形與峰值偵測 (對標 Figure 2) ---
    subplot(2, 1, 1);
    plot(t_sig, best_sig, 'b', 'LineWidth', 1.5); hold on;
    % 畫出灰色小叉叉 (False Peaks)
    plot(c_idx/fs, c_val, 'x', 'Color', [0.6 0.6 0.6], 'MarkerSize', 7);
    % 畫出紅色圓點 (True Peaks)
    plot(p_idx/fs, p_val, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 7);
    
    title(['[1] Respiration Waveform (Source: ', best_name, ') - Window-based Validation'], 'FontWeight', 'bold');
    ylabel('Amplitude'); grid on;
    legend('Filtered Signal', 'False Peaks', 'True Peaks', 'Location', 'northeast');

    % --- 子圖 2: 動態呼吸率 (BPM 曲線) ---
    subplot(2, 1, 2);
    avg_bpm = nanmean(bpm_timeline);
    % 綠色點線圖 (對標舊圖樣式)
    plot(time_axis_bpm, bpm_timeline, '-s', 'Color', [0.2 0.7 0.2], 'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerFaceColor', 'w');
    hold on;
    % 紅色虛線平均值基準線 (相容 2015 版)
    plot([0 t_sig(end)], [avg_bpm avg_bpm], 'r--', 'LineWidth', 2);
    
    title(sprintf('[2] Dynamic Respiration Rate (Average: %.1f BPM)', avg_bpm), 'FontWeight', 'bold');
    xlabel('Time (seconds)'); ylabel('BPM'); ylim([5 35]); grid on;
    text(t_sig(end)*0.85, avg_bpm+2, ['Avg: ', num2str(round(avg_bpm,1))], 'Color', 'r', 'FontWeight', 'bold');

    fprintf('所有流程執行完畢！整合圖表已產出。\n');
end

% =========================================================================
% 下方為整合的功能函式區
% =========================================================================

function [csi_matrix] = read_intel5300_dat(filename)
    try
        csi_trace = read_bf_file(filename);
    catch
        error('找不到 read_bf_file.m，請確認已加入路徑！');
    end
    N = length(csi_trace);
    csi_matrix = zeros(N, 30, 2, 3);
    v = 0;
    for i = 1:N
        entry = csi_trace{i};
        if isempty(entry) || ~isfield(entry, 'csi'), continue; end
        v = v + 1;
        csi_matrix(v, :, 1:size(entry.csi,1), 1:size(entry.csi,2)) = permute(entry.csi, [3, 1, 2]);
    end
    csi_matrix = csi_matrix(1:v, :, :, :);
end

function [amp_f, phase_f] = process_csi_signal(csi_matrix)
    c1 = squeeze(csi_matrix(:,:,1,1)); 
    c2 = squeeze(csi_matrix(:,:,1,2));
    cj = c1 .* conj(c2);
    amp_f = sgolayfilt(abs(cj), 3, 31);
    phase_f = sgolayfilt(unwrap(angle(cj)), 3, 31);
end

function [best_name, best_signal] = select_respiration_stream(amp_f, ph_f, fs)
    % Z-score 歸一化 (2015 兼容版)
    an = bsxfun(@rdivide, bsxfun(@minus, amp_f, mean(amp_f)), std(amp_f));
    pn = bsxfun(@rdivide, bsxfun(@minus, ph_f, mean(ph_f)), std(ph_f));
    a1 = mean(an, 2); p1 = mean(pn, 2);
    % 帶通濾波器 (0.1-0.2 Hz)
    [b, a] = butter(3, [0.1 0.2]/(fs/2), 'bandpass');
    as = filtfilt(b, a, a1); ps = filtfilt(b, a, p1);
    if var(as) >= var(ps), best_name = 'Amplitude'; best_signal = as; 
    else best_name = 'Phase'; best_signal = ps; end
end

function [p_idx, p_val, c_idx, c_val] = detect_respiration_peaks_classic(signal, fs)
    % 取得所有候選點 (灰色叉叉)
    [c_val, c_idx] = findpeaks(signal);
    win = round(1.5 * fs);
    p_idx = []; p_val = [];
    % 驗證真點 (紅色圓點)
    for i = 1:length(c_idx)
        curr = c_idx(i);
        r = max(1, curr-floor(win/2)):min(length(signal), curr+floor(win/2));
        if signal(curr) >= max(signal(r))
            p_idx(end+1) = curr; p_val(end+1) = signal(curr);
        end
    end
end

function [bpm, t_bpm] = calculate_dynamic_bpm(p_idx, total_samples, fs)
    tp = p_idx / fs;
    win = 20; step = 1;
    t_s = 0:step:(total_samples/fs - win);
    bpm = zeros(1, length(t_s)); t_bpm = zeros(1, length(t_s));
    for i = 1:length(t_s)
        ts = t_s(i); te = ts + win; t_bpm(i) = ts + win/2;
        p_in = tp(tp >= ts & tp <= te);
        if length(p_in) >= 2, bpm(i) = 60 / mean(diff(p_in));
        else bpm(i) = NaN; end
    end
end