function [best_stream_name, best_signal, best_fpsd] = select_respiration_stream(amp_filtered, phase_filtered)
    % 呼吸特徵頻譜分析與自動選擇邏輯
    % 輸入:
    %   amp_filtered   - 濾波後的振幅矩陣 [N, 30]
    %   phase_filtered - 濾波後的相位矩陣 [N, 30] (已解捲繞)
    % 輸出:
    %   best_stream_name - 字串: 'Amplitude' 或 'Phase'
    %   best_signal      - 選擇的最佳時域訊號 (1D)
    %   best_fpsd        - 最佳訊號的峰值頻率 (Hz)

    %Z-score 歸一化
    % MATLAB 內建的 normalize 預設會對矩陣的每一個 Column (也就是每一個子載波) 獨立計算 mean 與 std
    % 這正是我們要的：讓 30 個子載波各自獨立拉平到相同的基準線上
    %amp_norm = normalize(amp_filtered, 'zscore');
    %phase_norm = normalize(phase_filtered, 'zscore');
    amp_norm = zscore(amp_filtered);
    phase_norm = zscore(phase_filtered);

    fs = 200; % 採樣頻率 200 Hz
    
    % 1. 降維：將 30 個子載波平均成一個代表性串流 (1D Array)
    amp_1d = mean(amp_norm, 2);
    phase_1d = mean(phase_norm, 2);
    
    N = length(amp_1d);
    if N == 0
        error('資料長度不可為零！');
    end

    %======================================================================
    % 2. 帶通濾波器 (0.1 Hz ~ 0.5 Hz)
    % 使用 3 階 Butterworth 濾波器
    [b, a] = butter(3, [0.1 0.2]/(fs/2), 'bandpass');
    %======================================================================
    
    % 使用 filtfilt 進行零相位濾波，避免訊號發生時間偏移
    amp_filt = filtfilt(b, a, amp_1d);
    phase_filt = filtfilt(b, a, phase_1d);

%=========================================
    % time_axis = (0:N-1) / 200;
    % % 上方圖形：帶通濾波後的振幅隨時間變化
    % subplot(2, 1, 1);
    % plot(time_axis, amp_filt, 'b', 'LineWidth', 1.2);
    % title(sprintf('振幅特徵 (帶通濾波 0.1-0.2 Hz)'));
    % xlabel('時間 (秒)');
    % ylabel('振幅變化量');
    % grid on; axis tight;
    % legend('Amplitude Stream', 'Location', 'northeast');
    % 
    % % 下方圖形：帶通濾波後的相位隨時間變化
    % subplot(2, 1, 2);
    % plot(time_axis, phase_filt, 'r', 'LineWidth', 1.2);
    % title(sprintf('相位特徵 (帶通濾波 0.1-0.2 Hz)'));
    % xlabel('時間 (秒)');
    % ylabel('相位變化量');
    % grid on; axis tight;
    % legend('Phase Stream', 'Location', 'northeast');
%===================================

    % 3. 快速傅立葉變換 (FFT) 與功率譜密度 (PSD) 計算
    % 定義頻率軸
    f = fs * (0:(N/2)) / N;
    
    % 計算振幅的 PSD
    Y_amp = fft(amp_filt);
    P2_amp = abs(Y_amp/N);
    P1_amp = P2_amp(1:floor(N/2)+1);
    P1_amp(2:end-1) = 2 * P1_amp(2:end-1);
    psd_amp = P1_amp.^2; % 功率譜密度
    
    % 計算相位的 PSD
    Y_phase = fft(phase_filt);
    P2_phase = abs(Y_phase/N);
    P1_phase = P2_phase(1:floor(N/2)+1);
    P1_phase(2:end-1) = 2 * P1_phase(2:end-1);
    psd_phase = P1_phase.^2;

    %======================================================================
    % 4. 尋找主要頻率峰值與初步過濾
    % 呼吸範圍：10 bpm ~ 37 bpm -> 約 0.167 Hz ~ 0.617 Hz
    freq_min = 5 / 60;
    freq_max = 10 / 60;
    %======================================================================
    
    [~, max_idx_amp] = max(psd_amp);
    peak_f_amp = f(max_idx_amp);
    amp_is_valid = (peak_f_amp >= freq_min) && (peak_f_amp <= freq_max);
    
    [~, max_idx_phase] = max(psd_phase);
    peak_f_phase = f(max_idx_phase);
    phase_is_valid = (peak_f_phase >= freq_min) && (peak_f_phase <= freq_max);

    % 5. 最佳串流選擇邏輯
    % 找出呼吸頻帶內的 PSD 區間以計算方差
    band_idx = (f >= freq_min) & (f <= freq_max);
    var_amp = var(psd_amp(band_idx));
    var_phase = var(psd_phase(band_idx));
    
    if amp_is_valid && ~phase_is_valid
        best_stream_name = 'Amplitude';
        best_signal = amp_filt;
        best_fpsd = peak_f_amp;
    elseif ~amp_is_valid && phase_is_valid
        best_stream_name = 'Phase';
        best_signal = phase_filt;
        best_fpsd = peak_f_phase;
    elseif amp_is_valid && phase_is_valid
        % 兩者皆符合，比較呼吸頻帶內的 PSD 方差
        if var_amp >= var_phase
            best_stream_name = 'Amplitude (依據方差較大)';
            best_signal = amp_filt;
            best_fpsd = peak_f_amp;
        else
            best_stream_name = 'Phase (依據方差較大)';
            best_signal = phase_filt;
            best_fpsd = peak_f_phase;
        end
    else
        % 若兩者皆不符合正常呼吸範圍 (可能是沒人在場或動作干擾)
        best_stream_name = 'None (無明顯呼吸特徵)';
        best_signal = zeros(N, 1);
        best_fpsd = 0;
    end

    % 6. 圖表顯示
    time_axis = (0:N-1) / fs; % 將封包索引轉換為時間(秒)
    
    figure('Name', '呼吸頻譜分析與最佳特徵選擇', 'Position', [150, 50, 900, 700]);
    
    % 主標題顯示選擇結果
    title(sprintf('自動選擇結果: %s | 估測呼吸頻率: %.2f Hz (約 %.1f bpm)', ...
        best_stream_name, best_fpsd, best_fpsd * 60), 'FontSize', 14, 'FontWeight', 'bold');
    
    % 上方圖形：帶通濾波後的振幅隨時間變化
    subplot(2, 1, 1);
    plot(time_axis, amp_filt, 'b', 'LineWidth', 1.2);
    title(sprintf('振幅特徵 (帶通濾波 0.1-0.2 Hz) | 峰值頻率: %.3f Hz', peak_f_amp));
    xlabel('時間 (秒)');
    ylabel('振幅變化量');
    grid on; axis tight;
    legend('Amplitude Stream', 'Location', 'northeast');
    
    % 下方圖形：帶通濾波後的相位隨時間變化
    subplot(2, 1, 2);
    plot(time_axis, phase_filt, 'r', 'LineWidth', 1.2);
    title(sprintf('相位特徵 (帶通濾波 0.1-0.2 Hz) | 峰值頻率: %.3f Hz', peak_f_phase));
    xlabel('時間 (秒)');
    ylabel('相位變化量');
    grid on; axis tight;
    legend('Phase Stream', 'Location', 'northeast');

    fprintf('分析完成！系統選擇了 [%s] 進行後續呼吸估算。\n', best_stream_name);
end