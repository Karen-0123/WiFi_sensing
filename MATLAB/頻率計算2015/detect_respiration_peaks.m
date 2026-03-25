function [true_peak_idx, true_peak_vals] = detect_respiration_peaks(signal)
    % 呼吸訊號的精細峰值偵測與虛假峰值剔除
    % 輸入:
    %   signal - 1D 時域訊號 (來自上一步的最佳串流)
    % 輸出:
    %   true_peak_idx  - 有效峰值的索引值陣列
    %   true_peak_vals - 有效峰值的訊號數值陣列

    fs = 200; % 採樣頻率 200 Hz
    N = length(signal);
    time_axis = (0:N-1) / fs; % 時間軸 (秒)

    % 1. 初步偵測：找出訊號中所有的局部最大值 (Local Maxima)
    % 使用 MATLAB 內建的 findpeaks 來尋找所有的候選波峰
    [candidate_pks, candidate_locs] = findpeaks(signal);

    %======================================================================
    % 2. 虛假峰值剔除 (Window-based Validation)
    % 設定 1.5 秒的驗證窗口 (包含當前點，前後共涵蓋 1.5 秒)
    window_time = 5; 
    window_samples = round(window_time * fs); % 1.5 * 200 = 300 個採樣點
    %======================================================================
    
    % 將窗口分為前後兩半，各約 0.75 秒 (150 個點)
    half_window = floor(window_samples / 2); 

    true_peak_idx = [];
    true_peak_vals = [];

    % 3. 邏輯判斷：驗證每一個候選峰值
    for i = 1:length(candidate_locs)
        curr_idx = candidate_locs(i);
        curr_val = candidate_pks(i);

        % 定義窗口的安全邊界，避免在陣列頭尾時發生 Index Out of Bounds 錯誤
        start_idx = max(1, curr_idx - half_window);
        end_idx = min(N, curr_idx + half_window);

        % 提取該時間窗口內的所有原始訊號
        window_signal = signal(start_idx:end_idx);

        % 檢查當前候選峰值是否為該窗口範圍內的「絕對最大值」
        % (若窗口內有其他點的值大於當前峰值，代表當前點只是上升/下降段的微小雜訊)
        if curr_val >= max(window_signal)
            % 通過驗證，保留為實際呼吸頂點
            true_peak_idx(end+1) = curr_idx;
            true_peak_vals(end+1) = curr_val;
        end
    end

    % 4. 繪製視覺化圖表
    figure('Name', '呼吸特徵峰值偵測', 'Position', [200, 200, 900, 400]);
    
    % 畫出原始處理後的平滑波形
    plot(time_axis, signal, 'b', 'LineWidth', 1.5);
    hold on;

    % 畫出剔除前所有的「候選峰值」(用灰色小叉叉標示，方便對比觀察剔除效果)
    plot(time_axis(candidate_locs), candidate_pks, 'x', 'Color', [0.6 0.6 0.6], 'MarkerSize', 6);

    % 畫出剔除後的「實際呼吸頂點」(用明顯的紅點標示)
    plot(time_axis(true_peak_idx), true_peak_vals, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');

    title('呼吸波形與精細峰值偵測 (Window-based Validation)', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('時間 (秒)', 'FontSize', 12);
    ylabel('特徵訊號振幅', 'FontSize', 12);
    
    % 設定圖例 (Legend)
    legend('呼吸時域訊號', '被剔除的虛假峰值 (False Peaks)', '有效呼吸頂點 (True Peaks)', ...
           'Location', 'northeast', 'FontSize', 10);
    
    grid on;
    axis tight;
    hold off;

    % 在終端機印出處理結果
    fprintf('峰值偵測完成！初步找到 %d 個候選峰值，經 1.5 秒窗口剔除後，保留 %d 個有效呼吸頂點。\n', ...
            length(candidate_locs), length(true_peak_idx));
end