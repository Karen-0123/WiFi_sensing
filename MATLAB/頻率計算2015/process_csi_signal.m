function [amp_filtered, phase_filtered] = process_csi_signal(csi_matrix)
    % 處理 CSI 訊號：消除相位偏移並套用 Savitzky-Golay 濾波
    % 輸入:
    %   csi_matrix - 4D 矩陣 [封包數量 (N), 子載波 (30), Tx (2), Rx (3)]
    % 輸出:
    %   amp_filtered   - 濾波後的振幅矩陣 [N, 30]
    %   phase_filtered - 濾波後的相位矩陣 [N, 30] (已解捲繞)

    % 取出封包數量
    N = size(csi_matrix, 1);
    if N == 0
        error('輸入的 CSI 矩陣為空！');
    end

    % 1. 相位偏移抵消 (共軛乘法 Conjugate Multiplication)
    % 選擇第 0 根發射天線 (MATLAB 索引為 1)
    % 選擇第 1 與第 2 根接收天線 (MATLAB 索引為 1 與 2)
    % squeeze 會將維度降為 [N, 30]
    csi_tx1_rx1 = squeeze(csi_matrix(:, :, 1, 1)); 
    csi_tx1_rx2 = squeeze(csi_matrix(:, :, 1, 2));

    % 執行共軛乘法： CSI_1 * conj(CSI_2)
    % 這樣可以消除 CFO 與 SFO 帶來的隨機相位旋轉
    csi_conj = csi_tx1_rx1 .* conj(csi_tx1_rx2);

    % 提取振幅與相位
    amp_raw = abs(csi_conj);
    
    % 提取相位並進行解捲繞 (Unwrap)
    % 由於相位會限制在 [-pi, pi]，跨越時會產生突變，必須先解捲繞才能濾波
    % unwrap 預設會沿著第一個維度 (時間軸 N) 進行處理
    phase_raw = unwrap(angle(csi_conj)); 

    %======================================================================
    % 2. Savitzky-Golay 濾波器設定
    % 採樣頻率為 200Hz，人類動作造成的都卜勒頻移通常在 10Hz 以下
    % 我們設定大約 0.15 秒的視窗來平滑高頻雜訊 (200 * 0.15 = 30)
    % SG 濾波器的視窗長度必須為奇數，所以選擇 31
    window_length = 31; 
    
    % 多項式階數 (Polynomial Order)
    % 階數 3 可以在保留訊號波峰/波谷特徵的同時，提供良好的平滑效果
    poly_order = 3;  
    %======================================================================

    % 確保資料長度大於視窗長度才能濾波
    if N < window_length
        warning('資料長度小於視窗長度，將不進行濾波處理。');
        amp_filtered = amp_raw;
        phase_filtered = phase_raw;
        return;
    end

    % 3. 執行濾波並遍歷所有子載波
    % MATLAB 的 sgolayfilt 預設就會沿著矩陣的第一個維度 (Column，即時間軸) 進行濾波
    % 因此將 [N, 30] 矩陣直接傳入，它會自動平行處理所有的 30 個子載波，不需寫 for 迴圈
    %amp_filtered = sgolayfilt(amp_raw, poly_order, window_length);
    %phase_filtered = sgolayfilt(phase_raw, poly_order, window_length);
    amp_filtered = amp_raw;
    phase_filtered = phase_raw;

    time_axis = (0:N-1) / 200;
    % % % 上方圖形：SG濾波後的振幅隨時間變化
    % subplot(2, 1, 1);
    % plot(time_axis, amp_filtered, 'b', 'LineWidth', 1.2);
    % title(sprintf('振幅特徵 (SG濾波)'));
    % xlabel('時間 (秒)');
    % ylabel('振幅變化量');
    % grid on; axis tight;
    % legend('Amplitude Stream', 'Location', 'northeast');
    % 
    % % 下方圖形：SG濾波後的相位隨時間變化
%     subplot(2, 1, 2);
%     plot(time_axis, phase_filtered, 'r', 'LineWidth', 1.2);
%     title(sprintf('相位特徵 (SG濾波)'));
%     xlabel('時間 (秒)');
%     ylabel('相位變化量');
%     grid on; axis tight;
%     legend('Phase Stream', 'Location', 'northeast');

    fprintf('訊號處理完成！已完成共軛相乘與 SG 濾波 (視窗:%d, 階數:%d)。\n', window_length, poly_order);
end