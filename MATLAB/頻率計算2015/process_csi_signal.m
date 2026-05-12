function [amp_pcs_norm, phase_pcs_norm] = process_csi_signal(csi_matrix)
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

    % 1. 執行共軛乘法 (Conjugate Multiplication)
    % 選擇 Tx1 與 Rx1, Rx2 進行處理，結果維度為 [N, 30]
    csi_tx1_rx1 = squeeze(csi_matrix(:, :, 1, 1)); 
    csi_tx1_rx2 = squeeze(csi_matrix(:, :, 1, 2));
    csi_conj = csi_tx1_rx1 .* conj(csi_tx1_rx2);

    % 2. 資料重塑與分離 (振幅與相位)
    % 提取振幅矩陣 [N x 30]
    amp_matrix = abs(csi_conj);
    
    % 提取相位矩陣並「預先執行相位展開 (Unwrap)」
    % 注意：unwrap 必須沿著時間軸 (第一維度) 執行，以消除 2*pi 跳變
    phase_matrix = unwrap(angle(csi_conj), [], 1);

    % 3. 執行 PCA 降維處理 (提取前 3 個主成分)
    fprintf('正在執行 PCA 降維 (提取前 3 主成分)...\n');
    
    % 對振幅執行 PCA
    % score 是投影後的座標，我們只取前三欄
    [~, score_amp, ~] = pca(amp_matrix);
    if size(score_amp, 2) >= 3
        amp_pcs = score_amp(:, 1:3);
    else
        amp_pcs = score_amp; % 若子載波不足 3 個則全取
    end

    % 對相位執行 PCA
    [~, score_phase, ~] = pca(phase_matrix);
    if size(score_phase, 2) >= 3
        phase_pcs = score_phase(:, 1:3);
    else
        phase_pcs = score_phase;
    end

    % 4. Savitzky-Golay 濾波處理
    % 設定視窗長度 31 (約 0.15s) 與 3 階多項式
    window_length = 31;
    poly_order = 3;

    if N > window_length
        amp_filtered = sgolayfilt(amp_pcs, poly_order, window_length);
        phase_filtered = sgolayfilt(phase_pcs, poly_order, window_length);
    else
        amp_filtered = amp_pcs;
        phase_filtered = phase_pcs;
        warning('資料長度不足，跳過 SG 濾波。');
    end

    time_axis = (0:N-1) / 200;
    % % 上方圖形：SG濾波後的振幅隨時間變化
    subplot(2, 1, 1);
    plot(time_axis, amp_filtered, 'b', 'LineWidth', 1.2);
    title(sprintf('振幅特徵 (SG濾波)'));
    xlabel('時間 (秒)');
    ylabel('振幅變化量');
    grid on; axis tight;
    legend('Amplitude Stream', 'Location', 'northeast');
    
    % % 下方圖形：SG濾波後的相位隨時間變化
    subplot(2, 1, 2);
    plot(time_axis, phase_filtered, 'r', 'LineWidth', 1.2);
    title(sprintf('相位特徵 (SG濾波)'));
    xlabel('時間 (秒)');
    ylabel('相位變化量');
    grid on; axis tight;
    legend('Phase Stream', 'Location', 'northeast');
    
    % 5. Z-score 歸一化
    % 對過濾後的三個 PC 分別進行歸一化，使量級統一
    amp_pcs_norm = zscore(amp_filtered);
    phase_pcs_norm = zscore(phase_filtered);

    fprintf('訊號處理完成！已完成共軛相乘與 SG 濾波 (視窗:%d, 階數:%d)。\n', window_length, poly_order);
end