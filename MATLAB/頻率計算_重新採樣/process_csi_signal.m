function [amp_pc1_norm, phase_pc1_norm] = process_csi_signal(csi_matrix, Fs_target)
% process_csi_signal  MRC-PCA 架構的 CSI 呼吸訊號提取
%
% 架構流程：
%   [N,30,2,3] CSI
%       ↓
%   Step 1: MRC 加權合併 (3 Rx → 1 虛擬 Rx，per Tx per Subcarrier)
%       ↓
%   [N,30,2,1] 合併後 CSI
%       ↓
%   Step 2: 共軛相乘消除相位偏移 (2 Tx × 1 Rx對 = 2 組合 × 30 = 60 特徵)
%       ↓
%   Step 3: PCA 降維 → PC1 (呼吸主成分)
%       ↓
%   Step 4: Savitzky-Golay 濾波
%       ↓
%   Step 5: Z-score 歸一化
%
% 輸入:
%   csi_matrix  - [N, 30, 2, 3]  (N封包, 30子載波, 2 Tx, 3 Rx)
%   Fs_target   - 重採樣後的採樣率 (預設 20 Hz)
%
% 輸出:
%   amp_pc1_norm   - MRC-PCA 振幅 PC1，Z-score 歸一化 [N, 1]
%   phase_pc1_norm - MRC-PCA 相位 PC1，Z-score 歸一化 [N, 1]

    if nargin < 2, Fs_target = 20; end
    N = size(csi_matrix, 1);
    if N == 0, error('輸入的 CSI 矩陣為空！'); end

    % =========================================================================
    % Step 1: MRC — 計算每根 Rx 天線的加權權重並合併
    % =========================================================================
    % MRC 原理：w_rx = mean_power(rx) / sum(mean_power(all_rx))
    %   合併訊號 = Σ conj(w_rx) * rx_signal
    %   當權重基於功率時，等效為 SNR 最大化合併
    %
    % 實作細節：
    %   - 功率計算跨越 [時間, 子載波] 兩個維度取平均，代表該天線的全頻寬平均功率
    %   - 對每個 Tx 分別計算獨立的 MRC 權重（因不同 Tx 的路徑特性不同）
    %   - 合併後保留複數形式以保留相位資訊供後續共軛相乘使用
    % =========================================================================

    fprintf('[MRC] 計算 3 根 Rx 天線加權權重並執行 Maximal Ratio Combining...\n');

    % 合併結果存於 [N, 30, 2]（Rx 維度消除）
    csi_mrc = zeros(N, 30, 2);

    for tx = 1:2
        % 提取當前 Tx 下所有 Rx 訊號，各為 [N, 30]
        rx_signals = cell(1, 3);
        power_per_rx = zeros(1, 3);

        for rx = 1:3
            rx_signals{rx} = squeeze(csi_matrix(:, :, tx, rx));  % [N, 30]
            % 平均功率 = mean(|h|^2) over 時間 × 子載波
            power_per_rx(rx) = mean(mean(abs(rx_signals{rx}).^2));
        end

        % 歸一化權重（使權重總和為 1，保持訊號尺度穩定）
        total_power = sum(power_per_rx);
        if total_power < eps
            weights = ones(1, 3) / 3;  % 功率為零時退化為等權重
            warning('[MRC] Tx%d 總功率接近零，使用等權重合併。', tx);
        else
            weights = power_per_rx / total_power;
        end

        fprintf('[MRC]   Tx%d 天線功率比例: Rx1=%.3f  Rx2=%.3f  Rx3=%.3f\n', ...
                tx, weights(1), weights(2), weights(3));

        % MRC 合併：加權疊加 (複數保留)
        % shape: [N, 30]
        csi_mrc(:, :, tx) = weights(1) * rx_signals{1} + ...
                             weights(2) * rx_signals{2} + ...
                             weights(3) * rx_signals{3};
    end

    fprintf('[MRC] 完成！輸出虛擬天線矩陣大小: [%d, 30, 2]\n', N);

    % =========================================================================
    % Step 2: 共軛相乘消除隨機相位偏移
    % =========================================================================
    % MRC 合併後每個 Tx 僅剩 1 根虛擬 Rx：
    %   - Tx1_vRx × conj(Tx2_vRx)：跨 Tx 消除公共相位偏移
    %   - 此處保留 2 個 Tx 的自共軛（對自身取 |h|^2）作為補充特徵
    %   - 總共 3 組合 × 30 子載波 = 90 個融合特徵
    %
    % 組合說明：
    %   combo1 [1:30]   = Tx1_vRx  .*  conj(Tx2_vRx)   ← 跨Tx差分，最重要
    %   combo2 [31:60]  = Tx1_vRx  .*  conj(Tx1_vRx)   = |Tx1|^2（功率包絡）
    %   combo3 [61:90]  = Tx2_vRx  .*  conj(Tx2_vRx)   = |Tx2|^2（功率包絡）
    % =========================================================================

    fprintf('[共軛相乘] 對 MRC 輸出執行跨 Tx 共軛相乘 (90 個融合特徵)...\n');

    tx1 = squeeze(csi_mrc(:, :, 1));   % [N, 30]
    tx2 = squeeze(csi_mrc(:, :, 2));   % [N, 30]

    csi_features = [tx1 .* conj(tx2), ...   % 跨Tx差分相位（含振幅乘積）
                    tx1 .* conj(tx1), ...   % Tx1 功率包絡（純實數）
                    tx2 .* conj(tx2)];      % Tx2 功率包絡（純實數）
    % csi_features: [N, 90]

    % 分離振幅與相位
    amp_matrix   = abs(csi_features);                      % [N, 90]
    phase_matrix = unwrap(angle(csi_features), [], 1);     % [N, 90]

    % =========================================================================
    % Step 3: PCA 降維 — 提取呼吸主成分 PC1
    % =========================================================================
    % 從 90 個融合特徵中提取方差最大的主成分（最接近呼吸週期的訊號）
    % 僅保留 PC1（前 3 個用於診斷輸出）
    % =========================================================================

    fprintf('[PCA] 對 MRC 合併特徵執行 PCA 降維 (提取 PC1)...\n');

    [coeff_amp, score_amp, latent_amp] = pca(amp_matrix);
    [coeff_phase, score_phase, latent_phase] = pca(phase_matrix);

    % 取前 3 個主成分（用於視覺化）；PC1 為主要輸出
    n_pcs = min(3, size(score_amp, 2));
    amp_pcs   = score_amp(:, 1:n_pcs);    % [N, 3]
    phase_pcs = score_phase(:, 1:n_pcs);  % [N, 3]

    % 列印各主成分解釋方差比例（有助判斷呼吸訊號強度）
    total_var_amp   = sum(latent_amp);
    total_var_phase = sum(latent_phase);
    fprintf('[PCA] 振幅 PC1 解釋方差: %.1f%%  PC2: %.1f%%  PC3: %.1f%%\n', ...
            latent_amp(1)/total_var_amp*100, ...
            latent_amp(2)/total_var_amp*100, ...
            latent_amp(3)/total_var_amp*100);
    fprintf('[PCA] 相位 PC1 解釋方差: %.1f%%  PC2: %.1f%%  PC3: %.1f%%\n', ...
            latent_phase(1)/total_var_phase*100, ...
            latent_phase(2)/total_var_phase*100, ...
            latent_phase(3)/total_var_phase*100);

    % =========================================================================
    % Step 4: Savitzky-Golay 濾波 (保持局部波形形狀)
    % =========================================================================

    window_length = 2 * floor((0.5 * Fs_target) / 2) + 1;  % 0.5 秒視窗，需為奇數
    poly_order = 3;

    if N > window_length
        amp_filtered   = sgolayfilt(amp_pcs,   poly_order, window_length);
        phase_filtered = sgolayfilt(phase_pcs, poly_order, window_length);
    else
        amp_filtered   = amp_pcs;
        phase_filtered = phase_pcs;
        warning('[SG濾波] 資料長度不足，跳過 SG 濾波。');
    end

    % =========================================================================
    % Step 5: Z-score 歸一化
    % =========================================================================

    amp_pcs_norm   = zscore(amp_filtered);    % [N, 3]
    phase_pcs_norm = zscore(phase_filtered);  % [N, 3]

    % 主要輸出為 PC1
    amp_pc1_norm   = amp_pcs_norm(:, 1);      % [N, 1]
    phase_pc1_norm = phase_pcs_norm(:, 1);    % [N, 1]

    % =========================================================================
    % 視覺化
    % =========================================================================

    time_axis = (0:N-1) / Fs_target;

    figure('Name', 'MRC-PCA CSI 呼吸訊號提取結果', 'NumberTitle', 'off');

    % --- 子圖1: MRC 天線合併效果（各Tx的3根Rx vs MRC輸出功率包絡）---
    subplot(3, 2, [1, 2]);
    % 以 Tx1 的子載波 #15 為例展示 MRC 合併效果
    sc = 15;
    rx1_env = abs(squeeze(csi_matrix(:, sc, 1, 1)));
    rx2_env = abs(squeeze(csi_matrix(:, sc, 1, 2)));
    rx3_env = abs(squeeze(csi_matrix(:, sc, 1, 3)));
    mrc_env = abs(squeeze(csi_mrc(:, sc, 1)));
    plot(time_axis, rx1_env, 'Color', [0.7 0.7 0.7], 'LineWidth', 0.8); hold on;
    plot(time_axis, rx2_env, 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8);
    plot(time_axis, rx3_env, 'Color', [0.3 0.3 0.3], 'LineWidth', 0.8);
    plot(time_axis, mrc_env, 'r',  'LineWidth', 1.8);
    hold off;
    title(sprintf('MRC 合併效果 (Tx1, 子載波 #%d)', sc));
    legend('Rx1', 'Rx2', 'Rx3', 'MRC 輸出', 'Location', 'best');
    xlabel('時間 (秒)'); ylabel('振幅'); grid on; axis tight;

    % --- 子圖2: 振幅 PC1~PC3 ---
    subplot(3, 2, 3);
    plot(time_axis, amp_pcs_norm, 'LineWidth', 1.2);
    title('MRC-PCA 振幅主成分 (SG 濾波 + Z-score)');
    legend('PC1', 'PC2', 'PC3', 'Location', 'best');
    xlabel('時間 (秒)'); ylabel('Z-score'); grid on; axis tight;

    % --- 子圖3: 振幅 PC1 解釋方差圓餅圖 ---
    subplot(3, 2, 4);
    var_ratio_amp = latent_amp(1:min(5,end)) / total_var_amp * 100;
    bar(var_ratio_amp, 'FaceColor', [0.2 0.6 0.8]);
    title('振幅 PCA 各主成分解釋方差 (%)');
    xlabel('主成分'); ylabel('解釋方差 (%)'); grid on;
%     xticklabels(arrayfun(@(x) sprintf('PC%d', x), 1:length(var_ratio_amp), ...
%                 'UniformOutput', false));

    % --- 子圖4: 相位 PC1~PC3 ---
    subplot(3, 2, 5);
    plot(time_axis, phase_pcs_norm, 'LineWidth', 1.2);
    title('MRC-PCA 相位主成分 (SG 濾波 + Z-score)');
    legend('PC1', 'PC2', 'PC3', 'Location', 'best');
    xlabel('時間 (秒)'); ylabel('Z-score'); grid on; axis tight;

    % --- 子圖5: 相位 PC1 解釋方差 ---
    subplot(3, 2, 6);
    var_ratio_phase = latent_phase(1:min(5,end)) / total_var_phase * 100;
    bar(var_ratio_phase, 'FaceColor', [0.8 0.4 0.2]);
    title('相位 PCA 各主成分解釋方差 (%)');
    xlabel('主成分'); ylabel('解釋方差 (%)'); grid on;
%     xticklabels(arrayfun(@(x) sprintf('PC%d', x), 1:length(var_ratio_phase), ...
%                 'UniformOutput', false));

%     sgtitle('MRC-PCA 架構：CSI 呼吸訊號提取完整流程');

    fprintf('[完成] MRC-PCA 處理完畢。輸出 PC1 (振幅/相位) 各為 [%d×1] 向量。\n', N);
end