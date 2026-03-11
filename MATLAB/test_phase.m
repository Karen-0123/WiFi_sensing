%% 讀取原始數據
str = 'C:\Users\fupei\Desktop\csi\data\breathe\breathe_200hz_003.dat'; 
csi_trace = read_bf_file(str);
csi_trace(all(cellfun(@isempty,csi_trace),2),:) = []; % 去掉空部分

row = size(csi_trace,1);

% 恢復為你的原始設計：361 欄 (1~180振幅, 181~360相位, 361時間戳)
result_matrix = zeros(row, 361); 
valid_idx = 1; % 新增：用來記錄「完整有效封包」的實際行數

Fs = 200; % 採樣頻率

%% 計算並儲存
for i = 1:row
    csi_entry = csi_trace{i};
 
    % 讀取 CSI 值
    csi = get_scaled_csi(csi_entry); 
    
    % ? 關鍵修改：檢查元素總數是否確實為 180 (3 * 2 * 30)
    % 如果因為網路波動導致天線數改變，就跳過這個不完整的封包
    if numel(csi) ~= 180
        continue; 
    end
 
    % 計算振幅與相位 (3D 矩陣直接計算即可)
    amplitude = abs(csi);
    phase = angle(csi);
 
    % 賦值 (使用 valid_idx 確保不會留下空行)
    result_matrix(valid_idx, 1:180) = reshape(amplitude, 1, []);
    result_matrix(valid_idx, 181:360) = reshape(phase, 1, []);
    result_matrix(valid_idx, 361) = csi_entry.timestamp_low;
    
    valid_idx = valid_idx + 1; % 成功寫入一筆，指標往下移
end
 
% ? 清除沒有用到的多餘預分配空間
result_matrix = result_matrix(1:valid_idx-1, :);

%% 視覺化
% 振幅圖
figure(1);
plot(result_matrix(:, 1:180));
xlabel('時間 (封包數)');
ylabel('振幅');
title('振幅隨時間的變化 (3x2 天線)');
grid on;
 
% 相位圖
figure(2);
plot(result_matrix(:, 181:360));
xlabel('時間 (封包數)');
ylabel('相位 (弧度)');
title('相位隨時間的變化 (3x2 天線)');
grid on;
 
% 相位解卷積 (Unwrap)
phase_wrapped = result_matrix(:, 181:360); 
phase_unwrapped = unwrap(phase_wrapped);

figure(3);
subplot(2, 1, 1);
plot(phase_wrapped);
title('Wrapped Phase');
subplot(2, 1, 2);
plot(phase_unwrapped);
title('Unwrapped Phase');