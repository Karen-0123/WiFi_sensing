function [stage, features] = analyze_sleep_stage(breathing_rates)
    % ANALYZE_SLEEP_STAGE 基於 3 分鐘的呼吸頻率數據判斷睡眠階段。
    % 
    % 輸入:
    %   breathing_rates - 一維數值陣列，包含一段時間內的呼吸頻率 (bpm)
    % 輸出:
    %   stage - 字串，代表預測的睡眠階段
    %   features - 結構體 (struct)，包含統計特徵 mean, sd, cv

    % ==========================================
    % 定義生理學閾值常數 (避免 Magic Numbers)
    % ==========================================
    
    % 邊界與異常值閾值
    MIN_REQUIRED_SAMPLES = 3;      % 至少需要 3 個樣本
    APNEA_MEAN_RR_THRESHOLD = 5.0; % 平均 < 5 bpm 視為異常或呼吸中止
    NOISE_MEAN_RR_THRESHOLD = 30.0;% 平均 > 30 bpm 視為雜訊
    
    % 變異係數 (CV) 閾值
    CV_THRESHOLD_WAKE = 0.15;      % CV >= 15% 視為清醒
    CV_THRESHOLD_REM_N1 = 0.10;    % 10% <= CV < 15% 視為 REM 或 N1
    CV_THRESHOLD_DEEP = 0.05;      % CV <= 5% 視為 N3 深睡
    
    % 平均頻率 (Mean) 閾值 (bpm)
    MEAN_THRESHOLD_REM = 15.0;     % 中等變異下，平均 >= 15 視為 REM
    MEAN_THRESHOLD_DEEP = 14.0;    % 極低變異下，平均 <= 14 視為 N3

    % ==========================================
    % 演算法實作
    % ==========================================

    % 邊界處理 1：輸入數據不足
    if isempty(breathing_rates) || length(breathing_rates) < MIN_REQUIRED_SAMPLES
        stage = 'Insufficient Data';
        features = struct('mean', NaN, 'sd', NaN, 'cv', NaN);
        return;
    end
    
    % 確保輸入轉換為行向量 (Colum vector) 以利計算
    rates = breathing_rates(:);
    
    % 特徵提取
    mean_rr = mean(rates);
    
    % 邊界處理 2：避免 Mean 為 0 導致計算 CV 時發生除以零 (Divide by zero) 錯誤
    if mean_rr == 0
        stage = 'Anomaly (Zero Mean)';
        features = struct('mean', 0, 'sd', 0, 'cv', 0);
        return;
    end
    
    % 計算樣本標準差與變異係數
    sd_rr = std(rates); 
    cv_rr = sd_rr / mean_rr;
    
    % 將特徵存入 MATLAB 結構體 (Struct)
    features = struct('mean', round(mean_rr, 2), ...
                      'sd', round(sd_rr, 2), ...
                      'cv', round(cv_rr, 4));
                      
    % 邊界處理 3：病理狀態或雜訊 (如呼吸中止)
    if mean_rr < APNEA_MEAN_RR_THRESHOLD || mean_rr > NOISE_MEAN_RR_THRESHOLD
        stage = 'Anomaly / Apnea';
        return;
    end
    
    % 邏輯判斷：決策樹 (Decision Tree) 基於生理特徵
    if cv_rr >= CV_THRESHOLD_WAKE
        % 變異度最大 -> 清醒
        stage = 'Wake';
        
    elseif cv_rr >= CV_THRESHOLD_REM_N1
        % 變異度中高 -> REM 或 N1 
        if mean_rr >= MEAN_THRESHOLD_REM
            stage = 'REM'; % 頻率快且不規律
        else
            stage = 'N1';  % 頻率較低的過渡期
        end
        
    elseif cv_rr > CV_THRESHOLD_DEEP
        % 變異度低 (5% ~ 10%) -> 穩定的淺睡期
        stage = 'N2';
        
    else
        % 變異度極低 (<= 5%) -> 規律如節拍器
        if mean_rr <= MEAN_THRESHOLD_DEEP
            stage = 'N3'; % 頻率最低且最規律
        else
            stage = 'N2'; % 若頻率未達深睡標準，仍歸類為穩定的 N2
        end
    end
end