function featureTable = export_sleep_features(bpm_deviation, var_history, num_events, all_90th_percentile, varience, csv_filename, seg_start_times, seg_end_times, wake_sleep, extra_features)
% EXPORT_SLEEP_FEATURES 將原有 9 個睡眠特徵與《WiFi-Sleep》論文完整 13 維生理特徵整合並匯出為 CSV 檔
%
% 欄位結構 (共 20 欄：保留前 9 欄相容既有流程 + 後 11 欄補齊論文全部 13 項呼吸與體動特徵):
%   [既有 9 欄]
%   1. Start_Time                 - 區段起始時間 (yyyy-mm-dd HH:MM:ss)
%   2. End_Time                   - 區段結束時間 (yyyy-mm-dd HH:MM:ss)
%   3. Wake_Sleep                 - 清醒/睡眠初步判別 (1=Sleep, 0=Awake)
%   4. Breathing_Rate_Deviation   - 呼吸率基線偏差 |90th_BPM - NREM_Baseline|
%   5. Breathing_Rate_Variability - 去趨勢後之呼吸率變異度
%   6. Num_Events                 - 翻身/大型體動 (MBM) 事件次數
%   7. all_90th_percentile        - 區段動態 BPM 之第 90 百分位數
%   8. variance                   - 區段插值後 BPM 之原始變異數
%   9. Sleep_Stage                - 睡眠階段標籤欄位 (預設填 0)
%
%   [新增《WiFi-Sleep》論文特徵欄 (Section IV-D & Section VI-C2)]
%   10. Resp_Rate_Mean            - 區段平均呼吸率 (BPM)
%   11. Resp_Rate_Smoothed        - 整晚高階多項式去趨勢後之平滑呼吸率 (BPM)
%   12. Resp_Rate_Derivative      - 平滑呼吸率之一階導數 (BPM/min)
%   13. Resp_Rate_IQR             - 呼吸率四分位距 (Interquartile Range)
%   14. Resp_Depth_Variance       - 正規化呼吸深度變異度 (Respiration Depth Variance)
%   15. FIT_Mean                  - 吸氣時間佔比平均值 (Fractional Inspiratory Time Mean)
%   16. FIT_Variance              - 吸氣時間佔比變異數 (FIT Variance)
%   17. IE_Ratio_Mean             - 吸吐氣時間比平均值 (Inhale-to-Exhale Ratio Mean)
%   18. IE_Ratio_Variance         - 吸吐氣時間比變異數 (I/E Ratio Variance)
%   19. MBM_Frequency             - 大型體動/翻身高斯加權移動平均頻率 (events/min)
%   20. PLM_Frequency             - 週期性肢體微動高斯加權移動平均頻率 (events/min)

    % ---------------------------------------------------------------------
    % 1. 預設參數處理 (Default Arguments)
    % ---------------------------------------------------------------------
    if nargin < 6 || isempty(csv_filename)
        csv_filename = 'sleep_stage_features.csv';
    end

    % ---------------------------------------------------------------------
    % 2. 向量轉型與維度對齊 (Dimension Alignment)
    % ---------------------------------------------------------------------
    bpm_deviation       = bpm_deviation(:);
    var_history         = var_history(:);
    all_90th_percentile = all_90th_percentile(:);
    varience            = varience(:);
    num_segments        = length(bpm_deviation);

    % 處理 num_events (若為 cell 則轉成數值欄向量)
    num_events = to_numeric_col(num_events, num_segments, 0);

    if length(var_history) ~= num_segments
        error('特徵數量不一致：var_history (%d 筆) 與 bpm_deviation (%d 筆) 長度不同！', ...
              length(var_history), num_segments);
    end

    % 處理 wake_sleep 判定欄位 (第 9 個參數)
    if nargin < 9 || isempty(wake_sleep)
        Wake_Sleep = zeros(num_segments, 1);
    else
        Wake_Sleep = to_numeric_col(wake_sleep, num_segments, 0);
    end

    % ---------------------------------------------------------------------
    % 3. 計算 3 分鐘時間區間 (Timestamp Generation)
    % ---------------------------------------------------------------------
    timeFormat = 'yyyy-mm-dd HH:MM:ss';

    if nargin >= 8 && iscell(seg_start_times) && ~isempty(seg_start_times)
        Start_Time = cell(num_segments, 1);
        End_Time   = cell(num_segments, 1);
        for k = 1:num_segments
            if ~isempty(seg_start_times{k})
                Start_Time{k} = datestr(seg_start_times{k}, timeFormat);
            else
                Start_Time{k} = '';
            end
            if ~isempty(seg_end_times{k})
                End_Time{k}   = datestr(seg_end_times{k}, timeFormat);
            else
                End_Time{k}   = '';
            end
        end
    elseif nargin >= 7 && ~isempty(seg_start_times)
        t_start_dt = seg_start_times + (0:num_segments-1)' * minutes(3);
        t_end_dt   = t_start_dt + minutes(3);
        Start_Time = cellstr(datestr(t_start_dt, timeFormat));
        End_Time   = cellstr(datestr(t_end_dt, timeFormat));
    else
        Start_Time = repmat({''}, num_segments, 1);
        End_Time   = repmat({''}, num_segments, 1);
    end

    % 預設睡眠階段欄位 (Sleep_Stage = 0)
    Sleep_Stage = zeros(num_segments, 1);

    % ---------------------------------------------------------------------
    % 4. 解析《WiFi-Sleep》新增之 11 個特徵欄位 (若未傳入則安全填入 NaN/0)
    % ---------------------------------------------------------------------
    if nargin < 10 || isempty(extra_features)
        extra_features = struct();
    end

    Resp_Rate_Mean       = get_field_col(extra_features, 'Resp_Rate_Mean', num_segments, NaN);
    Resp_Rate_Smoothed   = get_field_col(extra_features, 'Resp_Rate_Smoothed', num_segments, NaN);
    Resp_Rate_Derivative = get_field_col(extra_features, 'Resp_Rate_Derivative', num_segments, NaN);
    Resp_Rate_IQR        = get_field_col(extra_features, 'Resp_Rate_IQR', num_segments, NaN);
    Resp_Depth_Variance  = get_field_col(extra_features, 'Resp_Depth_Variance', num_segments, NaN);
    FIT_Mean             = get_field_col(extra_features, 'FIT_Mean', num_segments, NaN);
    FIT_Variance         = get_field_col(extra_features, 'FIT_Variance', num_segments, NaN);
    IE_Ratio_Mean        = get_field_col(extra_features, 'IE_Ratio_Mean', num_segments, NaN);
    IE_Ratio_Variance    = get_field_col(extra_features, 'IE_Ratio_Variance', num_segments, NaN);
    MBM_Frequency        = get_field_col(extra_features, 'MBM_Frequency', num_segments, 0);
    PLM_Frequency        = get_field_col(extra_features, 'PLM_Frequency', num_segments, 0);

    % ---------------------------------------------------------------------
    % 5. 建立 Table 與匯出無 Index 欄位的 CSV
    % ---------------------------------------------------------------------
    featureTable = table(Start_Time, ...
                         End_Time, ...
                         Wake_Sleep, ...
                         bpm_deviation, ...
                         var_history, ...
                         num_events, ...
                         all_90th_percentile, ...
                         varience, ...
                         Sleep_Stage, ...
                         Resp_Rate_Mean, ...
                         Resp_Rate_Smoothed, ...
                         Resp_Rate_Derivative, ...
                         Resp_Rate_IQR, ...
                         Resp_Depth_Variance, ...
                         FIT_Mean, ...
                         FIT_Variance, ...
                         IE_Ratio_Mean, ...
                         IE_Ratio_Variance, ...
                         MBM_Frequency, ...
                         PLM_Frequency, ...
                         'VariableNames', {'Start_Time', ...
                                           'End_Time', ...
                                           'Wake_Sleep', ...
                                           'Breathing_Rate_Deviation', ...
                                           'Breathing_Rate_Variability', ...
                                           'Num_Events', ...
                                           'all_90th_percentile', ...
                                           'variance', ...
                                           'Sleep_Stage', ...
                                           'Resp_Rate_Mean', ...
                                           'Resp_Rate_Smoothed', ...
                                           'Resp_Rate_Derivative', ...
                                           'Resp_Rate_IQR', ...
                                           'Resp_Depth_Variance', ...
                                           'FIT_Mean', ...
                                           'FIT_Variance', ...
                                           'IE_Ratio_Mean', ...
                                           'IE_Ratio_Variance', ...
                                           'MBM_Frequency', ...
                                           'PLM_Frequency'});

    % 匯出 CSV
    try
        writetable(featureTable, csv_filename, 'WriteRowNames', false);
        fprintf(' [成功] 特徵已成功匯出 %d 筆資料 (共 %d 欄特徵) 至：%s\n', ...
                num_segments, width(featureTable), csv_filename);
    catch ME
        if strcmp(ME.identifier, 'MATLAB:table:write:FileOpenError') || ...
           ~isempty(strfind(ME.message, 'Permission denied'))

            [filepath, name, ext] = fileparts(csv_filename);
            alt_filename = fullfile(filepath, sprintf('%s_%s%s', name, datestr(now, 'HHMMSS'), ext));

            warning('檔案 [%s] 被占用或唯讀，改寫入至替代檔案：%s', csv_filename, alt_filename);
            writetable(featureTable, alt_filename, 'WriteRowNames', false);
        else
            rethrow(ME);
        end
    end
end

function col = to_numeric_col(val, N, default_val)
    col = default_val * ones(N, 1);
    if iscell(val)
        for idx = 1:min(N, length(val))
            if ~isempty(val{idx}) && isnumeric(val{idx})
                col(idx) = val{idx}(1);
            end
        end
    elseif isnumeric(val) || islogical(val)
        val = double(val(:));
        n_copy = min(N, length(val));
        col(1:n_copy) = val(1:n_copy);
    end
end

function col = get_field_col(s, field_name, N, default_val)
    if isstruct(s) && isfield(s, field_name)
        col = to_numeric_col(s.(field_name), N, default_val);
    else
        col = default_val * ones(N, 1);
    end
end
