function featureTable = export_sleep_features(bpm_deviation, var_history, num_events, csv_filename, seg_start_times, seg_end_times, wake_sleep)
% EXPORT_SLEEP_FEATURES 將睡眠特徵整合為 MATLAB Table 並匯出為乾淨的 CSV 檔

    % ---------------------------------------------------------------------
    % 1. 預設參數處理 (Default Arguments)
    % ---------------------------------------------------------------------
    if nargin < 4 || isempty(csv_filename)
        csv_filename = 'sleep_stage_features.csv';
    end

    % ---------------------------------------------------------------------
    % 2. 向量轉型與維度對齊 (Dimension Alignment)
    % ---------------------------------------------------------------------
    bpm_deviation = bpm_deviation(:); % 確保為 [N, 1] 欄向量
    var_history   = var_history(:);   % 確保為 [N, 1] 欄向量
    num_segments  = length(bpm_deviation); % 取得總片段數

    % 處理 num_events (若為 cell 則轉成數值欄向量)
    if iscell(num_events)
        num_events_mat = zeros(num_segments, 1);
        for idx = 1:length(num_events)
            if ~isempty(num_events{idx})
                num_events_mat(idx) = num_events{idx};
            end
        end
        num_events = num_events_mat;
    end
    num_events = num_events(:);

    % 檢查變異度特徵數量是否對齊
    if length(var_history) ~= num_segments
        error('特徵數量不一致：var_history (%d 筆) 與 bpm_deviation (%d 筆) 長度不同！', ...
              length(var_history), num_segments);
    end
    
    % 處理 wake_sleep 判定欄位
    if nargin < 7 || isempty(wake_sleep)
        Wake_Sleep = zeros(num_segments, 1);
    else
        if iscell(wake_sleep)
            wake_sleep_mat = zeros(num_segments, 1);
            for idx = 1:length(wake_sleep)
                if ~isempty(wake_sleep{idx})
                    wake_sleep_mat(idx) = wake_sleep{idx};
                end
            end
            Wake_Sleep = wake_sleep_mat;
        else
            Wake_Sleep = wake_sleep(:);
        end
    end

    % ---------------------------------------------------------------------
    % 3. 計算 3 分鐘時間區間 (Timestamp Generation)
    % ---------------------------------------------------------------------
    timeFormat = 'yyyy-mm-dd HH:MM:ss';
    
    if iscell(seg_start_times)
        Start_Time = cell(num_segments, 1);
        End_Time   = cell(num_segments, 1);
        for k = 1:num_segments
            Start_Time{k} = datestr(seg_start_times{k}, timeFormat);
            End_Time{k}   = datestr(seg_end_times{k}, timeFormat);
        end
    else
        t_start_dt = seg_start_times + (0:num_segments-1)' * minutes(3);
        t_end_dt   = t_start_dt + minutes(3);
        Start_Time = cellstr(datestr(t_start_dt, timeFormat));
        End_Time   = cellstr(datestr(t_end_dt, timeFormat));
    end

    % 預設睡眠階段欄位 (Sleep_Stage = 0)
    Sleep_Stage = zeros(num_segments, 1);

    % ---------------------------------------------------------------------
    % 4. 建立 Table 與匯出無 Index 欄位的 CSV
    % ---------------------------------------------------------------------
    featureTable = table(Start_Time, ...
                         End_Time, ...
                         Wake_Sleep, ...
                         bpm_deviation, ...
                         var_history, ...
                         num_events, ...
                         Sleep_Stage, ...
                         'VariableNames', {'Start_Time', ...
                                           'End_Time', ...
                                           'Wake_Sleep', ...
                                           'Breathing_Rate_Deviation', ...
                                           'Breathing_Rate_Variability', ...
                                           'Num_Events', ...
                                           'Sleep_Stage'});
    % 匯出 CSV
    try
        writetable(featureTable, csv_filename, 'WriteRowNames', false);
        fprintf(' [成功] 特徵已成功匯出 %d 筆資料至：%s\n', num_segments, csv_filename);
    catch ME
        if strcmp(ME.identifier, 'MATLAB:table:write:FileOpenError') || ...
           contains(ME.message, 'Permission denied')
            
            [filepath, name, ext] = fileparts(csv_filename);
            alt_filename = fullfile(filepath, sprintf('%s_%s%s', name, datestr(now, 'HHMMSS'), ext));
            
            warning('檔案 [%s] 被占用或唯讀，改寫入至替代檔案：%s', csv_filename, alt_filename);
            writetable(featureTable, alt_filename, 'WriteRowNames', false);
        else
            rethrow(ME);
        end
    end
end