function featureTable = export_sleep_features(bpm_deviation, var_history, csv_filename, seg_start_times, seg_end_times, wake_sleep)
% EXPORT_SLEEP_FEATURES 將睡眠特徵整合為 MATLAB Table 並匯出為乾淨的 CSV 檔
%
% [輸入參數]
%   bpm_deviation       : 呼吸頻率偏差向量 (1D array)
%   var_history         : 呼吸變異度向量 (1D array)
%   csv_filename        : (選填) 匯出的 CSV 檔名，預設為 'sleep_stage_features.csv'
%   seg_start_times     : (選填) 開始時間 (datetime 物件)，預設為 2026-01-01 00:00:00
%   wake_sleep          : (選填) 清醒/睡眠判定向量 (0/1)，若未提供則預設全為 0
%
% [輸出參數]
%   featureTable  : 整合後的 MATLAB Table 物件

    % ---------------------------------------------------------------------
    % 1. 預設參數處理 (Default Arguments)
    % ---------------------------------------------------------------------
    if nargin < 3 || isempty(csv_filename)
        csv_filename = 'sleep_stage_features.csv';
    end

    % ---------------------------------------------------------------------
    % 2. 向量轉型與維度對齊 (Dimension Alignment)
    % ---------------------------------------------------------------------
    bpm_deviation = bpm_deviation(:); % 確保為 [N, 1] 欄向量
    var_history   = var_history(:);   % 確保為 [N, 1] 欄向量
    num_segments = length(bpm_deviation); % 取得總片段數

    % 檢查變異度特徵數量是否對齊
    if length(var_history) ~= num_segments
        error('特徵數量不一致：var_history (%d 筆) 與 bpm_deviation (%d 筆) 長度不同！', ...
              length(var_history), num_segments);
    end
    
    if nargin < 6 || isempty(wake_sleep)
        Wake_Sleep = zeros(num_segments, 1);
    else
        Wake_Sleep = wake_sleep(:);
    end

    % ---------------------------------------------------------------------
    % 3. 計算 3 分鐘時間區間 (3-minute Window Timestamp Generation)
    % ---------------------------------------------------------------------
    timeFormat = 'yyyy-mm-dd HH:MM:ss'; % 適合 Python pandas 讀取的格式
    
    if iscell(seg_start_times)
        % 若傳入的是包含每個區段時間的 cell 陣列
        Start_Time = cell(num_segments, 1);
        End_Time   = cell(num_segments, 1);
        for k = 1:num_segments
            Start_Time{k} = datestr(seg_start_times{k}, timeFormat);
            End_Time{k}   = datestr(seg_end_times{k}, timeFormat);
        end
    else
        % 若只傳入單一 startTime (相容舊呼叫方式)
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
    % 核心 6 欄位：Start_Time, End_Time, Wake_Sleep, Breathing_Rate_Deviation, Breathing_Rate_Variability, Sleep_Stage
    featureTable = table(Start_Time, ...
                         End_Time, ...
                         Wake_Sleep, ...
                         bpm_deviation, ...
                         var_history, ...
                         Sleep_Stage, ...
                         'VariableNames', {'Start_Time', ...
                                           'End_Time', ...
                                           'Wake_Sleep', ...
                                           'Breathing_Rate_Deviation', ...
                                           'Breathing_Rate_Variability', ...
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