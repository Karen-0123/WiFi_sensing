function [state, features] = detect_sleep_state(events, total_time, varargin)
% detect_sleep_state
%
% 根據翻身偵測結果判斷整段資料為 Sleep 或 Awake
%
% 輸入:
%   events      - detect_rollover() 輸出的 events
%   total_time  - 此段資料總長度(秒)，例如180
%
% 可選參數:
%   'MotionThresh'  - Motion Ratio閾值 (預設0.08)
%   'EventThresh'   - 最大允許翻身次數 (預設3)
%
% 輸出:
%   state     - 'Sleep' 或 'Awake'
%   features  - 特徵結構體
%
% MATLAB R2015b Compatible

%% 1. 解析輸入參數

p = inputParser;

addRequired(p,'events');
addRequired(p,'total_time');

addParameter(p,'MotionThresh',0.08);
addParameter(p,'EventThresh',3);

parse(p,events,total_time,varargin{:});

motion_thresh = p.Results.MotionThresh;
event_thresh  = p.Results.EventThresh;

%% 2. 計算特徵

num_events = length(events);

motion_time = 0;

for i = 1:num_events
    motion_time = motion_time + events(i).duration;
end

motion_ratio = motion_time / total_time;

%% 3. 判斷 Sleep / Awake

if motion_ratio < motion_thresh && num_events < event_thresh
    state = 0;  %Sleep
else
    state = 1;  %Awake
end

%% 4. 回傳特徵

features = struct();

features.total_time   = total_time;
features.motion_time  = motion_time;
features.motion_ratio = motion_ratio;
features.num_events   = num_events;

%% 5. 顯示結果

fprintf('\n');
fprintf('========== Sleep State Report ==========\n');
fprintf('Total Time      : %.1f sec\n', total_time);
fprintf('Motion Time     : %.1f sec\n', motion_time);
fprintf('Motion Ratio    : %.3f\n', motion_ratio);
fprintf('Motion Events   : %d\n', num_events);
fprintf('State           : %d\n', state);
fprintf('========================================\n');

end