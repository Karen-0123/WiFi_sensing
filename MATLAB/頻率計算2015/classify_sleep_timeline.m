function [stage_timeline, time_axis_epochs] = classify_sleep_timeline(bpm_timeline, time_axis_bpm)
    % �z�L�ưʵ��� (Sliding Window) �i��s��ίv���q����
    % ��J:
    %   bpm_timeline   - �� calculate_dynamic_bpm ��X���ʺA�I�l�v�}�C
    %   time_axis_bpm  - �������ɶ��b (��)
    % ��X:
    %   stage_timeline   - �C�� Epoch ���ίv���q���G (Cell Array)
    %   time_axis_epochs - �C�ӧP�_�I�������ɶ������I (����)

    % ======================================================================
    % 1. �����ѼƳ]�w
    % �ھ��{�� AASM �зǡA�ڭ̨C 30 ����X�@�ӵ��G (Epoch)
    % �����F�έpí�w�ʡA�P�_�޿�|�ѦҹL�h 180 �� (3 ����) ���ƾ�
    % ======================================================================
    window_size_sec = 180;  % ���f����: 180 ��
    step_size_sec = 30;     % �B�i����: 30 ��
    fs_bpm = 1;             % ���] bpm_timeline ���ļ˲v�O�C�� 1 �I (�� step_size �M�w)

    % �p���`�ɶ��P�` Epoch �ƶq
    total_duration = time_axis_bpm(end) - time_axis_bpm(1);
    num_epochs = floor((total_duration - window_size_sec) / step_size_sec) + 1;

    if num_epochs <= 0
        error('����`���פ��� 180 ���A�L�k�i��ίv�������R�C');
    end

    % �w���t�m�O����
    stage_timeline = cell(1, num_epochs);
    time_axis_epochs = zeros(1, num_epochs);

    % ======================================================================
    % 2. ����ưʵ����j��
    % ======================================================================
    fprintf('�B�J 7: �}�l����s��ίv���� (���f: %ds, �B�i: %ds)...\n', window_size_sec, step_size_sec);

    for i = 1:num_epochs
        % �p����e���f���ɶ��d��
        t_start = (i-1) * step_size_sec;
        t_end = t_start + window_size_sec;
        
        % �O���� Epoch �����߮ɶ��I (�ഫ�������A��K�e��)
        time_axis_epochs(i) = (t_start + (window_size_sec / 2)) / 60;

        % ��X���b���e 3 �������f�����Ҧ� BPM �ƾگ���
        logical_idx = (time_axis_bpm >= t_start & time_axis_bpm <= t_end);
        current_window_data = bpm_timeline(logical_idx);

        % �I�s�֤ߧP�_���
        % �o�̷|�� 3 �������ƾڥ�i�h�A�^�Ǥ@�ӳ̥i�઺���q
        [stage, ~] = analyze_sleep_stage(current_window_data);
        
        % �x�s���G
        stage_timeline{i} = stage;
    end

    % ======================================================================
    % 3. ���G�έp��X
    % ======================================================================
    % ²��έp�U���q�X�{���
    unique_stages = unique(stage_timeline);
    fprintf('--- �ίv�����έp ---\n');
    for k = 1:length(unique_stages)
        count = sum(strcmp(stage_timeline, unique_stages{k}));
        fprintf('%s: %.1f%%\n', unique_stages{k}, (count/num_epochs)*100);
    end
    fprintf('�s��ίv���������I\n');
end