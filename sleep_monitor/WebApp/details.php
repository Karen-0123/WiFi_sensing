<?php
session_start();
if (!isset($_SESSION['user_id'])) { header("Location: login.html"); exit(); }

$db_path = '../sleep.db';
$session_data = null;
$chart_logs = [];

try {
    $db = new PDO("sqlite:$db_path");
    $db->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);

    // 1. 抓取最新的結算總結
    $stmt = $db->prepare("SELECT * FROM sleep_summaries WHERE user_id = ? ORDER BY id DESC LIMIT 1");
    $stmt->execute([$_SESSION['user_id']]);
    $session_data = $stmt->fetch(PDO::FETCH_ASSOC);

    if ($session_data) {
        // 2. 抓取該 Session 所有的時序資料點 (用於折線圖)
        $log_stmt = $db->prepare("SELECT timestamp, respiration_rate, inferred_stage FROM respiration_logs WHERE session_id = ? ORDER BY timestamp ASC");
        $log_stmt->execute([$session_data['id']]);
        $chart_logs = $log_stmt->fetchAll(PDO::FETCH_ASSOC);
    }
} catch (Exception $e) { die("Error: " . $e->getMessage()); }
?>

<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <title>Sleep Analysis Report</title>
    <link rel="stylesheet" href="style.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .pg-details { background: #f8f9fb; padding: 40px 20px; }
        .report-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 25px; max-width: 1000px; margin: 0 auto; }
        .chart-card { background: white; border-radius: 20px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); }
        .full-width { grid-column: span 2; }
        .stat-value { font-size: 24px; font-weight: 800; color: #111; }
        .stat-label { color: #999; font-size: 13px; }
        .back-link { display: inline-block; margin-bottom: 20px; color: #666; font-weight: 600; text-decoration: none; }
    </style>
</head>
<body class="pg-details">

<div style="max-width: 1000px; margin: 0 auto;">
    <a href="dashboard.php" class="back-link">← Back to Dashboard</a>
    <h1 style="margin-bottom: 30px;">Sleep Analysis Report</h1>

    <div class="report-grid">
        <div class="chart-card">
            <h3>Sleep Stages 分佈</h3>
            <canvas id="stageChart"></canvas>
            <div style="display: flex; justify-content: space-around; margin-top: 20px; text-align: center;">
                <div><div class="stat-value"><?php echo $session_data['deep_sleep_minutes']; ?>m</div><div class="stat-label">Deep</div></div>
                <div><div class="stat-value"><?php echo $session_data['rem_sleep_minutes']; ?>m</div><div class="stat-label">REM</div></div>
                <div><div class="stat-value"><?php echo $session_data['light_sleep_minutes']; ?>m</div><div class="stat-label">Light</div></div>
            </div>
        </div>

        <div class="chart-card">
            <h3>呼吸率統計</h3>
            <div style="margin-top: 40px;">
                <p class="stat-label">平均呼吸率</p>
                <p class="stat-value" style="font-size: 48px; color: #00ff88;"><?php echo $session_data['avg_respiration_rate']; ?> <span style="font-size: 16px; color: #999;">BPM</span></p>
            </div>
            <hr style="border: 0; border-top: 1px solid #eee; margin: 30px 0;">
            <h3>AI 睡眠建議</h3>
            <p style="color: #666; font-size: 14px; line-height: 1.6;">
                <?php 
                    if($session_data['sleep_score'] >= 8) echo "您的睡眠品質優異，請繼續保持穩定的作息。";
                    else if($session_data['sleep_score'] >= 6) echo "睡眠品質尚可，建議睡前減少藍光曝露，這有助於增加深睡比例。";
                    else echo "睡眠品質欠佳，監測到呼吸起伏不穩，建議諮詢專業醫師。";
                ?>
            </p>
        </div>

        <div class="chart-card full-width">
            <h3>呼吸率趨勢 (Respiration Rate Timeline)</h3>
            <canvas id="lineChart" style="height: 300px;"></canvas>
        </div>
    </div>
</div>

<script>
    // 準備折線圖數據
    const logLabels = <?php echo json_encode(array_map(function($l){ return substr($l['timestamp'], 11, 5); }, $chart_logs)); ?>;
    const logData = <?php echo json_encode(array_map(function($l){ return $l['respiration_rate']; }, $chart_logs)); ?>;

    // 折線圖實作
    new Chart(document.getElementById('lineChart'), {
        type: 'line',
        data: {
            labels: logLabels,
            datasets: [{
                label: 'Respiration Rate (BPM)',
                data: logData,
                borderColor: '#00ff88',
                backgroundColor: 'rgba(0, 255, 136, 0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 4
            }]
        },
        options: {
            responsive: true,
            scales: { y: { min: 10, max: 20 } }
        }
    });

    // 圓餅圖實作
    new Chart(document.getElementById('stageChart'), {
        type: 'doughnut',
        data: {
            labels: ['Deep', 'REM', 'Light'],
            datasets: [{
                data: [
                    <?php echo $session_data['deep_sleep_minutes']; ?>, 
                    <?php echo $session_data['rem_sleep_minutes']; ?>, 
                    <?php echo $session_data['light_sleep_minutes']; ?>
                ],
                backgroundColor: ['#0a192f', '#4a90e2', '#00ff88'],
                borderWidth: 0
            }]
        },
        options: { cutout: '70%' }
    });
</script>

</body>
</html>