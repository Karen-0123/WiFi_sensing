<?php
session_start();
if (!isset($_SESSION['user_id'])) { header("Location: login.html"); exit(); }

$session_data = null;
$chart_logs = [];

// Aiven 雲端資料庫設定
$host = 'mysql-46cb3ab-ntou-project.h.aivencloud.com';
$port = 21225;
$db_name = 'defaultdb';
$username_db = 'avnadmin';
$password_db = 'AVNS_NiPQssShIbu0Shs-vYB';

try {
    Aiven 雲端資料庫
    $dsn = "mysql:host=$host;port=$port;dbname=$db_name;charset=utf8mb4";
    $options = [
        PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
        PDO::MYSQL_ATTR_SSL_VERIFY_SERVER_CERT => false, // 🛡️ 寬鬆安全憑證放行，防止環境阻擋
        PDO::MYSQL_ATTR_SSL_COMMAND => 'SET NAMES utf8mb4'
    ];
    $db = new PDO($dsn, $username_db, $password_db, $options);

    // 1. 抓取最新的 Session 數據（由原本的用戶 ID 查詢）
    $stmt = $db->prepare("SELECT * FROM sleep_summaries WHERE user_id = ? ORDER BY id DESC LIMIT 1");
    $stmt->execute([$_SESSION['user_id']]);
    $session_data = $stmt->fetch(PDO::FETCH_ASSOC);

    if ($session_data) {
        // 2. 抓取該 Session 所有的呼吸率時序資料 (用於時序折線圖)
        $log_stmt = $db->prepare("SELECT timestamp, respiration_rate FROM respiration_logs WHERE session_id = ? ORDER BY timestamp ASC");
        $log_stmt->execute([$session_data['id']]);
        $chart_logs = $log_stmt->fetchAll(PDO::FETCH_ASSOC);
    }
} catch (Exception $e) { 
    die("Error: " . $e->getMessage()); 
}
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
        .stat-value { font-size: 20px; font-weight: 800; color: #111; }
        .stat-label { color: #999; font-size: 12px; margin-top: 5px; }
        .back-link { display: inline-block; margin-bottom: 20px; color: #666; font-weight: 600; text-decoration: none; }
        .ai-tag { display: inline-block; background: rgba(0, 255, 136, 0.1); color: #00cc6a; padding: 4px 12px; border-radius: 50px; font-size: 12px; font-weight: 700; margin-bottom: 15px; }
    </style>
</head>
<body class="pg-details">

<div style="max-width: 1000px; margin: 0 auto;">
    <a href="dashboard.php" class="back-link">← Back to Dashboard</a>
    <h1>Sleep Analysis Report</h1>

    <?php if ($session_data): ?>
    <div class="report-grid">
        <div class="chart-card">
            <h3>Sleep Stages 分佈</h3>
            <canvas id="stageChart"></canvas>
            <div style="display: flex; justify-content: space-between; margin-top: 25px; text-align: center;">
                <div><div class="stat-value"><?php echo $session_data['awake_minutes'] ?? 0; ?>m</div><div class="stat-label">Awake</div></div>
                <div><div class="stat-value"><?php echo $session_data['rem_sleep_minutes'] ?? 0; ?>m</div><div class="stat-label">REM</div></div>
                <div><div class="stat-value"><?php echo $session_data['light_sleep_minutes'] ?? 0; ?>m</div><div class="stat-label">Core</div></div>
                <div><div class="stat-value"><?php echo $session_data['deep_sleep_minutes'] ?? 0; ?>m</div><div class="stat-label">Deep</div></div>
            </div>
        </div>

        <div class="chart-card">
            <h3>呼吸率統計</h3>
            <div style="margin: 20px 0;">
                <p class="stat-label">平均呼吸率</p>
                <p style="font-size: 40px; font-weight: 800; color: #00ff88; margin: 5px 0;"><?php echo round($session_data['avg_respiration_rate'], 1); ?> <span style="font-size: 16px; color: #999;">BPM</span></p>
            </div>
            
            <hr style="border: 0; border-top: 1px solid #eee; margin: 20px 0;">
            
            <div class="ai-tag">睡眠建議</div>
            <p style="color: #444; font-size: 14px; line-height: 1.7; text-align: justify;">
                <?php 
                    $score = floatval($session_data['sleep_score']);
                    $avg_rr = floatval($session_data['avg_respiration_rate']);
                    $awake_min = intval($session_data['awake_minutes'] ?? 0);

                    if ($score >= 8.5) {
                        echo "<b>【完美落地】</b>您的 Wi-Fi CSI 睡眠監測表現堪稱極佳！深睡與核心睡眠區間分佈非常健康，代表大腦與大肌肉群在昨晚得到了深度的修復與放鬆。請繼續保持目前的規律作息。";
                    } else if ($score >= 6.5) {
                        echo "<b>【品質尚可】</b>您的睡眠品質處於標準區間。";
                        if ($awake_min >= 45) {
                            echo "監測到半夜 Awake（清醒狀態）累積達 {$awake_min} 分鐘，這可能降低了您的睡眠連續性。建議睡前 2 小時內減少水分攝取，並避免藍光曝露，這有助於優化睡眠效率與深睡比例。";
                        } else {
                            echo "整體結構穩定，打若想進一步提升白天的精神，建議可以將睡前環境溫度調低 1-2°C，並嘗試在固定的時間入睡，這能讓入睡速度與睡眠深度表現得更好。";
                        }
                    } else {
                        echo "<b>【恢復不足】</b>昨晚的睡眠總體分數偏低，身體可能尚未得到充足的休息。";
                    }

                    if ($avg_rr > 18.0) {
                        echo "<br><br><b>生理提示：</b>本次監測到您的平均呼吸率（" . round($avg_rr, 1) . " BPM）稍高於正常睡眠基準。這通常與睡前劇烈運動、壓力過大或環境悶熱、不通風有關。建議睡前進行 5-10 分鐘的腹式呼吸以安定副交感神經。";
                    } else if ($avg_rr < 10.0 && $avg_rr > 0) {
                        echo "<br><br><b>生理提示：</b>監測到您的中樞睡眠呼吸率（" . round($avg_rr, 1) . " BPM）偏低。請留意是否有晨起口乾、頭痛等現象。若持續偏低，建議諮詢專業醫師進行睡眠檢測。";
                    } else if ($score < 6.5 && $avg_rr >= 12.0 && $avg_rr <= 18.0) {
                        echo "<br><br><b>健康建議：</b>雖然總分數較低，但您的呼吸規律度（" . round($avg_rr, 1) . " BPM）十分平穩，屬於優質的基礎呼吸特徵。這通常代表睡眠環境本身很安全，您只需專注於『增加總睡眠時間』，避免熬夜即可大幅改善。";
                    }
                ?>
            </p>
        </div>

        <div class="chart-card full-width">
            <h3>呼吸率趨勢 (Respiration Rate Timeline)</h3>
            <canvas id="lineChart" style="height: 250px;"></canvas>
        </div>
    </div>
    <?php else: ?>
        <p>目前尚無完整的睡眠分析數據。</p>
    <?php endif; ?>
</div>

<script>
    // 時序資料轉換
    const logLabels = <?php echo json_encode(array_map(function($l){ return substr($l['timestamp'], 11, 5); }, $chart_logs)); ?>;
    const logData = <?php echo json_encode(array_map(function($l){ return $l['respiration_rate']; }, $chart_logs)); ?>;

    // 呼吸率折線圖
    new Chart(document.getElementById('lineChart'), {
        type: 'line',
        data: {
            labels: logLabels,
            datasets: [{ 
                label: 'Respiration Rate (BPM)', 
                data: logData, 
                borderColor: '#00ff88', 
                backgroundColor: 'rgba(0, 255, 136, 0.05)',
                fill: true,
                tension: 0.3 
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });

    // 圓餅圖 (四階段)
    new Chart(document.getElementById('stageChart'), {
        type: 'doughnut',
        data: {
            labels: ['Awake', 'REM', 'Core', 'Deep'],
            datasets: [{
                data: [
                    <?php echo $session_data['awake_minutes'] ?? 0; ?>,
                    <?php echo $session_data['rem_sleep_minutes'] ?? 0; ?>,
                    <?php echo $session_data['light_sleep_minutes'] ?? 0; ?>,
                    <?php echo $session_data['deep_sleep_minutes'] ?? 0; ?>
                ],
                backgroundColor: ['#ff6384', '#36a2eb', '#ffce56', '#4bc0c0']
            }]
        }
    });
</script>
</body>
</html>