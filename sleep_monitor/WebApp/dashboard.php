<?php
date_default_timezone_set('Asia/Taipei');
session_start();

$email = $_SESSION['email'] ?? 'Guest'; 
$score_100 = null; 

if (!isset($_SESSION['user_id'])) {
    $_SESSION['user_id'] = 1; // 開發與除錯容錯
}

$host = 'mysql-46cb3ab-ntou-project.h.aivencloud.com';
$port = 21225;
$db_name = 'defaultdb';
$username_db = 'avnadmin';
$password_db = 'AVNS_kegvXqQywhPKN1Xr4Yp';

try {
    $dsn = "mysql:host=$host;port=$port;dbname=$db_name;charset=utf8mb4";
    $ca_cert_path = __DIR__ . '/ca.pem'; 

    $options = [
        PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
        PDO::MYSQL_ATTR_SSL_CA => $ca_cert_path, 
        PDO::MYSQL_ATTR_SSL_VERIFY_SERVER_CERT => false
    ];
    $db = new PDO($dsn, $username_db, $password_db, $options);
    $db->exec("SET NAMES utf8mb4");

    $stmt = $db->prepare("SELECT sleep_score FROM sleep_summaries WHERE user_id = ? ORDER BY id DESC LIMIT 1");
    $stmt->execute([$_SESSION['user_id']]);
    $data = $stmt->fetch(PDO::FETCH_ASSOC);
    
    if ($data && $data['sleep_score'] !== null) {
        $raw_val = floatval($data['sleep_score']);
        // 雙向相容：若資料庫為 10 分制 (<=10) 則自動轉換為 100 分制
        $score_100 = ($raw_val <= 10.0) ? round($raw_val * 10, 0) : round($raw_val, 0);
    }
} catch (Exception $e) {
    error_log("Aiven DB Error in dashboard.php: " . $e->getMessage());
}
?>
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <title>Sleep Dashboard</title>
    <!-- 引入 style.css 並加入時間戳防止瀏覽器快取舊樣式 -->
    <link rel="stylesheet" href="style.css?v=<?php echo time(); ?>">
</head>
<body class="pg-dashboard flex-center-body">
    <div class="main-card card-wide">
        <!-- 邊緣端硬體連線狀態（維持單純的狀態綠燈） -->
        <div class="device-status">
            <div class="status-dot"></div>System Online
        </div>

        <div class="user-info-box">
            <span class="user-account"><?php echo htmlspecialchars($email); ?></span>
            <a href="login.html" class="logout-link">Logout</a>
        </div>

        <?php if ($score_100 !== null): ?>
            <!-- 系統後端模組運作架構標記（單一專屬 Pipeline 膠囊） -->
            <div class="system-pipeline-badge">
                <span class="pipeline-dot"></span>
                Paramiko SSH/SFTP 遠端監測中 (Ubuntu ↔ Windows 演算法 Pipeline)
            </div>

            <h2 class="title">睡眠品質評分</h2>
            
            <div class="circle-container">
                <svg class="circle-svg">
                    <circle class="circle-bg" cx="140" cy="140" r="120"></circle>
                    <!-- 圓周長 = 2 * PI * 120 ≈ 754，依照 100 分制計算 offset -->
                    <circle class="circle-progress" cx="140" cy="140" r="120" 
                            style="stroke-dashoffset:<?php echo 754 - (754 * ($score_100 / 100)); ?>;"></circle>
                </svg>
                <div class="score-num"><?php echo $score_100; ?></div>
            </div>

            <div class="score-label">Sleep Quality Score</div>
            <p class="score-desc">滿分 100 分 | 基於 AASM 臨床權重與非接觸式 CSI 訊號解析。</p>
            
            <button class="details-btn" onclick="window.location.href='details.php'">View More Details</button>

        <?php else: ?>
            <div style="padding: 60px 0;">
                <div class="system-pipeline-badge">
                    <span class="pipeline-dot"></span>
                    SSH 連線就緒，等待 Ubuntu 端傳輸 .dat / .csv
                </div>
                <h1 style="font-size:3rem; margin:15px 0 0 0;">Welcome!</h1>
                <p style="font-size:1.1rem; color:#666; margin-top:15px;">目前尚無已完成的睡眠會話</p>
                <p style="color:#aaa; font-size:13px; line-height:1.6;">
                    遠端異質監控模組將在偵測到傳輸完整後自動調用 MATLAB 特徵擷取。<br>
                    請確認 Ubuntu CSI 收集程式已啟動。
                </p>
            </div>
        <?php endif; ?>
    </div>
</body>
</html>