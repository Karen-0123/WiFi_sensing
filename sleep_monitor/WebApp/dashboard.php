<?php
session_start();

$email = $_SESSION['email'] ?? 'Guest'; 
$score = null; 

if (!isset($_SESSION['user_id'])) {
    header("Location: login.html");
    exit();
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
    
    if ($data) {
        $score = floatval($data['sleep_score']);
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
    <link rel="stylesheet" href="style.css">
</head>
<body class="pg-dashboard flex-center-body">
    <div class="main-card card-wide">
        <div class="device-status">
            <div class="status-dot"></div>Device: Online
        </div>

        <div class="user-info-box">
            <span class="user-account"><?php echo htmlspecialchars($email); ?></span>
            <a href="login.html" class="logout-link">Logout</a>
        </div>

        <?php if ($score !== null): ?>
            <h2 class="title">睡眠分數</h2>
            <div class="circle-container">
                <svg class="circle-svg">
                    <circle class="circle-bg" cx="140" cy="140" r="120"></circle>
                    <circle class="circle-progress" cx="140" cy="140" r="120" 
                            style="stroke-dashoffset:<?php echo 754 - (754 * $score / 10); ?>;"></circle>
                </svg>
                <div class="score-num"><?php echo $score; ?></div>
            </div>
            <div class="score-label">Sleep Quality Score</div>
            <p class="score-desc">滿分 10 分 | 您的睡眠品質評估。</p>
            
            <button class="details-btn" onclick="window.location.href='details.php'">View More Details</button>

        <?php else: ?>
            <div style="padding: 80px 0;">
                <h1 style="font-size:3.5rem; margin:0;">Welcome!</h1>
                <p style="font-size:1.2rem; color:#666; margin-top:15px;">目前尚無睡眠數據</p>
                <p style="color:#aaa;">系統正在等待 Wi-Fi 訊號傳輸...<br>請開始監測以獲取您的第一份睡眠報告。</p>
            </div>
        <?php endif; ?>
    </div>
</body>
</html>