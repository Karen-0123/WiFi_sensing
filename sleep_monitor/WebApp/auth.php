<?php
session_start();
header('Content-Type: application/json');

// 顯示所有 PHP 隱藏錯誤
error_reporting(E_ALL);
ini_set('display_errors', 0); 

$host = 'mysql-46cb3ab-ntou-project.h.aivencloud.com';
$port = 21225;
$db_name = 'defaultdb';
$username_db = 'avnadmin';
$password_db = 'AVNS_NiPQssShIbu0Shs-vYB';

try {
    // 確切分離 host 和 port
    $dsn = "mysql:host=$host;port=$port;dbname=$db_name;charset=utf8mb4";
    
    $options = [
        PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
        PDO::ATTR_TIMEOUT => 15, // 15秒超時限制
        PDO::MYSQL_ATTR_SSL_VERIFY_SERVER_CERT => false,
        PDO::MYSQL_ATTR_SSL_COMMAND => 'SET NAMES utf8mb4'
    ];
    
    $db = new PDO($dsn, $username_db, $password_db, $options);
} catch (PDOException $e) {
    // 這裡會把最底層的錯誤（例如：Connection refused 或是 Permission denied）直接印給前端看！
    echo json_encode(["status" => "error", "message" => "Aiven 拒絕網頁連線！詳細原因: " . $e->getMessage()]);
    exit();
}

$data = json_decode(file_get_contents("php://input"), true);
$action = $data['action'] ?? '';
$email = $data['email'] ?? '';
$password = $data['password'] ?? '';

if (empty($email) || empty($password)) {
    echo json_encode(["status" => "error", "message" => "請輸入帳號與密碼"]);
    exit();
}

if ($action === 'register') {
    $stmt = $db->prepare("SELECT id FROM users WHERE username = ?");
    $stmt->execute([$email]);
    if ($stmt->fetch()) {
        echo json_encode(["status" => "error", "message" => "此 Email 已被註冊"]);
        exit();
    }

    $hashed_password = password_hash($password, PASSWORD_BCRYPT);
    $stmt = $db->prepare("INSERT INTO users (username, password) VALUES (?, ?)");
    if ($stmt->execute([$email, $hashed_password])) {
        $_SESSION['user_id'] = $db->lastInsertId();
        $_SESSION['email'] = $email;
        echo json_encode(["status" => "success", "message" => "註冊成功"]);
    } else {
        echo json_encode(["status" => "error", "message" => "註冊失敗"]);
    }
    exit();
}

if ($action === 'login') {
    $stmt = $db->prepare("SELECT id, password FROM users WHERE username = ?");
    $stmt->execute([$email]);
    $user = $stmt->fetch(PDO::FETCH_ASSOC);

    if ($user && password_verify($password, $user['password'])) {
        $_SESSION['user_id'] = $user['id'];
        $_SESSION['email'] = $email;
        echo json_encode(["status" => "success", "message" => "登入成功"]);
    } else {
        echo json_encode(["status" => "error", "message" => "帳號或密碼錯誤"]);
    }
    exit();
}
?>