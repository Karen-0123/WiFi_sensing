<?php
session_start();
header('Content-Type: application/json');

// Aiven 雲端資料庫設定
$host = 'mysql-46cb3ab-ntou-project.h.aivencloud.com';
$port = 21225;
$db_name = 'defaultdb';
$username_db = 'avnadmin';
$password_db = 'AVNS_kegvXqQywhPKN1Xr4Yp'; 

try {
    // 最保險、最乾淨的雙引號直譯法，絕對不會發生連接點（.）漏掉的語法錯誤！
    $dsn = "mysql:host=$host;port=$port;dbname=$db_name;charset=utf8mb4";
    
    $ca_cert_path = __DIR__ . '/ca.pem'; 

    $options = [
        PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
        PDO::ATTR_TIMEOUT => 15,
        PDO::MYSQL_ATTR_SSL_CA => $ca_cert_path, 
        PDO::MYSQL_ATTR_SSL_VERIFY_SERVER_CERT => false
    ];
    
    $db = new PDO($dsn, $username_db, $password_db, $options);
    $db->exec("SET NAMES utf8mb4");

} catch (Exception $e) { 
    die(json_encode(['status' => 'error', 'message' => 'Aiven 連線失敗: ' . $e->getMessage()])); 
}

//  接收前端資料
$data = json_decode(file_get_contents("php://input"), true);
$username = $data['email'] ?? $data['username'] ?? $_POST['username'] ?? $_POST['email'] ?? '';
$password = $data['password'] ?? $_POST['password'] ?? '';
$action = $data['action'] ?? $_POST['action'] ?? 'login';

if (empty($username) || empty($password)) {
    echo json_encode(['status' => 'error', 'message' => '帳號或密碼不能為空']);
    exit();
}

try {
    $stmt = $db->prepare("SELECT * FROM users WHERE username = ?");
    $stmt->execute([$username]);
    $user = $stmt->fetch(PDO::FETCH_ASSOC);

    if ($action === 'login') {
        if ($user) {
            if (password_verify($password, $user['password'])) {
                $_SESSION['user_id'] = $user['id'];
                $_SESSION['email'] = $user['username'];
                echo json_encode(['status' => 'success', 'message' => '登入成功']);
            } else {
                echo json_encode(['status' => 'error', 'message' => '密碼錯誤，請再試一次']);
            }
        } else {
            echo json_encode(['status' => 'error', 'message' => '帳號不存在，請先註冊']);
        }
    } else {
        if ($user) {
            echo json_encode(['status' => 'error', 'message' => '此 Email 已被註冊']);
        } else {
            $hashed_password = password_hash($password, PASSWORD_BCRYPT);

            $ins = $db->prepare("INSERT INTO users (username, password, display_name) VALUES (?, ?, 'New User')");
            $ins->execute([$username, $hashed_password]);
            
            $_SESSION['user_id'] = $db->lastInsertId();
            $_SESSION['email'] = $username;
            
            echo json_encode(['status' => 'success', 'message' => '註冊成功']);
        }
    }
} catch (Exception $e) { 
    echo json_encode(['status' => 'error', 'message' => 'DB Error: ' . $e->getMessage()]); 
}
?>