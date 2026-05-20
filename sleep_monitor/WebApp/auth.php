<?php
session_start();
header('Content-Type: application/json');
$db_path = '../sleep.db';

try {
    $db = new PDO("sqlite:$db_path");
    $db->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);

    $username = $_POST['username'] ?? '';
    $password = $_POST['password'] ?? ''; // 接收前端傳來的密碼
    $action = $_POST['action'] ?? 'login';

    if (empty($username) || empty($password)) {
        echo json_encode(['success' => false, 'message' => '帳號或密碼不能為空']);
        exit();
    }

    // 查詢使用者
    $stmt = $db->prepare("SELECT * FROM users WHERE username = ?");
    $stmt->execute([$username]);
    $user = $stmt->fetch(PDO::FETCH_ASSOC);

    if ($action === 'login') {
        if ($user) {
            // 密碼驗證】：比對密碼是否與資料庫中的 Hash 值吻合
            if (password_verify($password, $user['password'])) {
                $_SESSION['user_id'] = $user['id'];
                $_SESSION['email'] = $user['username'];
                echo json_encode(['success' => true]);
            } else {
                echo json_encode(['success' => false, 'message' => '密碼錯誤，請再試一次']);
            }
        } else {
            echo json_encode(['success' => false, 'message' => '帳號不存在，請先註冊']);
        }
    } else {
        // 註冊邏輯
        if ($user) {
            echo json_encode(['success' => false, 'message' => '此 Email 已被註冊']);
        } else {
            // 密碼加密】：絕對不能直接存明文！使用 BCRYPT 安全加密
            $hashed_password = password_hash($password, PASSWORD_BCRYPT);

            $ins = $db->prepare("INSERT INTO users (username, password, display_name) VALUES (?, ?, 'New User')");
            $ins->execute([$username, $hashed_password]);
            
            $_SESSION['user_id'] = $db->lastInsertId();
            $_SESSION['email'] = $username;
            
            echo json_encode(['success' => true]);
        }
    }
} catch (Exception $e) { 
    echo json_encode(['success' => false, 'message' => 'DB Error: ' . $e->getMessage()]); 
}
?>