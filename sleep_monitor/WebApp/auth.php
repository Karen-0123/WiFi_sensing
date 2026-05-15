<?php
session_start();
header('Content-Type: application/json');
$db_path = '../sleep.db';

try {
    $db = new PDO("sqlite:$db_path");
    $db->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);

    $username = $_POST['username'] ?? '';
    $action = $_POST['action'] ?? 'login';

    $stmt = $db->prepare("SELECT * FROM users WHERE username = ?");
    $stmt->execute([$username]);
    $user = $stmt->fetch(PDO::FETCH_ASSOC);

    if ($action === 'login') {
        if ($user) {
            $_SESSION['user_id'] = $user['id'];
            $_SESSION['email'] = $user['username'];
            echo json_encode(['success' => true]);
        } else {
            echo json_encode(['success' => false, 'message' => '帳號不存在，請先註冊']);
        }
    } else {
        if ($user) {
            echo json_encode(['success' => false, 'message' => '此 Email 已被註冊']);
        } else {
            $ins = $db->prepare("INSERT INTO users (username, display_name) VALUES (?, 'New User')");
            $ins->execute([$username]);
            $_SESSION['user_id'] = $db->lastInsertId();
            $_SESSION['email'] = $username;
            echo json_encode(['success' => true]);
        }
    }
} catch (Exception $e) { echo json_encode(['success' => false, 'message' => 'DB Error']); }