const toggleBtn = document.getElementById('toggleBtn');
const title = document.getElementById('title');
const submitBtn = document.getElementById('submitBtn');
const actionInput = document.getElementById('action');
const desc = document.getElementById('desc');
const switchText = document.getElementById('switchText');

// 處理 Google 回傳
function handleCredentialResponse(response) {
    if (!response.credential) {
        document.getElementById('message').innerText = "Google 登入授權失敗";
        document.getElementById('message').style.color = "#ef4444";
        return;
    }
    // 若後端尚未實作 JWT 解碼，可先提示或以 payload 傳送
    handleLogin({ action: 'google', credential: response.credential });
}

// 完美連動：切換 登入 / 註冊 模式與文字
toggleBtn.onclick = () => {
    // 清空上一次的錯誤訊息
    document.getElementById('message').innerText = "";
    
    if(actionInput.value === 'login') {
        title.innerText = "Create Account";
        desc.innerText = "Sign up with your email to get started";
        submitBtn.innerText = "Register";
        actionInput.value = "register";
        switchText.innerText = "Already have an account?";
        toggleBtn.innerText = "Back to Login";
    } else {
        title.innerText = "Login";
        desc.innerText = "Enter your email to access your dashboard";
        submitBtn.innerText = "Login";
        actionInput.value = "login";
        switchText.innerText = "Don't have an account?";
        toggleBtn.innerText = "Create one";
    }
};

// 監聽表單送出
document.getElementById('loginForm').onsubmit = function(e) {
    e.preventDefault();
    
    // ✨【精準抓取】完美對齊你的 HTML name 屬性
    const emailInput = this.querySelector('input[name="username"]');
    const passwordInput = this.querySelector('input[name="password"]');
    
    // 清空舊的錯誤提示，顯示正在連線中
    document.getElementById('message').innerText = "連線中...";
    document.getElementById('message').style.color = "#3b82f6"; // 藍色提示
    
    const payload = {
        action: actionInput.value,
        email: emailInput ? emailInput.value.trim() : '',
        password: passwordInput ? passwordInput.value : ''
    };
    
    handleLogin(payload);
};

function handleLogin(payload) {
    // 防止重複點擊
    if (submitBtn) submitBtn.disabled = true;

    // 發射 JSON 到 auth.php
    fetch('auth.php', { 
        method: 'POST', 
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload) 
    })
    .then(r => {
        // 如果後端死機（例如 500 錯誤），直接抓出來
        if (!r.ok) throw new Error("後端伺服器回應異常 (" + r.status + ")");
        return r.json();
    })
    .then(data => {
        if (submitBtn) submitBtn.disabled = false;

        if(data.status === 'success') { 
            document.getElementById('message').innerText = "驗證成功，正在跳轉...";
            document.getElementById('message').style.color = "#10b981"; // 綠色
            // 稍作停頓後閃現進儀表板
            setTimeout(() => {
                window.location.href = 'dashboard.php';
            }, 800);
        } else {
            // 真正把 Aiven 傳回來的錯誤（如：此 Email 已被註冊）吐在畫面的紅色格子裡！
            document.getElementById('message').innerText = data.message || "帳號或密碼驗證失敗";
            document.getElementById('message').style.color = "#ef4444"; // 紅色
        }
    })
    .catch(err => {
        if (submitBtn) submitBtn.disabled = false;
        console.error("連線發生錯誤:", err);
        document.getElementById('message').innerText = "伺服器連線中斷或逾時，請稍後重試";
        document.getElementById('message').style.color = "#ef4444";
    });
}