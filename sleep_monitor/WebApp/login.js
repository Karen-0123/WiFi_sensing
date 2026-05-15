const toggleBtn = document.getElementById('toggleBtn');
const title = document.getElementById('title');
const submitBtn = document.getElementById('submitBtn');
const actionInput = document.getElementById('action');

// 處理 Google 回傳
function handleCredentialResponse(response) {
    handleLogin(null, response.credential);
}

toggleBtn.onclick = () => {
    if(actionInput.value === 'login') {
        title.innerText = "Create Account";
        submitBtn.innerText = "Register";
        actionInput.value = "register";
        toggleBtn.innerText = "Back to Login";
    } else {
        title.innerText = "Login";
        submitBtn.innerText = "Login";
        actionInput.value = "login";
        toggleBtn.innerText = "Create one";
    }
};

document.getElementById('loginForm').onsubmit = function(e) {
    e.preventDefault();
    const formData = new FormData(this);
    formData.append('action', actionInput.value);
    handleLogin(formData);
};

function handleLogin(formData) {
    fetch('auth.php', { method: 'POST', body: formData })
    .then(r => r.json())
    .then(data => {
        if(data.success) window.location.href = 'dashboard.php';
        else document.getElementById('message').innerText = data.message;
    });
}