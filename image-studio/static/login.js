/* 登录页：先查询是否需要首次设置，再在设置 / 登录两个表单之间切换。 */
(function(){
  'use strict';
  const $ = id => document.getElementById(id);

  // 跟随主界面的主题偏好
  let saved = 'dark';
  try{ saved = localStorage.getItem('image-fantasy-theme') || 'dark'; }catch(_){}
  document.documentElement.setAttribute('data-theme', saved === 'light' ? 'light' : 'dark');

  const view = id => {
    ['loading', 'setupForm', 'loginForm'].forEach(v => $(v).classList.toggle('hide', v !== id));
  };
  const showError = (el, msg) => {
    $(el).textContent = msg || '';
    $(el).classList.toggle('hide', !msg);
  };
  const busy = (btn, on, text) => {
    btn.disabled = on;
    if(on){
      btn.dataset.label = btn.textContent;
      btn.textContent = text;
    }else if(btn.dataset.label){
      btn.textContent = btn.dataset.label;
    }
  };

  async function post(path, body){
    let r, data;
    try{
      r = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        credentials: 'same-origin'
      });
      data = await r.json().catch(() => ({}));
    }catch(_){
      throw new Error('无法连接服务器，请检查网络后重试');
    }
    if(!r.ok) throw new Error((data && data.error) || ('请求失败：HTTP ' + r.status));
    return data;
  }

  function goApp(){
    // 只把应用自己的视图锚点（#gen / #config）带回去，其他内容一律丢弃
    const view = /^#(gen|config)$/.test(location.hash || '') ? location.hash : '';
    location.replace('/' + view);
  }

  async function init(){
    let status;
    try{
      const r = await fetch('/api/auth/status', { credentials: 'same-origin', cache: 'no-store' });
      status = await r.json();
    }catch(_){
      $('loading').textContent = '无法连接服务器，请稍后刷新重试。';
      return;
    }
    if(status.logged_in){
      goApp();
      return;
    }
    if(!status.configured){
      view('setupForm');
      // 支持把令牌放在链接里：/login#setup=xxxx，方便从服务器日志直接复制链接
      const m = /(?:^|[#&])setup=([^&]+)/.exec(location.hash || '');
      if(m){
        $('setupToken').value = decodeURIComponent(m[1]);
        try{ history.replaceState(null, '', location.pathname); }catch(_){}
        $('setupUser').focus();
      }else{
        $('setupToken').focus();
      }
    }else{
      view('loginForm');
      $('loginUser').focus();
    }
  }

  $('setupForm').addEventListener('submit', async e => {
    e.preventDefault();
    showError('setupError', '');
    const setup_token = $('setupToken').value.trim();
    const username = $('setupUser').value.trim();
    const password = $('setupPass').value;
    if(!setup_token) return showError('setupError', '请填写设置令牌');
    if(!username) return showError('setupError', '请填写账号');
    if(password !== $('setupPass2').value) return showError('setupError', '两次输入的密码不一致');
    const btn = $('setupBtn');
    busy(btn, true, '正在设置…');
    try{
      await post('/api/auth/setup', { setup_token, username, password });
      goApp();
    }catch(err){
      showError('setupError', err.message);
      busy(btn, false);
    }
  });

  const login = async () => {
    showError('loginError', '');
    const username = $('loginUser').value.trim();
    const password = $('loginPass').value;
    if(!username || !password) return showError('loginError', '请填写账号和密码');
    const btn = $('loginBtn');
    busy(btn, true, '正在登录…');
    try{
      await post('/api/auth/login', { username, password });
      goApp();
    }catch(err){
      showError('loginError', err.message);
      busy(btn, false);
      $('loginPass').select();
    }
  };
  $('loginForm').addEventListener('submit', e => { e.preventDefault(); login(); });

  init();
})();
