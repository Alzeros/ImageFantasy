#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI 图像 API 全参数调用工具（本地单文件，零依赖，仅用 Python 标准库）
功能：
  - 填 API Key / Base URL（支持 OpenAI 兼容代理）/ 模型名
  - 全参数暴露：n, size, quality, style, background, output_format,
    output_compression, moderation, response_format
  - 生成(generate) 与 局部重绘(edit, 上传图 + 可选遮罩) 两种模式
  - 结果内联显示并支持下载
运行：  python3 openai-image-tool.py
访问：  http://127.0.0.1:8000
"""
import http.server
import socketserver
import json
import urllib.request
import urllib.error
import os

PORT = 8000

HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OpenAI 图像 API 调用器</title>
<style>
  :root{ --bg:#f6f7f9; --card:#fff; --line:#e3e6ea; --txt:#1f2329; --mut:#8a9099;
         --pri:#2f6bff; --pri2:#1d4ed8; --ok:#16a34a; --err:#dc2626; --soft:#eef2ff; }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--txt);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif}
  .wrap{max-width:980px;margin:0 auto;padding:20px}
  h1{font-size:20px;margin:0 0 4px}
  .sub{color:var(--mut);font-size:12px;margin-bottom:16px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:14px}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  .grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
  label{display:block;font-size:12px;color:var(--mut);margin-bottom:4px}
  input,select,textarea{width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:8px;
    background:#fff;color:var(--txt);font-size:13px;font-family:inherit}
  textarea{resize:vertical;min-height:74px}
  input:focus,select:focus,textarea:focus{outline:none;border-color:var(--pri);box-shadow:0 0 0 2px var(--soft)}
  .row{display:flex;align-items:center;gap:14px;margin:6px 0 2px}
  .row label{margin:0}
  .seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
  .seg button{border:0;background:#fff;padding:7px 16px;cursor:pointer;font-size:13px;color:var(--mut)}
  .seg button.on{background:var(--pri);color:#fff}
  .btn{margin-top:6px;background:var(--pri);color:#fff;border:0;border-radius:9px;padding:11px 18px;
    font-size:14px;cursor:pointer;font-weight:600}
  .btn:hover{background:var(--pri2)}
  .btn:disabled{opacity:.55;cursor:not-allowed}
  details{margin-top:6px}
  summary{cursor:pointer;color:var(--pri);font-size:13px;font-weight:600;user-select:none}
  .hint{font-size:11px;color:var(--mut);margin-top:6px}
  .res{margin-top:6px}
  .imgs{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px;margin-top:10px}
  .imgs figure{margin:0;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#fafbfc}
  .imgs img{width:100%;display:block}
  .imgs a{display:block;padding:7px 10px;font-size:12px;color:var(--pri);text-align:center;text-decoration:none;border-top:1px solid var(--line)}
  .imgs a:hover{background:var(--soft)}
  .msg{font-size:13px;padding:10px 12px;border-radius:8px;margin-top:10px;white-space:pre-wrap;word-break:break-word}
  .msg.err{background:#fef2f2;color:var(--err);border:1px solid #fecaca}
  .msg.ok{background:#f0fdf4;color:var(--ok);border:1px solid #bbf7d0}
  .hide{display:none}
  .file{font-size:12px;color:var(--mut)}
</style>
</head>
<body>
<div class="wrap">
  <h1>OpenAI 图像 API 调用器</h1>
  <div class="sub">本地运行 · 全参数开放 · 支持 OpenAI 兼容代理与 gpt-image-1 / gpt-image-2</div>

  <div class="card">
    <div class="grid">
      <div><label>API Key</label><input id="key" type="password" placeholder="sk-..."></div>
      <div><label>Base URL（代理填根地址，自动补 /v1）</label><input id="base" value="https://api.openai.com/v1"></div>
    </div>
    <div class="grid" style="margin-top:12px">
      <div><label>模型名（可手填，如 gpt-image-1 / gpt-image-2）</label><input id="model" value="gpt-image-1"></div>
      <div><label>模式</label>
        <div class="seg" id="modeSeg">
          <button data-m="generate" class="on">生成 generate</button>
          <button data-m="edit">局部重绘 edit</button>
        </div>
      </div>
    </div>
  </div>

  <div class="card">
    <label>提示词 Prompt</label>
    <textarea id="prompt" placeholder="描述你想要生成的画面..."></textarea>
    <div id="editBox" class="hide" style="margin-top:12px">
      <div class="grid">
        <div><label>原图 image（必填）</label><input id="imgFile" type="file" accept="image/*"></div>
        <div><label>遮罩 mask（可选，透明区域被重绘）</label><input id="maskFile" type="file" accept="image/*"></div>
      </div>
      <div class="file" id="editNote">edit 模式下 prompt 可省略；image / mask 以 base64 上传。</div>
    </div>
  </div>

  <div class="card">
    <details>
      <summary>展开全参数面板</summary>
      <div class="grid3" style="margin-top:12px">
        <div><label>n（1–10）</label><input id="p_n" type="number" min="1" max="10" value="1"></div>
        <div><label>size</label>
          <select id="p_size">
            <option value="1024x1024" selected>1024x1024</option>
            <option value="1536x1024">1536x1024</option>
            <option value="1024x1536">1024x1536</option>
            <option value="auto">auto</option>
          </select>
        </div>
        <div><label>quality</label>
          <select id="p_quality"><option value="">— 不传 —</option><option>auto</option><option>low</option><option>medium</option><option>high</option></select>
        </div>
        <div><label>style</label>
          <select id="p_style"><option value="">— 不传 —</option><option>vivid</option><option>natural</option></select>
        </div>
        <div><label>background</label>
          <select id="p_background"><option value="">— 不传 —</option><option>auto</option><option>transparent</option><option>opaque</option></select>
        </div>
        <div><label>output_format</label>
          <select id="p_format"><option value="">— 不传 —</option><option>png</option><option>jpeg</option><option>webp</option></select>
        </div>
        <div><label>output_compression（jpeg/webp：0–100）</label><input id="p_comp" type="number" min="0" max="100" placeholder="如 90"></div>
        <div><label>moderation</label>
          <select id="p_mod"><option value="">— 不传 —</option><option>auto</option><option>low</option></select>
        </div>
        <div><label>response_format</label>
          <select id="p_resp"><option value="">— 不传（默认 url）—</option><option value="url">url</option><option value="b64_json">b64_json</option></select>
        </div>
      </div>
      <div class="hint">说明：留空的下拉项表示不发送该参数（由 API 取默认）。size 的 auto 仅部分模型支持；如用 gpt-image-2 请按代理文档调整可选值。</div>
    </details>
  </div>

  <button class="btn" id="go">生成图像</button>
  <div class="res" id="res"></div>
</div>

<script>
let mode = "generate";
document.querySelectorAll('#modeSeg button').forEach(b=>{
  b.onclick=()=>{
    document.querySelectorAll('#modeSeg button').forEach(x=>x.classList.remove('on'));
    b.classList.add('on'); mode=b.dataset.m;
    document.getElementById('editBox').classList.toggle('hide', mode!=='edit');
    document.getElementById('go').textContent = mode==='edit' ? '重绘图像' : '生成图像';
  };
});

function fileToB64(file){
  return new Promise((res,rej)=>{
    const r=new FileReader();
    r.onload=()=>res(r.result);
    r.onerror=rej;
    r.readAsDataURL(file);
  });
}

function buildParams(){
  const p={};
  const n=document.getElementById('p_n').value;          if(n) p.n=parseInt(n,10);
  const size=document.getElementById('p_size').value;    if(size) p.size=size;
  const q=document.getElementById('p_quality').value;    if(q) p.quality=q;
  const s=document.getElementById('p_style').value;      if(s) p.style=s;
  const bg=document.getElementById('p_background').value;if(bg) p.background=bg;
  const f=document.getElementById('p_format').value;     if(f) p.output_format=f;
  const c=document.getElementById('p_comp').value;       if(c!=='') p.output_compression=parseInt(c,10);
  const m=document.getElementById('p_mod').value;        if(m) p.moderation=m;
  const rf=document.getElementById('p_resp').value;      if(rf) p.response_format=rf;
  return p;
}

async function go(){
  const btn=document.getElementById('go'); btn.disabled=true; btn.textContent='请求中…';
  const res=document.getElementById('res'); res.innerHTML='';
  const payload={
    key:document.getElementById('key').value.trim(),
    base:document.getElementById('base').value.trim().replace(/\\/+$/,''),
    model:document.getElementById('model').value.trim()||'gpt-image-1',
    mode,
    prompt:document.getElementById('prompt').value,
    params:buildParams()
  };
  if(mode==='edit'){
    const img=document.getElementById('imgFile').files[0];
    if(!img){ res.innerHTML='<div class="msg err">edit 模式需要上传原图。</div>'; btn.disabled=false; btn.textContent='重绘图像'; return; }
    payload.image=await fileToB64(img);
    const mask=document.getElementById('maskFile').files[0];
    if(mask) payload.mask=await fileToB64(mask);
  }
  try{
    const r=await fetch('/api/call',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const d=await r.json();
    if(!r.ok){ const e=(d&&(d.error||d.message))||('HTTP '+r.status); res.innerHTML='<div class="msg err">'+escapeHtml(typeof e==='string'?e:JSON.stringify(e,null,2))+'</div>'; return; }
    render(d);
  }catch(err){
    res.innerHTML='<div class="msg err">请求失败：'+escapeHtml(err.message)+'</div>';
  }finally{
    btn.disabled=false; btn.textContent = mode==='edit' ? '重绘图像' : '生成图像';
  }
}

function render(d){
  const res=document.getElementById('res');
  const list=(d&&d.data)||[];
  if(!list.length){ res.innerHTML='<div class="msg err">返回为空：'+escapeHtml(JSON.stringify(d,null,2))+'</div>'; return; }
  let html='<div class="msg ok">成功生成 '+list.length+' 张图像。</div><div class="imgs">';
  list.forEach((it,i)=>{
    let src=it.b64_json ? ('data:image/png;base64,'+it.b64_json) : it.url;
    let dl = it.b64_json ? ('data:image/png;base64,'+it.b64_json) : it.url;
    let name = 'image_'+(i+1)+ (it.b64_json?'.png':'.img');
    html+='<figure><img src="'+src+'" alt="img'+(i+1)+'"><a href="'+dl+'" download="'+name+'">下载</a></figure>';
  });
  html+='</div>';
  res.innerHTML=html;
}
function escapeHtml(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
document.getElementById('go').onclick=go;
</script>
</body>
</html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, code, obj, ctype="application/json"):
        data = json.dumps(obj).encode("utf-8") if isinstance(obj, (dict, list)) else obj
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/") and ("?" not in self.path):
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/call":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            data = json.loads(raw)
        except Exception as e:
            self._send(400, {"error": "请求解析失败: " + str(e)})
            return

        key = (data.get("key") or "").strip()
        base = (data.get("base") or "https://api.openai.com/v1").strip().rstrip("/")
        model = data.get("model") or "gpt-image-1"
        mode = data.get("mode") or "generate"
        prompt = data.get("prompt") or ""
        params = data.get("params") or {}

        if not key:
            self._send(400, {"error": "缺少 API Key"})
            return

        payload = {"model": model, "prompt": prompt}
        for k, v in params.items():
            if v is None or v == "":
                continue
            payload[k] = v
        if mode == "edit":
            if data.get("image"):
                payload["image"] = data["image"]
            if data.get("mask"):
                payload["mask"] = data["mask"]

        endpoint = base + ("/images/edits" if mode == "edit" else "/images/generations")
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint, data=body,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                out = resp.read().decode("utf-8")
            self._send(200, json.loads(out))
        except urllib.error.HTTPError as e:
            try:
                err = e.read().decode("utf-8")
                err_obj = json.loads(err)
            except Exception:
                err_obj = {"error": err or ("HTTP " + str(e.code))}
            self._send(e.code, err_obj)
        except Exception as e:
            self._send(500, {"error": str(e)})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        print("OpenAI 图像调用工具已启动 ->  http://127.0.0.1:%d" % PORT)
        print("按 Ctrl+C 退出。")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n已停止。")
