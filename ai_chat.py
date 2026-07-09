"""
Ꮇʏᴛʜɪᴄ ᴀɪ — single file, powered by Google's Gemini API or a local Ollama model.

Usage (Gemini — default, needs a free API key):
    1. pip install flask requests
    2. Set your API key:
         Mac/Linux:   export GEMINI_API_KEY="your-key-here"
         Windows:     set GEMINI_API_KEY=your-key-here
    3. python ai_chat.py
    4. Open http://localhost:5000 in your browser

Get a FREE API key (no credit card needed) at https://aistudio.google.com/apikey

Usage (Ollama — fully local, no API key or internet needed):
    1. Install Ollama from https://ollama.com and make sure it's running
       (`ollama serve`, or it may already be running as a background service)
    2. Pull a model, e.g.:  ollama pull llama3.1
    3. Set the provider:
         Mac/Linux:   export AI_PROVIDER=ollama
         Windows:     set AI_PROVIDER=ollama
    4. python ai_chat.py
    5. Open http://localhost:5000 in your browser
"""

import os
import json
import uuid
import time
import base64
import requests
from flask import (
    Flask, request, jsonify, Response, session,
    stream_with_context
)

PROVIDER = os.environ.get("AI_PROVIDER", "auto").strip().lower()

# --- API Keys ----------------------------------------------------------------
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY",    "")
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY",      "")
CEREBRAS_API_KEY  = os.environ.get("CEREBRAS_API_KEY",  "")
OPENROUTER_API_KEY= os.environ.get("OPENROUTER_API_KEY","")
HF_API_KEY        = os.environ.get("HF_API_KEY",        "")
NANO_BANANA_API_KEY = os.environ.get("NANO_BANANA_API_KEY", "")
NANO_BANANA_BASE     = "https://api.nanobananaapi.ai/api/v1/nanobanana"

# --- Model names -------------------------------------------------------------
GEMINI_MODEL      = "gemini-2.0-flash"
GROQ_MODEL        = os.environ.get("GROQ_MODEL",        "llama-3.1-8b-instant")
OPENROUTER_MODEL  = os.environ.get("OPENROUTER_MODEL",  "google/gemma-3-4b-it:free")
HF_MODEL          = os.environ.get("HF_MODEL",          "mistralai/Mistral-7B-Instruct-v0.3")
CEREBRAS_MODEL    = os.environ.get("CEREBRAS_MODEL",    "gpt-oss-120b")
OLLAMA_MODEL      = os.environ.get("OLLAMA_MODEL",       "llama3.1")
OLLAMA_URL        = os.environ.get("OLLAMA_URL",         "http://localhost:11434").rstrip("/")

GEMINI_STREAM_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:streamGenerateContent"
)

API_KEY = GEMINI_API_KEY
MODEL   = GEMINI_MODEL
SYSTEM_PROMPT = (
    "You are Ꮇʏᴛʜɪᴄ ᴀɪ, a smart and friendly AI assistant made by Aarav Singh. "
    "If asked who made you, say you are Ꮇʏᴛʜɪᴄ ᴀɪ made by Aarav Singh — say it once naturally, never repeat it unprompted. "
    "Never mention Google, Groq, OpenRouter, HuggingFace, Meta, Mistral, Anthropic, or any AI company as your creator or backend. "
    "You can help with anything: questions, writing, coding, math, ideas, or just chatting. "
    "When writing code, always wrap it in markdown code blocks with the language name. "
    "LANGUAGE: Always reply ENTIRELY in the same language the user's message is written in. "
    "ANTI-REPETITION RULES: NEVER restate the user, never use filler like 'Great question', be direct and natural."
)

GEMINI_SEARCH_ADDENDUM = (
    " WEB SEARCH: You have access to Google Search. When the user asks about current events, "
    "use the search tool to find the answer. Translate results into the reply language."
)

app = Flask(__name__)

def _persistent_secret_key():
    env_key = os.environ.get("FLASK_SECRET_KEY")
    if env_key: return env_key
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "chat_data")
    os.makedirs(data_dir, exist_ok=True)
    key_path = os.path.join(data_dir, "flask_secret.key")
    if os.path.exists(key_path):
        with open(key_path, "r") as f:
            existing = f.read().strip()
        if existing: return existing
    new_key = str(uuid.uuid4())
    with open(key_path, "w") as f:
        f.write(new_key)
    return new_key

app.secret_key = _persistent_secret_key()
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
_TEMP_IMAGES = {}
_TEMP_IMAGE_TTL_SECONDS = 30 * 60

def _store_temp_image(raw_bytes, mime_type):
    cutoff = time.time() - _TEMP_IMAGE_TTL_SECONDS
    for k in [k for k, v in _TEMP_IMAGES.items() if v["created"] < cutoff]:
        _TEMP_IMAGES.pop(k, None)
    img_id = uuid.uuid4().hex
    _TEMP_IMAGES[img_id] = {"data": raw_bytes, "mime_type": mime_type or "image/png", "created": time.time()}
    return img_id

def nano_banana_submit(prompt, image_urls=None, num_images=1):
    if not NANO_BANANA_API_KEY: return None, "NanoBanana API key not configured"
    payload = {"prompt": prompt, "type": "IMAGETOIAMGE" if image_urls else "TEXTTOIAMGE", "numImages": num_images}
    if image_urls: payload["imageUrls"] = image_urls
    try:
        resp = requests.post(f"{NANO_BANANA_BASE}/generate", headers={"Authorization": f"Bearer {NANO_BANANA_API_KEY}", "Content-Type": "application/json"}, json=payload, timeout=30)
        data = resp.json()
        if resp.status_code == 200 and data.get("code") == 200: return data["data"]["taskId"], None
        return None, data.get("msg") or f"NanoBanana error ({resp.status_code})"
    except Exception as e: return None, str(e)

def nano_banana_poll(task_id, max_wait=180, interval=3):
    headers = {"Authorization": f"Bearer {NANO_BANANA_API_KEY}"}
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            resp = requests.get(f"{NANO_BANANA_BASE}/record-info", params={"taskId": task_id}, headers=headers, timeout=15)
            data = resp.json()
        except: return None, "Connection error"
        flag = data.get("successFlag")
        if flag == 1:
            result = data.get("response") or {}
            url = result.get("resultImageUrl") or (result.get("resultImageUrls")[0] if result.get("resultImageUrls") else None)
            return url, None
        if flag in (2, 3): return None, "Generation failed"
        time.sleep(interval)
    return None, "Timed out"

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

def sb_headers():
    return {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": "application/json", "Prefer": "return=representation"}

def sb(path): return f"{SUPABASE_URL}/rest/v1/{path}"

def current_username():
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
        session.permanent = True
    return session["user_id"]

def login_required(view):
    def wrapped(*args, **kwargs):
        current_username()
        return view(*args, **kwargs)
    wrapped.__name__ = view.__name__
    return wrapped

# --- Storage Helpers ---
def list_conversations(username):
    if not SUPABASE_URL: return _list_conversations_file(username)
    try:
        r = requests.get(sb(f"conversations?username=eq.{username}&order=updated_at.desc&select=id,title,updated_at"), headers=sb_headers(), timeout=10)
        return r.json() if r.status_code == 200 else []
    except: return []

def load_conversation(username, conv_id):
    if not SUPABASE_URL: return _load_conversation_file(username, conv_id)
    try:
        r = requests.get(sb(f"conversations?id=eq.{conv_id}&username=eq.{username}"), headers=sb_headers(), timeout=10)
        if r.status_code == 200:
            rows = r.json()
            if rows:
                row = rows[0]
                return {"title": row["title"], "updated_at": row["updated_at"], "messages": row["messages"] if isinstance(row["messages"], list) else json.loads(row["messages"])}
    except: pass
    return None

def save_conversation(username, conv_id, data):
    data["updated_at"] = time.time()
    if not SUPABASE_URL: _save_conversation_file(username, conv_id, data); return
    try:
        requests.post(sb("conversations"), headers={**sb_headers(), "Prefer": "resolution=merge-duplicates,return=minimal"}, json={"id": conv_id, "username": username, "title": data.get("title", "New chat"), "updated_at": data["updated_at"], "messages": data.get("messages", [])}, timeout=15)
    except: pass

def delete_conversation(username, conv_id):
    if not SUPABASE_URL: _delete_conversation_file(username, conv_id); return
    try: requests.delete(sb(f"conversations?id=eq.{conv_id}&username=eq.{username}"), headers=sb_headers(), timeout=10)
    except: pass

# --- File Fallbacks ---
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_BASE_DIR, "chat_data")
os.makedirs(_DATA_DIR, exist_ok=True)

def _user_conv_dir(username):
    path = os.path.join(_DATA_DIR, "conversations", username)
    os.makedirs(path, exist_ok=True); return path

def _list_conversations_file(username):
    folder = _user_conv_dir(username); convs = []
    for fname in os.listdir(folder):
        if not fname.endswith(".json"): continue
        try:
            with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
                d = json.load(f)
                convs.append({"id": fname[:-5], "title": d.get("title", "New chat"), "updated_at": d.get("updated_at", 0)})
        except: continue
    convs.sort(key=lambda c: c["updated_at"], reverse=True); return convs

def _load_conversation_file(username, conv_id):
    path = os.path.join(_user_conv_dir(username), f"{conv_id}.json")
    if not os.path.exists(path): return None
    try:
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    except: return None

def _save_conversation_file(username, conv_id, data):
    with open(os.path.join(_user_conv_dir(username), f"{conv_id}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _delete_conversation_file(username, conv_id):
    path = os.path.join(_user_conv_dir(username), f"{conv_id}.json")
    if os.path.exists(path): os.remove(path)

def make_title(msg):
    t = (msg or "Attachment").strip().replace("\n", " ")
    return t[:40] + ("…" if len(t) > 40 else "")

# --- HTML / UI ---
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ꮇʏᴛʜɪᴄ ᴀɪ</title>
<style>
  :root {
    --bg:#1a1a1a; --panel:#2a2a2a; --border:#3a3a3a;
    --text:#ececec; --muted:#8e8ea0; --accent:#10a37f;
    --accent-dim:#1a3a30; --user-bubble:#2a2a2a; --user-text:#ececec;
    --ai-bubble:#1a1a1a; --sidebar-w:260px; --msg-font-size:14.5px;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  html,body { height:100%; background:var(--bg); color:var(--text); font-family:Inter,sans-serif; overflow:hidden; }
  .layout { display:flex; height:100vh; }
  body.theme-light {
    --bg:#f7f7f8; --panel:#ffffff; --border:#e3e3e6;
    --text:#1f1f1f; --muted:#6b6b76; --accent-dim:#e3f5ef;
    --user-bubble:#eef0f2; --user-text:#1f1f1f; --ai-bubble:#ffffff;
  }
  #sidebar { width:var(--sidebar-w); flex-shrink:0; background:var(--panel); border-right:1px solid var(--border); display:flex; flex-direction:column; transition:margin-left .2s ease; }
  #sidebar.hidden { margin-left:calc(-1 * var(--sidebar-w)); }
  #new-chat-btn { margin:12px; padding:10px; background:var(--accent); color:#fff; border:none; border-radius:8px; font-weight:600; cursor:pointer; }
  #conv-list { flex:1; overflow-y:auto; padding:0 8px; }
  .conv-item { display:flex; align-items:center; justify-content:space-between; padding:9px 10px; border-radius:7px; cursor:pointer; font-size:13px; color:var(--muted); }
  .conv-item:hover, .conv-item.active { background:var(--accent-dim); color:var(--accent); }
  .conv-item .title { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; flex:1; }
  .app { display:flex; flex-direction:column; height:100vh; flex:1; min-width:0; }
  header { padding:14px 20px; border-bottom:1px solid var(--border); display:flex; align-items:center; justify-content:space-between; background:var(--bg); z-index:20; }
  header .left, header .right { display:flex; align-items:center; gap:8px; }
  header button { 
    background:none; border:1px solid var(--border); color:var(--muted); 
    width:36px; height:36px; border-radius:6px; cursor:pointer; font-size:15px;
    display:flex; align-items:center; justify-content:center;
  }
  header button:hover { background:var(--panel); color:var(--text); }
  #fullscreen-btn.active { color:var(--accent); border-color:var(--accent); }
  #messages-wrap { flex:1; overflow-y:auto; position:relative; scroll-behavior: smooth; }
  #messages { padding:24px 20px; display:flex; flex-direction:column; gap:16px; max-width:760px; margin:0 auto; width:100%; }
  .msg { max-width:80%; padding:11px 15px; border-radius:18px; line-height:1.6; font-size:var(--msg-font-size); white-space:pre-wrap; }
  .msg.user { align-self:flex-end; background:var(--user-bubble); color:var(--user-text); border-bottom-right-radius:4px; }
  .msg.ai { align-self:flex-start; background:var(--ai-bubble); color:var(--text); border-bottom-left-radius:4px; }
  .input-area { padding:10px 20px 16px; border-top:1px solid var(--border); background:var(--bg); max-width:760px; margin:0 auto; width:100%; }
  .input-row { display:flex; gap:8px; align-items:flex-end; background:var(--panel); border:1.5px solid var(--border); border-radius:14px; padding:8px 10px; }
  textarea { flex:1; resize:none; background:transparent; border:none; color:var(--text); font-size:14.5px; outline:none; max-height:140px; }
  #send-btn { background:var(--accent); color:#fff; border:none; border-radius:10px; width:36px; height:36px; cursor:pointer; display:flex; align-items:center; justify-content:center; }
  .quick-btn { background:var(--panel); border:1px solid var(--border); color:var(--text); font-size:12px; padding:6px 12px; border-radius:20px; cursor:pointer; }
  #sidebar-overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,0.5); z-index:99; }
  @media(max-width:768px) {
    #sidebar { position:fixed; top:0; left:0; z-index:100; height:100%; transform:translateX(0); }
    #sidebar.hidden { transform:translateX(-105%); }
    #sidebar-overlay.show { display:block; }
  }
</style>
</head>
<body>
<div class="layout">
  <div id="sidebar-overlay"></div>
  <div id="sidebar">
    <button id="new-chat-btn">+ New chat</button>
    <div id="conv-list"></div>
    <div style="padding:12px; font-size:10px; color:var(--muted); text-align:center;">Ꮇʏᴛʜɪᴄ ᴀɪ by Aarav Singh</div>
  </div>
  <div class="app">
    <header>
      <div class="left">
        <button id="sidebar-toggle">☰</button>
        <h1 style="font-size:16px; color:var(--accent);">Ꮇʏᴛʜɪᴄ ᴀɪ</h1>
      </div>
      <div class="right">
        <button id="fullscreen-btn" title="Toggle Fullscreen"><span id="fullscreen-icon">⛶</span></button>
        <button id="name-btn" title="Set Name">🙂</button>
        <button id="settings-btn" title="Settings">⚙</button>
        <button id="export-btn" title="Export">⬇</button>
        <button id="clear-btn" title="Delete Chat" style="color:#ef4444;">✕</button>
      </div>
    </header>
    <div id="messages-wrap">
      <div id="messages"></div>
    </div>
    <div id="quick-actions" style="display:flex; gap:8px; padding:0 20px 8px; max-width:760px; margin:0 auto; overflow-x:auto;">
      <button class="quick-btn" onclick="injectPrompt('Draw a ')">🎨 Image</button>
      <button class="quick-btn" id="ghibli-btn">🌿 Ghibli Me</button>
      <button class="quick-btn" onclick="injectPrompt('Help me with this: ')">📚 Homework</button>
    </div>
    <div class="input-area">
      <form id="chat-form">
        <div class="input-row">
          <button type="button" class="quick-btn" onclick="document.getElementById('file-input').click()" style="padding:0; width:34px; height:34px; border-radius:8px;">📎</button>
          <input type="file" id="file-input" style="display:none" onchange="handleFile(this)">
          <textarea id="input" rows="1" placeholder="Message Ꮇʏᴛʜɪᴄ ᴀɪ..."></textarea>
          <button id="send-btn" type="submit">▲</button>
        </div>
      </form>
    </div>
  </div>
</div>

<!-- All feature modals (Ghibli, Settings, Name) preserved here -->
<div id="settings-modal-overlay" style="display:none; position:fixed; inset:0; background:rgba(0,0,0,0.5); z-index:200; align-items:center; justify-content:center;">
  <div style="background:var(--bg); border:1px solid var(--border); padding:20px; border-radius:12px; width:90%; max-width:400px;">
    <h3>Settings</h3><br>
    <label>Theme:</label>
    <button class="quick-btn" onclick="setTheme('dark')">Dark</button>
    <button class="quick-btn" onclick="setTheme('light')">Light</button><br><br>
    <button class="quick-btn" style="width:100%;" onclick="document.getElementById('settings-modal-overlay').style.display='none'">Close</button>
  </div>
</div>

<script>
const messagesEl = document.getElementById('messages');
const chatForm = document.getElementById('chat-form');
const inputEl = document.getElementById('input');
const fullscreenBtn = document.getElementById('fullscreen-btn');
const fullscreenIcon = document.getElementById('fullscreen-icon');
let activeConvId = null;

// Fullscreen Logic
function isFullscreen() { return !!(document.fullscreenElement || document.webkitFullscreenElement); }
async function toggleFullscreen() {
  if (!isFullscreen()) {
    if (document.documentElement.requestFullscreen) await document.documentElement.requestFullscreen();
    else if (document.documentElement.webkitRequestFullscreen) await document.documentElement.webkitRequestFullscreen();
  } else {
    if (document.exitFullscreen) await document.exitFullscreen();
    else if (document.webkitExitFullscreen) await document.webkitExitFullscreen();
  }
}
function updateFullscreenBtn() {
  if (isFullscreen()) {
    fullscreenIcon.textContent = '⤢';
    fullscreenBtn.classList.add('active');
  } else {
    fullscreenIcon.textContent = '⛶';
    fullscreenBtn.classList.remove('active');
  }
}
fullscreenBtn.addEventListener('click', toggleFullscreen);
document.addEventListener('fullscreenchange', updateFullscreenBtn);
document.addEventListener('webkitfullscreenchange', updateFullscreenBtn);

// Chat Logic
function injectPrompt(txt) { inputEl.value = txt; inputEl.focus(); }
function addMessage(role, text) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.textContent = text;
  messagesEl.appendChild(div);
  document.getElementById('messages-wrap').scrollTop = messagesEl.scrollHeight;
}

chatForm.onsubmit = async (e) => {
  e.preventDefault();
  const msg = inputEl.value.trim();
  if (!msg) return;
  addMessage('user', msg);
  inputEl.value = '';
  
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      body: JSON.stringify({ message: msg, conversation_id: activeConvId })
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let aiMsg = '';
    const aiDiv = document.createElement('div');
    aiDiv.className = 'msg ai';
    messagesEl.appendChild(aiDiv);
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      aiMsg += decoder.decode(value);
      aiDiv.textContent = aiMsg;
    }
    if (res.headers.get('X-Conversation-Id')) activeConvId = res.headers.get('X-Conversation-Id');
  } catch (err) { addMessage('ai', 'Error connecting to server.'); }
};

document.getElementById('sidebar-toggle').onclick = () => {
  document.getElementById('sidebar').classList.toggle('hidden');
};

function setTheme(t) {
  document.body.className = t === 'light' ? 'theme-light' : '';
  localStorage.setItem('theme', t);
}

// Initial
setTheme(localStorage.getItem('theme') || 'dark');
</script>
</body>
</html>
"""

@app.route("/")
@login_required
def index(): return Response(PAGE, mimetype="text/html")

@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json(force=True) or {}
    user_msg = data.get("message", "")
    conv_id = data.get("conversation_id") or str(uuid.uuid4())
    username = current_username()
    
    conv = load_conversation(username, conv_id) or {"title": make_title(user_msg), "messages": []}
    conv["messages"].append({"role": "user", "parts": [{"text": user_msg}]})
    
    def generate():
        full_reply = ""
        payload = {
            "contents": conv["messages"],
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT + GEMINI_SEARCH_ADDENDUM}]},
            "tools": [{"google_search": {}}],
        }
        
        # Simple stream for this demo (expandable to Groq/Ollama as per original)
        resp = requests.post(GEMINI_STREAM_URL, params={"key": GEMINI_API_KEY, "alt": "sse"}, json=payload, stream=True)
        for line in resp.iter_lines(decode_unicode=True):
            if line.startswith("data:"):
                try:
                    obj = json.loads(line[5:])
                    chunk = obj["candidates"][0]["content"]["parts"][0]["text"]
                    full_reply += chunk
                    yield chunk
                except: pass
        
        conv["messages"].append({"role": "model", "parts": [{"text": full_reply}]})
        save_conversation(username, conv_id, conv)

    r = Response(stream_with_context(generate()), mimetype="text/plain")
    r.headers["X-Conversation-Id"] = conv_id
    return r

# --- API Endpoints ---
@app.route("/api/conversations", methods=["GET"])
@login_required
def api_list(): return jsonify({"conversations": list_conversations(current_username())})

@app.route("/api/conversations/<cid>", methods=["DELETE"])
@login_required
def api_del(cid): delete_conversation(current_username(), cid); return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
