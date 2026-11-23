"""The Bhaskera home page: one self-contained HTML document (no outside fonts, scripts or images), because it is
served through an SSH tunnel and must work on any laptop. It reads /api/state every few seconds (see hub_state.py)."""

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bhaskera</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Ccircle cx='16' cy='16' r='7' fill='%23a8381f'/%3E%3Cg stroke='%23a8381f' stroke-width='2.4' stroke-linecap='round'%3E%3Cpath d='M16 2v4M16 26v4M2 16h4M26 16h4M6 6l3 3M23 23l3 3M26 6l-3 3M9 23l-3 3'/%3E%3C/g%3E%3C/svg%3E">
<style>
:root{
  --bg:#f3efe6; --surface:#faf8f2; --surface-2:#ece7da; --border:#d6cdb8; --text:#1f1c17; --muted:#6c6555;
  --ink:#1f1c17; --paper:#f3efe6;
  --accent:#a8381f; --accent-ink:#8a2c17; --accent-soft:#e9dccf;
  --ok:#2f6b4f; --ok-soft:transparent; --warn:#94650a; --warn-soft:#efe3c2; --bad:#a8381f; --bad-soft:transparent;
  --info:#2c4a7a; --info-soft:transparent;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
:root[data-theme="dark"]{
  --bg:#16140f; --surface:#1d1a14; --surface-2:#26231b; --border:#3b362a; --text:#ece6d6; --muted:#a39b86;
  --ink:#ece6d6; --paper:#16140f; --accent:#e0765a; --accent-ink:#ee9077; --accent-soft:#2f2018;
  --ok:#7fc29f; --warn:#e0b04c; --warn-soft:#33290f; --bad:#e0765a; --info:#8fb0e0;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#16140f; --surface:#1d1a14; --surface-2:#26231b; --border:#3b362a; --text:#ece6d6; --muted:#a39b86;
  --ink:#ece6d6; --paper:#16140f; --accent:#e0765a; --accent-ink:#ee9077; --accent-soft:#2f2018;
  --ok:#7fc29f; --warn:#e0b04c; --warn-soft:#33290f; --bad:#e0765a; --info:#8fb0e0;
}}
*{box-sizing:border-box} html{-webkit-text-size-adjust:100%}
[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.55 var(--sans)}
a{color:inherit;text-decoration:none}
button{font:inherit;color:inherit;cursor:pointer}
.wrap{max-width:1120px;margin:0 auto;padding:0 24px}
.top{border-bottom:1px solid var(--ink)}
.top .wrap{display:flex;align-items:center;gap:14px;height:64px}
.brand{display:flex;align-items:center;gap:11px}
.brand svg{width:26px;height:26px;color:var(--accent)}
.brand b{font:600 21px/1 var(--serif);letter-spacing:.005em} .brand small{display:block;color:var(--muted);font-size:12px;margin-top:3px}
.spacer{flex:1}
.pill{font-size:12.5px;color:var(--muted);display:inline-flex;align-items:center;gap:7px}
.dot{width:7px;height:7px;background:var(--ok);display:inline-block}
.dot.stale{background:var(--bad)}
.icon-btn{width:30px;height:30px;display:grid;place-items:center;border:1px solid var(--border);background:transparent;border-radius:3px}
.icon-btn:hover{border-color:var(--ink)}
.icon-btn svg{width:16px;height:16px}
main{padding:10px 0 60px}
h2{font:500 22px/1.2 var(--serif);margin:34px 0 12px;display:flex;align-items:baseline;gap:10px}
h2 .count{font:12px var(--mono);color:var(--muted)}
.card{background:transparent;border:0;border-top:1px solid var(--ink);border-bottom:1px solid var(--border)}
.job{padding:16px 0 18px}
.job-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 14px}
.job-name{font:500 19px/1.2 var(--serif)}
.muted{color:var(--muted)}
.mono{font-family:var(--mono)}
.badge{display:inline-flex;align-items:center;gap:6px;font-size:12.5px;font-weight:600}
.badge::before{content:"";width:7px;height:7px;background:currentColor}
.b-ok{color:var(--ok)} .b-warn{color:var(--warn)} .b-bad{color:var(--bad)} .b-info{color:var(--info)} .b-mute{color:var(--muted)}
.chip{font:12px var(--mono);color:var(--muted);padding:0}
.chip+.chip::before{content:"/";margin:0 8px;color:var(--border)}
.meta{display:flex;flex-wrap:wrap;gap:0;margin-left:auto}
.bar{height:4px;background:var(--surface-2);overflow:hidden}
.bar>i{display:block;height:100%;background:var(--ink);transition:width .6s ease}
.bar.thin{height:3px} .bar.thin>i{background:var(--muted)} .bar.hot>i{background:var(--warn)}
.progress{margin:14px 0 4px}
.progress .row{display:flex;justify-content:space-between;font-size:12.5px;color:var(--muted);margin-bottom:6px}
.progress .row b{color:var(--text)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(118px,1fr));gap:0;margin:16px 0;border-left:1px solid var(--border)}
.tile{padding:2px 14px;border-right:0;border-left:0}
.tiles .tile:not(:first-child){border-left:1px solid var(--border)}
.tile .k{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em}
.tile .v{font:500 20px/1.3 var(--mono);margin-top:1px}
.tile .v small{font-size:12px;color:var(--muted)}
.nodes{display:grid;grid-template-columns:repeat(auto-fill,minmax(176px,1fr));gap:14px 18px;margin:6px 0 14px}
.node{padding:6px 0 0;border-top:1px solid var(--border)}
.node .h{display:flex;justify-content:space-between;font-size:12px;margin-bottom:6px}
.node .h b{font-family:var(--mono);font-weight:600}
.gpu{display:flex;align-items:center;gap:7px;font:11.5px var(--mono);color:var(--muted);margin-top:4px}
.gpu .bar{flex:1;height:4px}
.gpu span:last-child{width:32px;text-align:right}
.actions{display:flex;flex-wrap:wrap;gap:8px}
.btn{display:inline-flex;align-items:center;gap:7px;border:1px solid var(--ink);background:transparent;color:var(--text);border-radius:3px;padding:6px 13px;font-weight:550;font-size:13px}
.btn:hover{background:var(--surface-2)}
.btn.primary{background:var(--ink);color:var(--paper)}
.btn.primary:hover{background:var(--accent);border-color:var(--accent);color:#fff}
.btn[aria-disabled=true]{opacity:.4;pointer-events:none}
.btn:disabled{opacity:.4;pointer-events:none}
.btn svg{width:14px;height:14px}
.btn.small{padding:3px 9px;font-size:12px}
.empty{padding:28px 0}
.empty h3{margin:4px 0;font:500 17px var(--serif)} .empty p{margin:4px 0;color:var(--muted);max-width:560px}
code,.code{font:12.5px var(--mono);background:var(--surface-2);border-radius:2px;padding:1px 6px}
pre.code{padding:9px 12px;display:inline-block;text-align:left;margin:10px 0 0}
.tablebox{overflow:hidden}
.tools{display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:10px 0;border-bottom:1px solid var(--border)}
.tools input{flex:1;min-width:160px;border:0;border-bottom:1px solid var(--border);background:transparent;color:var(--text);padding:5px 2px;font:inherit}
.tools input:focus{outline:0;border-bottom-color:var(--ink)}
.seg{display:inline-flex;gap:16px}
.seg button{border:0;background:transparent;padding:4px 0;font-size:12.5px;color:var(--muted);border-bottom:2px solid transparent}
.seg button:hover{color:var(--text)}
.seg button[aria-pressed=true]{color:var(--text);font-weight:650;border-bottom-color:var(--accent)}
.seg button:disabled{opacity:.4}
.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13px}
th{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);font-weight:600;text-align:left;padding:9px 12px 8px 0;white-space:nowrap;border-bottom:1px solid var(--border)}
td{padding:10px 12px 10px 0;border-bottom:1px solid var(--border);white-space:nowrap}
tr:last-child td{border-bottom:0}
tr:hover td{background:var(--surface)}
td.num,th.num{text-align:right;font-family:var(--mono);font-size:12.5px}
td a.run{font-weight:600;display:inline-block;max-width:240px;overflow:hidden;text-overflow:ellipsis;vertical-align:bottom;white-space:nowrap}
td a.run:hover{text-decoration:underline;text-underline-offset:3px}
.nav{display:flex;gap:18px;margin-left:22px}
.nav a{font-size:13px;color:var(--muted);border-bottom:1px solid transparent;padding:2px 0}
.nav a:hover{color:var(--text);border-bottom-color:var(--text)}
.nav a.off{opacity:.45;pointer-events:none}
@media (max-width:820px){.nav{display:none}}
.note-card{padding:12px 0;color:var(--muted)}
.more{margin-top:30px;border-top:1px solid var(--border)}
.more summary{cursor:pointer;color:var(--muted);font-size:13px;padding:10px 0}
.more summary:hover{color:var(--text)}
.more summary .bad{color:var(--bad);font-weight:650}
.more>div{padding:2px 0 8px}
.more h3{margin:14px 0 6px;font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);font-weight:600}
.part{margin:11px 0}
.part .row{display:flex;flex-wrap:wrap;justify-content:space-between;gap:2px 10px;font-size:12.5px;margin-bottom:5px}
.part .row b{white-space:nowrap}
.note{font-size:12.5px;color:var(--muted)}
.bad-node{padding:8px 0;border-top:1px solid var(--border);font-size:12.5px}
.runact{display:flex;gap:6px;justify-content:flex-end}
.quiet{padding:20px 0;color:var(--muted);border-top:1px solid var(--ink);border-bottom:1px solid var(--border)}
footer{color:var(--muted);font-size:12.5px;margin-top:40px;padding-top:14px;border-top:1px solid var(--border)}
footer a{text-decoration:underline;text-underline-offset:3px}
dialog{width:min(900px,94vw);max-height:88vh;padding:0;border:1px solid var(--ink);border-radius:3px;background:var(--bg);color:var(--text);box-shadow:6px 6px 0 rgba(31,28,23,.18)}
dialog::backdrop{background:rgba(31,28,23,.45)}
.dlg-head{display:flex;align-items:center;gap:10px;padding:12px 22px;border-bottom:1px solid var(--border)}
.dlg-head b{font:500 18px var(--serif)}
#logtext{margin:0;padding:14px 18px;background:#14120d;color:#d9d2c0;font:12px/1.55 var(--mono);height:62vh;overflow:auto;white-space:pre-wrap;word-break:break-word}
.skeleton{height:90px;background:var(--surface-2)}
/* chat */
.chat-head{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:center;justify-content:space-between;padding:14px 0;border-bottom:1px solid var(--border)}
.chat-controls{display:flex;flex-wrap:wrap;gap:14px;align-items:center}
.chat-controls select,.chat-controls input[type=range]{accent-color:var(--accent)}
.chat-controls select{border:1px solid var(--border);background:transparent;color:var(--text);border-radius:3px;padding:4px 6px;font:inherit}
.chat-controls label{display:flex;align-items:center;gap:6px;font-size:12.5px}
.msgs{padding:16px 0;min-height:160px;max-height:460px;overflow:auto;display:flex;flex-direction:column;gap:16px}
.msg{max-width:78ch;white-space:pre-wrap;word-break:break-word;line-height:1.6}
.msg.user{align-self:flex-end;background:var(--ink);color:var(--paper);padding:8px 13px;border-radius:3px}
.msg.bot{align-self:flex-start;border-left:3px solid var(--border);padding:2px 0 2px 14px}
.msg.bot:empty::after{content:"thinking…";color:var(--muted)}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:22px;align-self:stretch}
@media (max-width:760px){.pair{grid-template-columns:1fr}}
.pair .col{border-left:3px solid var(--border);padding:2px 0 2px 14px}
.pair .col.tuned{border-left-color:var(--accent)}
.pair .col small{display:block;font-size:11px;text-transform:uppercase;letter-spacing:.08em;font-weight:650;margin-bottom:4px;color:var(--muted)}
.pair .col.tuned small{color:var(--accent-ink)}
.pair .col .txt{white-space:pre-wrap;word-break:break-word;line-height:1.6}
.pair .col .txt:empty::after{content:"thinking…";color:var(--muted)}
.composer{display:flex;gap:10px;padding:14px 0 16px;border-top:1px solid var(--border);align-items:flex-end}
.composer textarea{flex:1;resize:vertical;min-height:44px;max-height:180px;border:1px solid var(--border);background:var(--surface);color:var(--text);border-radius:3px;padding:9px 11px;font:inherit}
.composer textarea:focus{outline:0;border-color:var(--ink)}
.cmds td,.cmds th{padding:9px 12px 9px 0}
.fgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:16px;padding:18px 22px 6px}
.fld label{display:block;font-size:11px;color:var(--muted);margin-bottom:4px;font-weight:600;text-transform:uppercase;letter-spacing:.07em}
.fld select,.fld input[type=text],.fld input[type=number]{width:100%;border:1px solid var(--border);background:var(--surface);color:var(--text);border-radius:3px;padding:8px 10px;font:inherit}
.fld select:focus,.fld input:focus{outline:0;border-color:var(--ink)}
.fld .hint{font-size:12px;color:var(--muted);margin-top:4px}
.fld.wide{grid-column:1/-1}
.fld.check label{display:flex;gap:8px;align-items:center;font-size:13px;color:var(--text);font-weight:500;text-transform:none;letter-spacing:0}
.startnote{margin:10px 22px 0;padding:9px 12px;border-left:3px solid var(--warn);background:var(--warn-soft);font-size:13px}
.result{margin:12px 22px 0;padding:11px 14px;border-left:3px solid var(--border);background:var(--surface);font:12.5px/1.5 var(--mono);white-space:pre-wrap;word-break:break-word;max-height:300px;overflow:auto}
.result.ok{border-left-color:var(--ok)} .result.bad{border-left-color:var(--bad)}
.result b{font-family:var(--sans)}
.dlg-foot{display:flex;gap:10px;justify-content:flex-end;padding:16px 22px}
.h2row{display:flex;align-items:baseline;gap:14px;margin-top:30px}.h2row h2{margin:0}.h2row .btn{margin:0;align-self:center}.h2row .muted{font-size:13px}
.nodebox{margin:4px 0 14px}.nodebox>summary,.opt>summary{cursor:pointer;color:var(--muted);font-size:12.5px}
.nodebox>summary:hover,.opt>summary:hover{color:var(--text)}
.opt{position:relative}.opt>div{position:absolute;right:0;z-index:2;display:flex;flex-direction:column;gap:10px;padding:12px 14px;margin-top:6px;background:var(--bg);border:1px solid var(--ink);border-radius:3px;min-width:230px}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
</head>
<body>
<header class="top"><div class="wrap">
  <div class="brand">
    <svg viewBox="0 0 32 32" aria-hidden="true"><circle cx="16" cy="16" r="7" fill="currentColor"/><g stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M16 2v4M16 26v4M2 16h4M26 16h4M6 6l3 3M23 23l3 3M26 6l-3 3M9 23l-3 3"/></g></svg>
    <div><b>Bhaskera</b><small id="tag"></small></div>
  </div>
  <div class="spacer"></div>
  <nav class="nav" id="nav" aria-label="Tools"></nav>
  <span class="pill" id="who" hidden></span>
  <span class="pill" title="The page refreshes by itself"><span class="dot" id="dot"></span><span id="updated">connecting…</span></span>
  <button class="icon-btn" id="theme" aria-label="Switch between light and dark" title="Light / dark"></button>
</div></header>

<main class="wrap">
  <div class="h2row"><h2>Jobs <span class="count" id="jobcount">0</span></h2><span class="muted" id="stats"></span><span class="spacer"></span>
    <button class="btn primary" id="startbtn" type="button">Start a job</button></div>
  <div id="jobs"><div class="skeleton"></div></div>

  <div id="chat">
    <div id="chat-info" class="card note-card" hidden></div>
    <div id="chat-live" class="card chat" hidden>
      <div class="chat-head">
        <div><b id="chat-model"></b> <span class="muted" id="chat-adapter"></span> <span class="chip mono" id="chat-job"></span></div>
        <div class="chat-controls">
          <select id="chat-jobsel" hidden aria-label="Which chat server"></select>
          <div class="seg" id="chat-mode" role="group" aria-label="Which model answers">
            <button data-m="tuned" aria-pressed="true">Tuned</button><button data-m="base" aria-pressed="false">Base</button><button data-m="both" aria-pressed="false">Compare</button>
          </div>
          <details class="opt"><summary>Options</summary><div>
            <label class="muted" title="Higher gives more varied answers">Creativity <input id="chat-temp" type="range" min="0" max="1.5" step="0.1" value="0.7"></label>
            <label class="muted">Longest answer <select id="chat-max" aria-label="Longest answer in tokens"><option>128</option><option selected>256</option><option>512</option><option>1024</option></select></label></div></details>
          <button class="btn small" id="chat-clear" type="button">Clear</button>
        </div>
      </div>
      <div class="msgs" id="chat-msgs" aria-live="polite"></div>
      <form class="composer" id="chat-form">
        <textarea id="chat-input" rows="2" placeholder="Ask the model something … (Enter sends, Shift+Enter starts a new line)" aria-label="Your message"></textarea>
        <button class="btn primary" id="chat-send" type="submit">Send</button><button class="btn" id="chat-stop" type="button" hidden>Stop</button>
      </form>
    </div>
  </div>

  <section>
    <h2>Runs <span class="count" id="runcount">0</span></h2>
    <div class="card tablebox">
      <div class="tools" id="runtools" hidden>
        <input id="q" type="search" placeholder="Search runs by name or job number" aria-label="Search runs">
        <div class="seg" id="filter" role="group" aria-label="Filter by status">
          <button data-f="" aria-pressed="true">All</button><button data-f="FINISHED" aria-pressed="false">Finished</button>
          <button data-f="FAILED" aria-pressed="false">Failed</button><button data-f="KILLED" aria-pressed="false">Killed</button>
        </div>
      </div>
      <div class="scroll"><table>
        <thead id="runhead"></thead>
        <tbody id="runs"></tbody>
      </table></div>
    </div>
  </section>
  <details class="more" id="morebox"><summary>Cluster limits and node health <span id="morewarn"></span></summary><div id="cluster"></div></details>
  <footer id="foot"></footer>
</main>

<dialog id="startbox">
  <div class="dlg-head"><b>Start a job</b><span class="spacer" style="flex:1"></span><button class="btn" id="startclose" type="button">Close</button></div>
  <div style="padding:14px 22px 0"><div class="seg" id="startkind" role="group" aria-label="What to start">
    <button type="button" data-k="train">Train a model</button><button type="button" data-k="serve">Chat with a trained model</button><button type="button" data-k="eval">Benchmark a trained model</button></div></div>
  <div id="startnote" class="startnote" hidden></div>
  <div class="fgrid" id="startfields"></div>
  <pre class="result" id="startout" hidden></pre>
  <div class="dlg-foot"><button class="btn" id="startcheck" type="button" title="Runs the partition limits and the preflight; submits nothing">Check first</button>
    <button class="btn primary" id="startgo" type="button">Start</button></div>
</dialog>

<dialog id="logbox">
  <div class="dlg-head"><b id="logtitle">Log</b><span class="muted" id="logmeta"></span><span class="spacer" style="flex:1"></span>
    <label class="muted" style="display:flex;align-items:center;gap:6px"><input type="checkbox" id="follow" checked> follow</label>
    <button class="btn" id="copylog">Copy</button><button class="btn" id="closelog">Close</button></div>
  <pre id="logtext">loading…</pre>
</dialog>

<script>
(function(){
"use strict";
var $ = function(s){return document.querySelector(s)};
var esc = function(s){return String(s==null?"":s).replace(/[&<>"']/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]})};
var PORT = location.port || "8265";
var MLFLOW = "http://mlflow.localhost:"+PORT+"/";
var GITHUB = "https://github.com/kishoryd/Bhaskera";
var svg = function(d){return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'+d+'</svg>'};
var ICON = {
  ray:svg('<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M19 5l-2 2M7 17l-2 2"/>'),
  chart:svg('<path d="M3 3v18h18"/><path d="M7 15l4-5 3 3 5-7"/>'),
  log:svg('<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 9l3 3-3 3M13 15h4"/>'),
  flask:svg('<path d="M9 3h6M10 3v6l-5 9a2 2 0 0 0 2 3h10a2 2 0 0 0 2-3l-5-9V3"/>'),
  box:svg('<path d="M21 8l-9-5-9 5 9 5 9-5z"/><path d="M3 8v8l9 5 9-5V8M12 13v8"/>'),
  trace:svg('<path d="M3 6h10M7 12h14M5 18h8"/>'),
  book:svg('<path d="M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2z"/><path d="M4 21V5"/>'),
  code:svg('<path d="M8 7l-5 5 5 5M16 7l5 5-5 5M14 4l-4 16"/>'),
  sun:svg('<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5L19 19M19 5l-1.5 1.5M6.5 17.5L5 19"/>'),
  moon:svg('<path d="M21 13A9 9 0 1 1 11 3a7 7 0 0 0 10 10z"/>')
};
var GPN = null;
var state = null, filter = "", query = "", lastOk = 0, logJob = null, logTimer = null;

/* ---- theme ---- */
function isDark(){
  var t = document.documentElement.getAttribute("data-theme");
  return t ? t==="dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
}
function paintTheme(){ $("#theme").innerHTML = isDark() ? ICON.sun : ICON.moon; }
try{ var saved = localStorage.getItem("bhaskera-theme"); if(saved) document.documentElement.setAttribute("data-theme", saved); }catch(e){}
var asked = (location.search.match(/[?&]theme=(light|dark)/)||[])[1];
if(asked) document.documentElement.setAttribute("data-theme", asked);
$("#theme").addEventListener("click", function(){
  var next = isDark() ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  try{ localStorage.setItem("bhaskera-theme", next); }catch(e){}
  paintTheme();
});
paintTheme();

/* ---- formatting ---- */
function dur(s){
  if(s==null) return "–"; s=Math.max(0,Math.round(s));
  var d=Math.floor(s/86400), h=Math.floor(s%86400/3600), m=Math.floor(s%3600/60);
  if(d) return d+"d "+h+"h"; if(h) return h+"h "+(m<10?"0":"")+m+"m"; if(m) return m+" min"; return s+" s";
}
function ago(ts){
  if(!ts) return "–"; var s=Date.now()/1000-ts;
  if(s<60) return "just now"; if(s<3600) return Math.round(s/60)+" min ago";
  if(s<86400) return Math.round(s/3600)+" h ago"; return Math.round(s/86400)+" d ago";
}
function fx(v,d){ return v==null||isNaN(v) ? "–" : Number(v).toFixed(d==null?2:d); }
function big(v){ if(v==null) return "–"; return v>=1e6 ? (v/1e6).toFixed(1)+"M" : v>=1e3 ? (v/1e3).toFixed(1)+"k" : String(Math.round(v)); }
function runBadge(r){
  if(r.status==="RUNNING" && r.stale) return '<span class="badge b-mute" title="No new numbers for a while: the job has probably ended without closing the run">Stalled</span>';
  return badge(r.status);
}
function badge(st){
  var cls={RUNNING:"b-ok",FINISHED:"b-ok",PENDING:"b-warn",COMPLETING:"b-info",FAILED:"b-bad",KILLED:"b-bad",SCHEDULED:"b-info"}[st]||"b-mute";
  var label={RUNNING:"Running",FINISHED:"Finished",PENDING:"Queued",COMPLETING:"Finishing",FAILED:"Failed",KILLED:"Killed"}[st]||st;
  return '<span class="badge '+cls+'">'+esc(label)+'</span>';
}
function link(href,label,icon,primary){
  return '<a class="btn'+(primary?" primary":"")+'"'+(href?' href="'+esc(href)+'" target="_blank" rel="noopener"':' aria-disabled="true"')+'>'+ICON[icon]+esc(label)+'</a>';
}
function tile(k,v){ return '<div class="tile"><div class="k">'+esc(k)+'</div><div class="v">'+v+'</div></div>'; }
function stat(k,v,sub){ return '<div class="card stat"><div class="k">'+esc(k)+'</div><div class="v">'+esc(v==null?"–":v)+'</div><div class="s">'+esc(sub||"")+'</div></div>'; }
function safe(u){ return /^https?:\/\//.test(u||"") ? u : ""; }
function bench(r){
  var d=r.lm_eval||{}, keys=Object.keys(d);
  if(!keys.length) return '<span class="muted">–</span>';
  var title=keys.map(function(k){return k.replace(/_/g," ")+": "+(d[k]>=0?"+":"")+(100*d[k]).toFixed(1)+" points"}).join("\n");
  var avg=keys.reduce(function(a,k){return a+d[k]},0)/keys.length;
  return '<span title="'+esc(title)+'" class="'+(avg>0.005?"badge b-ok":avg<-0.005?"badge b-bad":"badge b-mute")+'">'+keys.length+(keys.length===1?" task ":" tasks ")+(avg>=0?"+":"")+(100*avg).toFixed(1)+' pts</span>';
}
function runUrl(r){ return MLFLOW+"#/experiments/"+r.experiment_id+"/runs/"+r.run_id; }

/* ---- jobs ---- */
var openNodes={};
function nodesHtml(nodes,id){
  if(!nodes || !nodes.length) return "";
  return '<details class="nodebox" data-job="'+esc(id)+'"'+(openNodes[id]?" open":"")+'><summary>GPU load per node ('+nodes.length+')</summary><div class="nodes">'+nodes.map(function(n){
    var gpus=(n.gpus||[]).map(function(g,i){
      var u=g.util==null?0:g.util;
      return '<div class="gpu"><span>GPU '+i+'</span><div class="bar'+(u>=97?" hot":"")+'"><i style="width:'+u+'%"></i></div><span>'+(g.util==null?"–":u+"%")+'</span></div>';
    }).join("");
    return '<div class="node"><div class="h"><b>'+esc(n.host)+'</b><span class="muted">'+(n.alive?"up":"down")+'</span></div>'+gpus+'</div>';
  }).join("")+'</div></details>';
}
function serviceCard(j){
  var running=(j.state==="RUNNING"||j.state==="COMPLETING"), s=j.serve||{}, serve=j.kind==="serve";
  var h='<article class="card job"><div class="job-head">'+badge(j.state)+'<span class="job-name">'+esc(j.name)+'</span><span class="muted mono">#'+esc(j.id)+'</span>'+
        '<div class="meta"><span class="chip">'+(serve?"chat server":"benchmark")+'</span><span class="chip">'+esc(j.partition)+'</span>'+
        (!running&&j.reason?'<span class="chip">'+esc(j.reason)+'</span>':"")+'</div></div>';
  if(!running) return h+'</article>';
  var used=j.limit_s?Math.min(100,Math.round(100*j.elapsed_s/j.limit_s)):null;
  h+='<div class="progress"><div class="row"><span>'+(serve?(s.ready?'<b>Ready</b> · '+esc(s.model||"")+(s.adapter?' + adapter '+esc(s.adapter):''):'loading the model (a few minutes)'):'running the benchmarks')+'</span><span>'+dur(j.elapsed_s)+(j.limit_s?' of '+dur(j.limit_s):'')+'</span></div>'+
     (used!=null?'<div class="bar thin'+(used>=85?" hot":"")+'"><i style="width:'+used+'%"></i></div>':"")+'</div><div class="actions">'+
     (serve&&s.ready?'<a class="btn primary" href="#chat">'+ICON.flask+'Open the chat</a>':"")+
     '<button class="btn" data-log="'+esc(j.id)+'" data-name="'+esc(j.name)+'">'+ICON.log+'Logs</button></div></article>';
  return h;
}
function jobCard(j){
  if(j.kind==="serve"||j.kind==="eval") return serviceCard(j);
  var t=j.train||{}, running=(j.state==="RUNNING"||j.state==="COMPLETING");
  var base="http://job-"+j.id+".localhost:"+PORT+"/";
  var charts="http://grafana-"+j.id+".localhost:"+PORT+"/d/rayDefaultDashboard?orgId=1&from=now-1h&to=now&refresh=10s";
  var h='<article class="card job"><div class="job-head">'+badge(j.state)+'<span class="job-name">'+esc(j.name)+'</span>'+
        '<span class="muted mono">#'+esc(j.id)+'</span><div class="meta"><span class="chip">'+esc(j.partition)+'</span>'+
        '<span class="chip">'+j.node_count+(j.node_count===1?" node":" nodes")+(GPN?' · '+(GPN*j.node_count)+' GPUs':'')+'</span>'+
        (!running && j.reason ? '<span class="chip" title="Why the job waits">'+esc(j.reason)+'</span>' : "")+'</div></div>';
  if(running){
    var pr=t.progress, label;
    if(pr!=null){ label='<b>'+Math.round(pr*100)+'%</b> · step '+t.step+' of '+t.max_steps+(t.eta_s!=null?' · about '+dur(t.eta_s)+' left':''); }
    else if(t.step!=null){ label='step <b>'+t.step+'</b>'; }
    else { label='starting up – loading the model'; }
    var used=j.limit_s ? Math.min(100,Math.round(100*j.elapsed_s/j.limit_s)) : null;
    h+='<div class="progress"><div class="row"><span>'+label+'</span><span>'+dur(j.elapsed_s)+(j.limit_s?' of '+dur(j.limit_s):'')+'</span></div>'+
       (pr!=null ? '<div class="bar"><i style="width:'+Math.round(100*pr)+'%"></i></div>' : "")+
       (used!=null&&used>=85?'<div class="bar thin hot" style="margin-top:6px" title="Time limit nearly used"><i style="width:'+used+'%"></i></div>':"")+'</div>';
    var tl=[];
    if(t.loss!=null) tl.push(tile("Loss",fx(t.loss,3)));
    if(t.eval_loss!=null) tl.push(tile("Eval loss",fx(t.eval_loss,3)));
    if(t.eval_acc!=null) tl.push(tile("Eval accuracy",fx(100*t.eval_acc,1)+"<small> %</small>"));
    if(t.mfu!=null) tl.push(tile("MFU",fx(t.mfu,1)+"<small> %</small>"));
    if(t.tok_s!=null) tl.push(tile("Tokens / s",big(t.tok_s)));
    if(t.step_s!=null) tl.push(tile("Step time",fx(t.step_s,1)+"<small> s</small>"));
    if(tl.length) h+='<div class="tiles">'+tl.join("")+'</div>';
    h+=nodesHtml(j.nodes,j.id);
  }
  if(!running){
    return h+'</article>';
  }
  h+='<div class="actions">';
  if(running){
    h+=link(j.linked?base:null,"Ray Dashboard","ray",true)+link(j.linked?charts:null,"GPU charts","chart")+
       '<button class="btn" data-log="'+esc(j.id)+'" data-name="'+esc(j.name)+'">'+ICON.log+'Logs</button>';
  }
  h+=link(j.run?runUrl(j.run):(running?MLFLOW:null),j.run?"MLflow run":"MLflow","flask")+'</div></article>';
  return h;
}
function emptyJobs(){
  return '<div class="quiet">Nothing is running. Press <b>Start a job</b> to train a model, or to chat with or benchmark one you trained.</div>';
}

/* ---- runs ---- */
function renderRuns(){
  var rows=(state.runs||[]).filter(function(r){
    if(filter && r.status!==filter) return false;
    if(query && (r.name+" "+r.job+" "+(r.group||"")).toLowerCase().indexOf(query.toLowerCase())<0) return false;
    return true;
  });
  $("#runcount").textContent=rows.length;
  $("#runtools").hidden=(state.runs||[]).length<=6;
  var all=state.runs||[], hasEval=all.some(function(r){return r.eval_loss!=null}), hasBench=all.some(function(r){return r.lm_eval&&Object.keys(r.lm_eval).length});
  $("#runhead").innerHTML='<tr><th>Run</th><th>Status</th><th>Started</th>'+(hasEval?'<th class="num">Eval loss</th>':'')+(hasBench?'<th>Benchmarks</th>':'')+'<th></th></tr>';
  $("#runs").innerHTML = rows.length ? rows.map(function(r){
    var act=(r.has_adapter&&r.status!=="RUNNING") ? '<div class="runact"><button class="btn small primary" data-start="serve" data-run="'+esc(r.run_id)+'">Chat</button><button class="btn small" data-start="eval" data-run="'+esc(r.run_id)+'">Benchmark</button></div>' : "";
    return '<tr><td><a class="run" href="'+esc(runUrl(r))+'" target="_blank" rel="noopener">'+esc(r.name)+'</a>'+(r.group?' <span class="chip">'+esc(r.group)+'</span>':"")+'</td>'+
      '<td>'+runBadge(r)+'</td><td class="muted">'+ago(r.started)+'</td>'+(hasEval?'<td class="num">'+fx(r.eval_loss,3)+'</td>':'')+(hasBench?'<td>'+bench(r)+'</td>':'')+'<td>'+act+'</td></tr>';
  }).join("") : '<tr><td colspan="6" class="muted" style="text-align:center;padding:26px">'+(all.length?"No runs match.":"No runs yet.")+'</td></tr>';
}

/* ---- side panel ---- */
function renderSide(){
  var s=state, first=(s.jobs||[]).filter(function(j){return j.linked&&(j.kind||"train")==="train"})[0];
  var items=[["MLflow",MLFLOW],["Models",MLFLOW+"#/models"],["Ray",first?"http://job-"+first.id+".localhost:"+PORT+"/":null],["Docs",safe(s.docs_url)||null]];
  $("#nav").innerHTML=items.map(function(l){
    return '<a'+(l[1]?' href="'+esc(l[1])+'" target="_blank" rel="noopener"':' class="off" title="Opens when a job runs"')+'>'+esc(l[0])+'</a>';
  }).join("");
  var parts=(s.partitions||[]).filter(function(p){return p.max_submitted!=null});
  var h='<h3>Your job slots</h3>'+(parts.length ? parts.map(function(p){
    var pct=Math.min(100,Math.round(100*p.submitted/Math.max(1,p.max_submitted)));
    return '<div class="part"><div class="row"><b>'+esc(p.partition)+'</b><span class="muted">'+p.submitted+' of '+p.max_submitted+' submitted · '+p.running+(p.max_running!=null?' of '+p.max_running:'')+' running'+(p.max_hours?' · '+p.max_hours+' h max':'')+'</span></div><div class="bar thin'+(pct>=100?" hot":"")+'"><i style="width:'+pct+'%"></i></div></div>';
  }).join("") : '<p class="note">Limits are not available.</p>');
  var bad=s.bad_nodes||[];
  h+='<h3 style="margin-top:18px">Faulty nodes</h3>'+(bad.length ? bad.map(function(b){
    return '<div class="bad-node"><b class="mono">'+esc(b.host)+'</b> <span class="muted">· '+ago(b.since)+'</span><br><span class="note">'+esc(b.reason)+'</span></div>';
  }).join("")+'<p class="note">New jobs skip these for three days.</p>' : '<p class="note">None. A job checks every node before it starts and remembers any that fail.</p>');
  $("#cluster").innerHTML=h;
  $("#morewarn").innerHTML=bad.length?'<span class="bad">· '+bad.length+' faulty node'+(bad.length>1?"s":"")+'</span>':"";
}

/* ---- page ---- */
function render(){
  var s=state; if(!s) return;
  GPN = s.gpus_per_node || null;
  $("#who").hidden=false; $("#who").textContent=s.user;
  $("#tag").textContent = s.cluster || ""; $("#tag").hidden = !s.cluster;
  document.title = s.cluster ? "Bhaskera · "+s.cluster : "Bhaskera";
  var T=s.totals||{}, bad=(s.bad_nodes||[]).length;
  var part=(s.partitions||[]).filter(function(p){return p.max_submitted!=null}).sort(function(a,b){return b.submitted-a.submitted})[0];
  var bits=[];
  bits.push(T.running ? T.running+" running"+(T.nodes?" · "+T.nodes+(T.nodes===1?" node":" nodes")+(T.gpus?" · "+T.gpus+" GPUs":""):"") : "idle");
  if(T.queued) bits.push(T.queued+" waiting for nodes");
  if(T.gpu_util!=null) bits.push("GPUs "+T.gpu_util+"% busy");
  if(part) bits.push(part.submitted+" of "+part.max_submitted+" job slots used");
  $("#stats").textContent=bits.join("  ·  ");
  var order={RUNNING:0,COMPLETING:1,PENDING:2};
  var jobs=(s.jobs||[]).slice().sort(function(a,b){ return ((order[a.state]||0)-(order[b.state]||0)) || (Number(b.id)-Number(a.id)); });
  $("#jobcount").textContent=jobs.length;
  $("#jobs").innerHTML=jobs.length ? jobs.map(jobCard).join("") : emptyJobs();
  renderRuns(); renderSide(); renderChat();
  $("#foot").innerHTML='Bhaskera '+esc(s.version||"")+' · <a href="'+esc(safe(s.docs_url)||"#")+'" target="_blank" rel="noopener">documentation</a> · <a href="'+GITHUB+'" target="_blank" rel="noopener">source</a>';
}
function clock(){
  var dot=$("#dot"), el=$("#updated");
  if(!lastOk){ return; }
  var age=Math.round((Date.now()-lastOk)/1000);
  dot.classList.toggle("stale", age>30);
  el.textContent = age>30 ? "connection lost – retrying" : (age<3 ? "live" : "updated "+age+" s ago");
}
function refresh(){
  fetch("/api/state",{cache:"no-store"}).then(function(r){ if(!r.ok) throw new Error(r.status); return r.json(); })
    .then(function(s){ state=s; lastOk=Date.now(); render(); clock(); })
    .catch(function(){ clock(); });
}
$("#q").addEventListener("input", function(e){ query=e.target.value; if(state) renderRuns(); });
$("#filter").addEventListener("click", function(e){
  var b=e.target.closest("button"); if(!b) return; filter=b.getAttribute("data-f");
  Array.prototype.forEach.call(this.querySelectorAll("button"), function(x){ x.setAttribute("aria-pressed", x===b ? "true":"false"); });
  if(state) renderRuns();
});

/* ---- log window ---- */
function loadLog(){
  if(!logJob) return;
  fetch("/api/log?job="+encodeURIComponent(logJob)+"&lines=400",{cache:"no-store"}).then(function(r){return r.json()}).then(function(d){
    var box=$("#logtext");
    if(d.error){ box.textContent=d.error; return; }
    box.textContent=d.lines.join("\n");
    $("#logmeta").textContent=d.lines.length+" lines, newest at the bottom";
    if($("#follow").checked) box.scrollTop=box.scrollHeight;
  }).catch(function(){ $("#logmeta").textContent="could not load the log"; });
}
$("#jobs").addEventListener("click", function(e){
  var b=e.target.closest("[data-log]"); if(!b) return;
  logJob=b.getAttribute("data-log"); $("#logtitle").textContent="Log of "+b.getAttribute("data-name")+" (#"+logJob+")";
  $("#logtext").textContent="loading…"; $("#logbox").showModal(); loadLog();
  clearInterval(logTimer); logTimer=setInterval(loadLog,5000);
});
function closeLog(){ clearInterval(logTimer); logJob=null; if($("#logbox").open) $("#logbox").close(); }
$("#closelog").addEventListener("click", closeLog);
$("#logbox").addEventListener("close", function(){ clearInterval(logTimer); logJob=null; });
$("#copylog").addEventListener("click", function(){
  var t=$("#logtext").textContent, b=this;
  (navigator.clipboard ? navigator.clipboard.writeText(t) : Promise.reject()).then(function(){ b.textContent="Copied"; setTimeout(function(){b.textContent="Copy"},1500); }).catch(function(){});
});

/* ---- chat with a tuned model ---- */
var chat = {job:null, messages:[], busy:false, ctrl:null, mode:"tuned"};
function serveJobs(){ return (state.jobs||[]).filter(function(j){return j.kind==="serve"&&(j.state==="RUNNING"||j.state==="COMPLETING")}); }
function copyCmd(text,btn){
  var done=function(){ var o=btn.textContent; btn.textContent="Copied"; setTimeout(function(){btn.textContent=o},1500); };
  (navigator.clipboard?navigator.clipboard.writeText(text):Promise.reject()).then(done).catch(function(){ window.prompt("Copy this command:",text); });
}
function renderChat(){
  if(!state) return;
  var sj=serveJobs(), ready=sj.filter(function(j){return j.serve&&j.serve.ready});
  var info=$("#chat-info"), live=$("#chat-live");
  if(!ready.length){
    live.hidden=true; info.hidden=!sj.length;
    if(sj.length) info.innerHTML='Loading the model for chat (job #'+esc(sj[0].id)+'). This takes a few minutes; the chat opens here by itself.';
    return;
  }
  info.hidden=true; live.hidden=false;
  var cur=ready.filter(function(j){return j.id===chat.job})[0]||ready[0];
  chat.job=cur.id;
  $("#chat-model").textContent=cur.serve.model||cur.name;
  $("#chat-adapter").textContent=cur.serve.adapter?"+ adapter "+cur.serve.adapter:"(no adapter: the base model only)";
  $("#chat-job").textContent="#"+cur.id;
  var sel=$("#chat-jobsel"); sel.hidden=ready.length<2;
  if(ready.length>1){ sel.innerHTML=ready.map(function(j){return '<option value="'+esc(j.id)+'"'+(j.id===chat.job?" selected":"")+'>#'+esc(j.id)+' '+esc(j.name)+'</option>'}).join(""); }
  var hasAdapter=!!cur.serve.adapter;
  Array.prototype.forEach.call($("#chat-mode").querySelectorAll("button"), function(b){ if(b.getAttribute("data-m")!=="tuned") b.disabled=!hasAdapter; });
  if(!hasAdapter) setMode("tuned");
}
function setMode(m){
  chat.mode=m;
  Array.prototype.forEach.call($("#chat-mode").querySelectorAll("button"), function(b){ b.setAttribute("aria-pressed", b.getAttribute("data-m")===m?"true":"false"); });
}
function addUser(text){ var d=document.createElement("div"); d.className="msg user"; d.textContent=text; $("#chat-msgs").appendChild(d); return d; }
function addBot(){ var d=document.createElement("div"); d.className="msg bot"; $("#chat-msgs").appendChild(d); return d; }
function addPair(){
  var p=document.createElement("div"); p.className="pair";
  p.innerHTML='<div class="col tuned"><small>Tuned model</small><div class="txt"></div></div><div class="col base"><small>Base model</small><div class="txt"></div></div>';
  $("#chat-msgs").appendChild(p); return [p.querySelector(".tuned .txt"), p.querySelector(".base .txt")];
}
function scrollChat(){ var m=$("#chat-msgs"); m.scrollTop=m.scrollHeight; }
function streamAnswer(history,useAdapter,el){
  var ctrl=new AbortController(); chat.ctrl=ctrl; var text="";
  return fetch("/api/chat?job="+encodeURIComponent(chat.job),{method:"POST",signal:ctrl.signal,
      headers:{"Content-Type":"application/json","X-Requested-With":"bhaskera"},
      body:JSON.stringify({messages:history,stream:true,use_adapter:useAdapter,max_tokens:parseInt($("#chat-max").value,10),temperature:parseFloat($("#chat-temp").value)})})
    .then(function(res){
      if(!res.ok) return res.text().then(function(t){ var m=t; try{m=JSON.parse(t).error||t}catch(e){} throw new Error(m); });
      var reader=res.body.getReader(), dec=new TextDecoder(), buf="";
      function pump(){
        return reader.read().then(function(r){
          if(r.done) return text;
          buf+=dec.decode(r.value,{stream:true});
          var lines=buf.split("\n"); buf=lines.pop();
          for(var i=0;i<lines.length;i++){
            if(lines[i].indexOf("data: ")!==0) continue;
            var d=lines[i].slice(6); if(d==="[DONE]") return text;
            var j=JSON.parse(d); if(j.error) throw new Error(j.error);
            var c=j.choices&&j.choices[0]&&j.choices[0].delta&&j.choices[0].delta.content;
            if(c){ text+=c; el.textContent=text; scrollChat(); }
          }
          return pump();
        });
      }
      return pump();
    })
    .catch(function(e){ if(e.name!=="AbortError") el.textContent=(text?text+"\n":"")+"[could not get an answer: "+e.message+"]"; return text; });
}
function ask(text){
  chat.busy=true; $("#chat-send").hidden=true; $("#chat-stop").hidden=false;
  var history=chat.messages.concat([{role:"user",content:text}]); addUser(text);
  var done;
  if(chat.mode==="both"){
    var els=addPair();
    done=streamAnswer(history,true,els[0]).then(function(a){ return streamAnswer(history,false,els[1]).then(function(){ return a; }); });
  } else {
    var el=addBot();
    done=streamAnswer(history,chat.mode!=="base",el);
  }
  scrollChat();
  return done.then(function(answer){ if(answer) chat.messages=history.concat([{role:"assistant",content:answer}]); else chat.messages=history.slice(0,-1);
    }).then(function(){ chat.busy=false; chat.ctrl=null; $("#chat-send").hidden=false; $("#chat-stop").hidden=true; $("#chat-input").focus(); });
}
$("#chat-form").addEventListener("submit", function(e){
  e.preventDefault(); var box=$("#chat-input"), text=box.value.trim();
  if(!text||chat.busy||!chat.job) return; box.value=""; ask(text);
});
$("#chat-input").addEventListener("keydown", function(e){ if(e.key==="Enter"&&!e.shiftKey){ e.preventDefault(); $("#chat-form").requestSubmit(); } });
$("#chat-stop").addEventListener("click", function(){ if(chat.ctrl) chat.ctrl.abort(); });
$("#chat-clear").addEventListener("click", function(){ if(chat.ctrl) chat.ctrl.abort(); chat.messages=[]; $("#chat-msgs").innerHTML=""; });
$("#chat-mode").addEventListener("click", function(e){ var b=e.target.closest("button"); if(b&&!b.disabled) setMode(b.getAttribute("data-m")); });
$("#chat-jobsel").addEventListener("change", function(e){ chat.job=e.target.value; chat.messages=[]; $("#chat-msgs").innerHTML=""; renderChat(); });
$("#jobs").addEventListener("toggle", function(e){ var d=e.target; if(d.classList&&d.classList.contains("nodebox")) openNodes[d.getAttribute("data-job")]=d.open; }, true);
$("#runs").addEventListener("click", function(e){
  var s=e.target.closest("[data-start]"); if(s) openStart(s.getAttribute("data-start"), s.getAttribute("data-run"));
});

/* ---- start a job ---- */
var L = {opts:null, kind:"train", run:"", busy:false};
function api(path,opts){ return fetch(path,opts).then(function(r){ return r.json().then(function(j){ return {ok:r.ok,j:j}; }); }); }
function planFor(n){
  var rules=(L.opts.partitions||[]).slice().sort(function(a,b){return a.max_nodes-b.max_nodes});
  for(var i=0;i<rules.length;i++){ if(n<=rules[i].max_nodes) return rules[i]; }
  return null;
}
function val(id){ var e=$("#"+id); return e?e.value.trim():""; }
function optionsFor(runs){
  if(!runs.length) return '<option value="">no run has a saved adapter yet</option>';
  return runs.map(function(r){ return '<option value="'+esc(r.run_id)+'"'+(r.run_id===L.run?" selected":"")+'>'+esc(r.name)+' ('+esc(r.run_id.slice(0,8))+')</option>'; }).join("");
}
function renderStart(){
  var o=L.opts, k=L.kind, h="";
  Array.prototype.forEach.call($("#startkind").querySelectorAll("button"), function(b){ b.setAttribute("aria-pressed", b.getAttribute("data-k")===k?"true":"false"); });
  var note=$("#startnote"); note.hidden=o.enabled; note.textContent=o.reason||"";
  $("#startcheck").disabled=$("#startgo").disabled=!o.enabled||L.busy;
  $("#startout").hidden=true;
  if(k==="train"){
    var rs=o.recipes||[], ready=rs.filter(function(r){return r.ready}), blocked=rs.filter(function(r){return !r.ready&&!r.tokenize});
    h+='<div class="fld wide"><label for="f-recipe">Recipe</label><select id="f-recipe">'+
      (ready.length?ready.map(function(r){return '<option value="'+esc(r.file)+'">'+esc(r.file)+' – '+esc(r.model||"?")+', '+esc(r.method)+(r.strategy?', '+esc(r.strategy):'')+'</option>'}).join(""):'<option value="">no ready recipe found</option>')+
      blocked.map(function(r){return '<option value="" disabled>'+esc(r.file)+' – '+esc(r.why)+'</option>'}).join("")+'</select>'+
      (ready.length?'':'<div class="hint">Put a recipe in the <code>recipes</code> folder next to submit.sh, or in a folder listed in $BHASKERA_RECIPES.</div>')+'</div>'+
      '<div class="fld"><label for="f-nodes">Nodes</label><input id="f-nodes" type="number" min="1" max="'+o.max_nodes+'" value="'+(L.nodes||2)+'"><div class="hint" id="f-plan"></div></div>'+
      '<div class="fld"><label for="f-hours">Hours (optional)</label><input id="f-hours" type="number" min="1" max="'+o.max_hours+'" value="'+(L.hours||"")+'"></div>';
  } else {
    h+='<div class="fld wide"><label for="f-run">Trained run</label><select id="f-run">'+optionsFor(o.runs||[])+'</select></div>';
    if(k==="eval") h+='<div class="fld"><label for="f-tasks">Benchmarks</label><input id="f-tasks" type="text" value="arc_easy,hellaswag"></div>'+
      '<div class="fld"><label for="f-limit">Examples per task (blank: all)</label><input id="f-limit" type="number" min="1" value="200"></div>';
    h+='<div class="fld"><label for="f-hours">Hours</label><input id="f-hours" type="number" min="1" max="'+o.max_hours+'" value="'+(o.default_hours||2)+'"></div>'+
      '<div class="fld check" style="align-self:end"><label><input id="f-4bit" type="checkbox"> Load the model in 4-bit (as QLoRA trained it)</label></div>';
  }
  $("#startfields").innerHTML=h; showPlan();
}
function showPlan(){
  var n=parseInt(val("f-nodes"),10), el=$("#f-plan"); if(!el) return;
  if(!n){ el.textContent=""; return; }
  var p=planFor(n), g=L.opts.gpus_per_node;
  el.textContent=(g?(n*g)+" GPUs. ":"")+(p?"Goes to "+p.partition+(p.max_hours?" (up to "+p.max_hours+" h).":"."):"");
}
function formData(action){
  var f={action:action,kind:L.kind};
  if(L.kind==="train"){ f.recipe=val("f-recipe"); f.nodes=val("f-nodes"); f.hours=val("f-hours"); }
  else { f.run=val("f-run"); f.hours=val("f-hours"); f.four_bit=$("#f-4bit").checked; if(L.kind==="eval"){ f.tasks=val("f-tasks"); f.limit=val("f-limit"); } }
  return f;
}
function showResult(cls,title,body){
  var o=$("#startout"); o.hidden=false; o.className="result "+cls; o.innerHTML="<b>"+esc(title)+"</b>"+(body?"\n"+esc(body):""); o.scrollIntoView({block:"nearest"});
}
function send(action){
  if(L.busy) return; var f=formData(action);
  if(action==="start"&&L.kind==="train"){
    var p=planFor(parseInt(f.nodes,10));
    if(!window.confirm("Start "+f.recipe+" on "+f.nodes+" node(s)"+(L.opts.gpus_per_node?" ("+f.nodes*L.opts.gpus_per_node+" GPUs)":"")+(f.hours?" for up to "+f.hours+" h":"")+"?\nThis uses GPU hours of your project.")) return;
  }
  L.busy=true; $("#startcheck").disabled=$("#startgo").disabled=true; showResult("", action==="check"?"Checking …":"Submitting …", "");
  api("/api/launch",{method:"POST",headers:{"Content-Type":"application/json","X-Requested-With":"bhaskera"},body:JSON.stringify(f)}).then(function(r){
    if(r.j.error){ showResult("bad","Not started",r.j.error); return; }
    if(r.j.dry){ showResult(r.j.ok?"ok":"bad", r.j.ok?"Checks passed. Nothing was submitted.":"A check failed. Nothing was submitted.", r.j.output); return; }
    if(r.j.ok){ showResult("ok","Submitted job #"+r.j.job+". It appears under Jobs in a moment.", r.j.output); refresh(); setTimeout(function(){ if($("#startbox").open) $("#startbox").close(); }, 2500); }
    else showResult("bad","Not submitted",r.j.output);
  }).catch(function(){ showResult("bad","Not started","The page could not reach the hub."); }).then(function(){ L.busy=false; $("#startcheck").disabled=$("#startgo").disabled=!(L.opts&&L.opts.enabled); });
}
function openStart(kind,run){
  api("/api/launch",{headers:{"X-Requested-With":"bhaskera"}}).then(function(r){ L.opts=r.j; }).catch(function(){ L.opts={enabled:false,reason:"The page could not reach the hub."}; }).then(function(){
    L.kind=kind||L.kind; L.run=run||L.run; renderStart(); if(!$("#startbox").open) $("#startbox").showModal();
  });
}
$("#startbtn").addEventListener("click", function(){ openStart("train"); });
$("#startclose").addEventListener("click", function(){ $("#startbox").close(); });
$("#startkind").addEventListener("click", function(e){ var b=e.target.closest("button"); if(b){ L.kind=b.getAttribute("data-k"); renderStart(); } });
$("#startfields").addEventListener("input", function(e){ if(e.target.id==="f-nodes") showPlan(); });
$("#startcheck").addEventListener("click", function(){ send("check"); });
$("#startgo").addEventListener("click", function(){ send("start"); });

refresh(); setInterval(refresh, 8000); setInterval(clock, 1000);
})();
</script>
</body>
</html>
"""
