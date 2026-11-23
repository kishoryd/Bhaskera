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
dialog{width:min(980px,96vw);height:min(860px,92vh);display:none;flex-direction:column;overflow:hidden;padding:0;border:1px solid var(--ink);border-radius:3px;background:var(--bg);color:var(--text);box-shadow:6px 6px 0 rgba(31,28,23,.18)}
dialog[open]{display:flex}
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
.dlg-foot{display:flex;gap:10px;align-items:center;justify-content:flex-end;padding:12px 22px;border-top:1px solid var(--border);flex:none;background:var(--bg)}
.dlg-foot .ssum{margin:0 auto 0 0;flex:1;min-width:0;padding:6px 11px;font-size:12.5px}
.h2row{display:flex;align-items:baseline;gap:14px;margin-top:30px}.h2row h2{margin:0}.h2row .btn{margin:0;align-self:center}.h2row .muted{font-size:13px}
.nodebox{margin:4px 0 14px}.nodebox>summary,.opt>summary{cursor:pointer;color:var(--muted);font-size:12.5px}
.nodebox>summary:hover,.opt>summary:hover{color:var(--text)}
.opt{position:relative}.opt>div{position:absolute;right:0;z-index:2;display:flex;flex-direction:column;gap:10px;padding:12px 14px;margin-top:6px;background:var(--bg);border:1px solid var(--ink);border-radius:3px;min-width:230px}
/*tabs*/.choices{display:flex;gap:2px;padding:0 22px;border-bottom:1px solid var(--border);overflow-x:auto;flex:none}
/*tab*/.choice{border:0;background:transparent;padding:11px 14px 10px;border-bottom:2px solid transparent;margin-bottom:-1px;white-space:nowrap}
/*tabs*/.choice b{font-size:14px;font-weight:500;color:var(--muted)}.choice span{display:none}
/*tabs*/.choice:hover b{color:var(--text)}
/*tabs*/.choice[aria-pressed=true]{border-bottom-color:var(--accent)}.choice[aria-pressed=true] b{color:var(--ink);font-weight:600}
.startdesc{padding:10px 22px 0;color:var(--muted);font-size:13px;flex:none}
.dlgbody{flex:1;overflow:auto;padding-bottom:10px}
.sec{grid-column:1/-1;font:500 15px var(--serif);margin:14px 0 -4px;padding-top:12px;border-top:1px solid var(--border)}
.sec:first-child{border-top:0;margin-top:0;padding-top:0}
.sfields{padding:14px 22px 0;display:grid;grid-template-columns:1fr 1fr;gap:16px 22px}.sfields>.srow{grid-column:1/-1}.sfields>.srow.half{grid-column:auto}#f-resume{padding:0;display:flex;flex-direction:column;gap:16px}
@media(max-width:700px){.sfields{grid-template-columns:1fr}.sfields>.srow.half{grid-column:1/-1}}
.srow label.lbl{display:block;font-size:11px;color:var(--muted);margin-bottom:5px;font-weight:600;text-transform:uppercase;letter-spacing:.07em}
.srow select,.srow input[type=number],.srow input[type=text]{border:1px solid var(--border);background:var(--surface);color:var(--text);border-radius:3px;padding:7px 10px;font:inherit}
.srow select{width:100%}.srow input[type=number]{width:90px}.srow input[type=text]{width:100%}
.srow select:focus,.srow input:focus{outline:0;border-color:var(--ink)}
.srow .sub{font-size:12.5px;color:var(--muted);margin-top:5px}
.chips{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.chipb{border:1px solid var(--border);background:transparent;border-radius:3px;padding:4px 11px;font-size:13px}
.chipb:hover{border-color:var(--ink)}
.chipb[aria-pressed=true]{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.rlist{border:1px solid var(--border);border-radius:3px;max-height:210px;overflow:auto}
.ritem{display:flex;gap:10px;align-items:baseline;width:100%;text-align:left;border:0;border-bottom:1px solid var(--border);background:transparent;padding:8px 11px}
.ritem:last-child{border-bottom:0}
.ritem:hover{background:var(--surface)}
.ritem[aria-pressed=true]{background:var(--surface);box-shadow:inset 3px 0 0 var(--accent)}
.ritem b{font-weight:600}.ritem .m{margin-left:auto;font:12px var(--mono);color:var(--muted);white-space:nowrap}
.srow details summary{cursor:pointer;color:var(--muted);font-size:13px}
#f-setlist{max-height:340px;overflow:auto;border:1px solid var(--border);border-radius:3px;padding:4px 10px}
.setgroup{margin:10px 0}.settitle{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px}
.setrow{display:grid;grid-template-columns:minmax(150px,1fr) minmax(120px,1fr);gap:4px 12px;align-items:center;padding:5px 0;border-top:1px solid var(--border)}
.setrow .sub{grid-column:1/-1;margin:0}.setrow input,.setrow select{width:100%}
.srow details label{display:flex;gap:8px;align-items:center;margin-top:8px;font-size:13px}
.ssum{margin:14px 22px 0;padding:10px 13px;border-left:3px solid var(--ink);background:var(--surface);font-size:13px;min-height:20px}
.ssum.warn{border-left-color:var(--warn);background:var(--warn-soft)}
.ssum:empty{display:none}
.toast{position:fixed;left:50%;top:78px;transform:translateX(-50%);z-index:20;background:var(--ink);color:var(--paper);padding:10px 18px;border-radius:3px;font-size:13.5px;box-shadow:4px 4px 0 rgba(31,28,23,.2)}
.job.flash{animation:fl 2.4s ease-out}@keyframes fl{0%{background:var(--warn-soft)}100%{background:transparent}}
.btn.confirm{background:var(--accent);border-color:var(--accent);color:#fff}
.attached{display:flex;flex-wrap:wrap;gap:8px;padding:10px 0 0;align-items:center}
.att{display:inline-flex;align-items:center;gap:8px;border:1px solid var(--border);border-radius:3px;padding:3px 4px 3px 8px;font-size:12.5px;background:var(--surface)}
.att img{height:34px;width:34px;object-fit:cover;border-radius:2px}
.att button{border:0;background:transparent;font-size:15px;line-height:1;padding:2px 6px;color:var(--muted)}
.att button:hover{color:var(--bad)}
.msg .files{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:6px}
.msg .files img{max-height:130px;max-width:200px;border-radius:2px;border:1px solid rgba(255,255,255,.25)}
.msg .files span{font:12px var(--mono);opacity:.85;border:1px solid rgba(255,255,255,.35);padding:1px 7px;border-radius:2px}
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
          <button class="btn small" id="chat-close" type="button" title="Stops the chat server and frees its GPU node">Close server</button>
        </div>
      </div>
      <div class="msgs" id="chat-msgs" aria-live="polite"></div>
      <div class="attached" id="chat-files" hidden></div>
      <form class="composer" id="chat-form">
        <button class="btn" id="chat-attachbtn" type="button" hidden title="Attach an image, a video or an audio file">Attach</button>
        <input id="chat-file" type="file" multiple hidden>
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
  <section id="cmpsec" hidden>
    <h2>Compare models</h2>
    <div class="card" id="cmp" style="padding:14px 16px"></div>
  </section>
  <details class="more" id="morebox"><summary>Cluster limits and node health <span id="morewarn"></span></summary><div id="cluster"></div></details>
  <footer id="foot"></footer>
</main>

<dialog id="startbox">
  <div class="dlg-head"><b>Start a job</b><span class="spacer" style="flex:1"></span><button class="btn" id="startclose" type="button">Close</button></div>
  <div class="choices" id="startkind" role="tablist" aria-label="What to start">
    <button type="button" class="choice" data-k="recipe"><b>Create recipe</b><span>Describe a training run in a few choices, or write it yourself</span></button>
    <button type="button" class="choice" data-k="train"><b>Train</b><span>Fine-tune with one of your recipes</span></button>
    <button type="button" class="choice" data-k="resume"><b>Resume</b><span>Continue a stopped run from its last checkpoint</span></button>
    <button type="button" class="choice" data-k="serve"><b>Chat</b><span>Talk to the tuned model, the base model, or both</span></button>
    <button type="button" class="choice" data-k="eval"><b>Benchmark</b><span>Test one or more models, with the standard tests or your own</span></button>
    <button type="button" class="choice" data-k="register"><b>Register</b><span>Keep a trained model under a name in the model registry</span></button>
  </div>
  <div class="startdesc" id="startdesc"></div>
  <div class="dlgbody">
    <div id="startnote" class="startnote" hidden></div>
    <div class="sfields" id="startfields"></div>
    <pre class="result" id="startout" hidden></pre>
  </div>
  <div class="dlg-foot"><div class="ssum" id="startsum"></div><button class="btn" id="startcheck" type="button" title="Runs the partition limits and the recipe check; submits nothing">Check first</button>
    <button class="btn primary" id="startgo" type="button">Start</button></div>
</dialog>
<div class="toast" id="toast" role="status" hidden></div>

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
var armed={id:null,until:0};
function cancelBtn(j,label,cls){
  var on=armed.id===String(j.id)&&Date.now()<armed.until;
  return '<button class="btn '+(cls||"")+(on?" confirm":"")+'" data-cancel="'+esc(j.id)+'" data-label="'+esc(label)+'" title="Stops this job on the cluster">'+(on?"Click again to "+label.toLowerCase():label)+'</button>';
}
function serviceCard(j){
  var running=(j.state==="RUNNING"||j.state==="COMPLETING"), s=j.serve||{}, serve=j.kind==="serve";
  var h='<article class="card job" data-job-card="'+esc(j.id)+'"><div class="job-head">'+badge(j.state)+'<span class="job-name">'+esc(j.name)+'</span><span class="muted mono">#'+esc(j.id)+'</span>'+
        '<div class="meta"><span class="chip">'+(serve?"chat server":"benchmark")+'</span><span class="chip">'+esc(j.partition)+'</span>'+
        (!running&&j.reason?'<span class="chip">'+esc(j.reason)+'</span>':"")+'</div></div>';
  if(!running) return h+'<div class="actions" style="margin-top:12px">'+cancelBtn(j,"Cancel job")+'</div></article>';
  var used=j.limit_s?Math.min(100,Math.round(100*j.elapsed_s/j.limit_s)):null;
  h+='<div class="progress"><div class="row"><span>'+(serve?(s.ready?'<b>Ready</b> · '+esc(s.model||"")+(s.adapter?' + adapter '+esc(s.adapter):''):'loading the model (a few minutes)'):'running the benchmarks')+'</span><span>'+dur(j.elapsed_s)+(j.limit_s?' of '+dur(j.limit_s):'')+'</span></div>'+
     (used!=null?'<div class="bar thin'+(used>=85?" hot":"")+'"><i style="width:'+used+'%"></i></div>':"")+'</div><div class="actions">'+
     (serve&&s.ready?'<a class="btn primary" href="#chat">'+ICON.flask+'Open the chat</a>':"")+
     '<button class="btn" data-log="'+esc(j.id)+'" data-name="'+esc(j.name)+'">'+ICON.log+'Logs</button>'+
     cancelBtn(j,serve?"Close server":"Cancel job")+'</div></article>';
  return h;
}
function jobCard(j){
  if(j.kind==="serve"||j.kind==="eval") return serviceCard(j);
  var t=j.train||{}, running=(j.state==="RUNNING"||j.state==="COMPLETING");
  var base="http://job-"+j.id+".localhost:"+PORT+"/";
  var charts="http://grafana-"+j.id+".localhost:"+PORT+"/d/rayDefaultDashboard?orgId=1&from=now-1h&to=now&refresh=10s";
  var h='<article class="card job" data-job-card="'+esc(j.id)+'"><div class="job-head">'+badge(j.state)+'<span class="job-name">'+esc(j.name)+'</span>'+
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
    return h+'<div class="actions" style="margin-top:12px">'+cancelBtn(j,"Cancel job")+'</div></article>';
  }
  h+='<div class="actions">';
  if(running){
    h+=link(j.linked?base:null,"Ray Dashboard","ray",true)+link(j.linked?charts:null,"GPU charts","chart")+
       '<button class="btn" data-log="'+esc(j.id)+'" data-name="'+esc(j.name)+'">'+ICON.log+'Logs</button>';
  }
  h+=link(j.run?runUrl(j.run):(running?MLFLOW:null),j.run?"MLflow run":"MLflow","flask")+cancelBtn(j,"Cancel job")+'</div></article>';
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
    var act=(r.has_adapter&&r.status!=="RUNNING") ? '<div class="runact"><button class="btn small primary" data-start="serve" data-run="'+esc(r.run_id)+'">Chat</button><button class="btn small" data-start="eval" data-run="'+esc(r.run_id)+'">Benchmark</button></div>' :
          ((r.status==="FAILED"||r.status==="KILLED"||r.stale)&&r.job ? '<div class="runact"><button class="btn small" data-start="resume" data-run="'+esc(r.run_id)+'">Resume</button></div>' : "");
    return '<tr><td><a class="run" href="'+esc(runUrl(r))+'" target="_blank" rel="noopener">'+esc(r.name)+'</a>'+(r.group?' <span class="chip">'+esc(r.group)+'</span>':"")+'</td>'+
      '<td>'+runBadge(r)+'</td><td class="muted">'+ago(r.started)+'</td>'+(hasEval?'<td class="num">'+fx(r.eval_loss,3)+'</td>':'')+(hasBench?'<td>'+bench(r)+'</td>':'')+'<td>'+act+'</td></tr>';
  }).join("") : '<tr><td colspan="6" class="muted" style="text-align:center;padding:26px">'+(all.length?"No runs match.":"No runs yet.")+'</td></tr>';
}

/* ---- compare benchmarked models ---- */
var cmpSel={};
function renderCompare(){
  var runs=(state.runs||[]).filter(function(r){return r.lm_scores&&Object.keys(r.lm_scores).length});
  $("#cmpsec").hidden=!runs.length; if(!runs.length) return;
  var chosen=runs.filter(function(r){return cmpSel[r.run_id]});
  var keys=[]; chosen.forEach(function(r){ Object.keys(r.lm_scores).forEach(function(k){ if(keys.indexOf(k)<0) keys.push(k); }); }); keys.sort();
  var h='<div class="chips">'+runs.map(function(r){ return '<button type="button" class="chipb" data-cmp="'+esc(r.run_id)+'" aria-pressed="'+(cmpSel[r.run_id]?"true":"false")+'">'+esc(r.name)+'</button>'; }).join("")+'</div>';
  if(!chosen.length) h+='<div class="sub" style="margin-top:8px">Pick the models to put side by side. Each was benchmarked on its own with Benchmark; the score of the base model it started from is shown next to its own.</div>';
  else {
    var pct=function(v){ return v==null?"–":(100*v).toFixed(1); };
    h+='<div class="scroll" style="margin-top:10px"><table><thead><tr><th>Test</th>'+chosen.map(function(r){return '<th class="num">'+esc(r.name)+'<div class="muted" style="font-weight:400">tuned (base)</div></th>'}).join("")+'</tr></thead><tbody>'+
      keys.map(function(k){
        var best=-1, vals=chosen.map(function(r){ var x=r.lm_scores[k]; return x&&x.tuned!=null?x.tuned:null; });
        vals.forEach(function(v){ if(v!=null&&v>best) best=v; });
        return '<tr><td>'+esc(k.replace(/_/g," "))+'</td>'+chosen.map(function(r,i){ var x=r.lm_scores[k]||{}; var top=vals[i]!=null&&vals[i]===best&&chosen.length>1;
          return '<td class="num"'+(top?' style="font-weight:600"':"")+'>'+pct(x.tuned)+' <span class="muted">('+pct(x.base)+')</span></td>'; }).join("")+'</tr>'; }).join("")+'</tbody></table></div>'+
      '<div class="sub" style="margin-top:6px">Scores are percent. The best tuned score in each row is bold. Models with different base models are comparable only on the same test and the same number of examples.</div>';
  }
  $("#cmp").innerHTML=h;
}
$("#cmp").addEventListener("click", function(e){ var b=e.target.closest("[data-cmp]"); if(!b) return; var id=b.getAttribute("data-cmp"); cmpSel[id]=!cmpSel[id]; renderCompare(); });

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
  if(part) bits.push(part.submitted+" of "+part.max_submitted+" job slots used in "+part.partition);
  $("#stats").textContent=bits.join("  ·  ");
  var order={RUNNING:0,COMPLETING:1,PENDING:2};
  var jobs=(s.jobs||[]).slice().sort(function(a,b){ return ((order[a.state]||0)-(order[b.state]||0)) || (Number(b.id)-Number(a.id)); });
  $("#jobcount").textContent=jobs.length;
  $("#jobs").innerHTML=jobs.length ? jobs.map(jobCard).join("") : emptyJobs();
  renderRuns(); renderCompare(); renderSide(); renderChat();
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
var chat = {job:null, messages:[], busy:false, ctrl:null, mode:"tuned", media:[], pending:[]};
var MEDIA_LIMIT={image:12,audio:25,video:48}, MEDIA_TOTAL=56;       // MB, as the chat server checks them
function mediaKind(f){ var t=(f.type||"").split("/")[0]; return t==="image"||t==="video"||t==="audio" ? t : ""; }
function renderAttached(){
  var box=$("#chat-files"); box.hidden=!chat.pending.length;
  box.innerHTML=chat.pending.map(function(a,i){
    return '<span class="att">'+(a.kind==="image"?'<img src="'+esc(a.data)+'" alt="">':'')+esc(a.kind)+': '+esc(a.name)+' <button type="button" data-rm="'+i+'" aria-label="Remove">&times;</button></span>';
  }).join("");
}
function attachFiles(files){
  var total=chat.pending.reduce(function(s,a){return s+a.size},0), errs=[];
  var reads=Array.prototype.map.call(files, function(f){
    var kind=mediaKind(f);
    if(!kind||chat.media.indexOf(kind)<0){ errs.push(f.name+": this model reads "+(chat.media.join(", ")||"text only")); return null; }
    if(f.size>MEDIA_LIMIT[kind]*1048576){ errs.push(f.name+" is larger than "+MEDIA_LIMIT[kind]+" MB"); return null; }
    if((total+=f.size)>MEDIA_TOTAL*1048576){ errs.push("the files add up to more than "+MEDIA_TOTAL+" MB"); return null; }
    return new Promise(function(ok){ var r=new FileReader(); r.onload=function(){ ok({kind:kind,name:f.name,size:f.size,data:r.result}); }; r.onerror=function(){ ok(null); }; r.readAsDataURL(f); });
  }).filter(Boolean);
  Promise.all(reads).then(function(got){ got.forEach(function(a){ if(a) chat.pending.push(a); }); renderAttached(); if(errs.length) toast(errs.join(". ")); });
}
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
  chat.media=(cur.serve.modalities||[]).filter(function(m){return m!=="text"});
  $("#chat-attachbtn").hidden=!chat.media.length;
  $("#chat-file").accept=chat.media.map(function(m){return m+"/*"}).join(",");
  if(!chat.media.length&&chat.pending.length){ chat.pending=[]; renderAttached(); }
  var hasAdapter=!!cur.serve.adapter;
  Array.prototype.forEach.call($("#chat-mode").querySelectorAll("button"), function(b){ if(b.getAttribute("data-m")!=="tuned") b.disabled=!hasAdapter; });
  if(!hasAdapter) setMode("tuned");
}
function setMode(m){
  chat.mode=m;
  Array.prototype.forEach.call($("#chat-mode").querySelectorAll("button"), function(b){ b.setAttribute("aria-pressed", b.getAttribute("data-m")===m?"true":"false"); });
}
function addUser(text,files){
  var d=document.createElement("div"); d.className="msg user";
  if(files&&files.length){
    var f=document.createElement("div"); f.className="files";
    files.forEach(function(a){ if(a.kind==="image"){ var im=document.createElement("img"); im.src=a.data; im.alt=a.name; f.appendChild(im); } else { var s=document.createElement("span"); s.textContent=a.kind+": "+a.name; f.appendChild(s); } });
    d.appendChild(f);
  }
  var t=document.createElement("div"); t.textContent=text; d.appendChild(t); $("#chat-msgs").appendChild(d); return d;
}
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
  var files=chat.pending; chat.pending=[]; renderAttached();
  var content = files.length ? [{type:"text",text:text}].concat(files.map(function(a){ return {type:a.kind,data:a.data}; })) : text;
  var history=chat.messages.concat([{role:"user",content:content}]); addUser(text,files);
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
  if((!text&&!chat.pending.length)||chat.busy||!chat.job) return;
  if(!text) text="Describe this."; box.value=""; ask(text);
});
$("#chat-input").addEventListener("keydown", function(e){ if(e.key==="Enter"&&!e.shiftKey){ e.preventDefault(); $("#chat-form").requestSubmit(); } });
$("#chat-stop").addEventListener("click", function(){ if(chat.ctrl) chat.ctrl.abort(); });
$("#chat-clear").addEventListener("click", function(){ if(chat.ctrl) chat.ctrl.abort(); chat.messages=[]; chat.pending=[]; renderAttached(); $("#chat-msgs").innerHTML=""; });
$("#chat-attachbtn").addEventListener("click", function(){ $("#chat-file").click(); });
$("#chat-file").addEventListener("change", function(e){ attachFiles(e.target.files); e.target.value=""; });
$("#chat-files").addEventListener("click", function(e){ var b=e.target.closest("[data-rm]"); if(b){ chat.pending.splice(parseInt(b.getAttribute("data-rm"),10),1); renderAttached(); } });
$("#chat-mode").addEventListener("click", function(e){ var b=e.target.closest("button"); if(b&&!b.disabled) setMode(b.getAttribute("data-m")); });
$("#chat-jobsel").addEventListener("change", function(e){ chat.job=e.target.value; chat.messages=[]; $("#chat-msgs").innerHTML=""; renderChat(); });
$("#jobs").addEventListener("toggle", function(e){ var d=e.target; if(d.classList&&d.classList.contains("nodebox")) openNodes[d.getAttribute("data-job")]=d.open; }, true);
$("#runs").addEventListener("click", function(e){
  var s=e.target.closest("[data-start]"); if(s) openStart(s.getAttribute("data-start"), s.getAttribute("data-run"));
});

/* ---- cancel a job, close a chat server ---- */
function doCancel(id,label,btn){
  var now=Date.now();
  if(!(armed.id===String(id)&&now<armed.until)){            // the first click only arms the button
    armed={id:String(id),until:now+5000}; if(btn){ btn.classList.add("confirm"); btn.textContent="Click again to "+label.toLowerCase(); }
    setTimeout(function(){ if(armed.id===String(id)&&Date.now()>=armed.until){ armed={id:null,until:0}; refresh(); } },5200); return;
  }
  armed={id:null,until:0}; if(btn){ btn.disabled=true; btn.textContent="Cancelling …"; }
  api("/api/launch",{method:"POST",headers:{"Content-Type":"application/json","X-Requested-With":"bhaskera"},body:JSON.stringify({action:"start",kind:"cancel",job:String(id)})}).then(function(r){
    if(r.j.error){ toast(r.j.error); if(btn){ btn.disabled=false; btn.textContent=label; } return; }
    toast(r.j.ok?("Job #"+id+" cancelled. It disappears from this page in a few seconds."):("Could not cancel job #"+id+": "+r.j.output)); refresh();
  }).catch(function(){ toast("The page could not reach the hub."); if(btn){ btn.disabled=false; btn.textContent=label; } });
}
$("#jobs").addEventListener("click", function(e){ var b=e.target.closest("[data-cancel]"); if(b) doCancel(b.getAttribute("data-cancel"),b.getAttribute("data-label")||"Cancel job",b); });
$("#chat-close").addEventListener("click", function(){ if(chat.job) doCancel(chat.job,"Close server",$("#chat-close")); });

/* ---- start a job ---- */
var L = {opts:null, kind:"train", run:"", evalSel:[], own:{name:"",data:"",q:"question",a:"answer"}, shots:"", skipBase:false, rinfo:null, step:0, regname:"", alias:"",
  rc:{name:"my-run",kind:"text",mods:["image"],model:"",method:"qlora",train:"",val:"",steps:"300",lr:"1e-4",rank:"16",batch:"",accum:"",every:"",evals:"",hsdp:"",stages:"",micro:"",yaml:"",copy:false,base:"",settings:{},open:{},raw:{train:"",val:"",format:"chatml",seq:"",media:""}}, nodes:2, hours:"", tasks:{arc_easy:1,hellaswag:1}, quick:true, confirm:false, busy:false};
var COMMON_TASKS=["arc_easy","hellaswag","winogrande","mmlu","gsm8k","truthfulqa_mc2"];
function api(path,opts){ return fetch(path,opts).then(function(r){ return r.json().then(function(j){ return {ok:r.ok,j:j}; }); }); }
function planFor(n){
  var rules=(L.opts.partitions||[]).slice().sort(function(a,b){return a.max_nodes-b.max_nodes});
  for(var i=0;i<rules.length;i++){ if(n<=rules[i].max_nodes) return rules[i]; }
  return null;
}
function slotsFor(part){
  var p=(state&&state.partitions||[]).filter(function(x){return x.partition===part})[0];
  return p&&p.max_submitted!=null ? {free:Math.max(0,p.max_submitted-p.submitted), max:p.max_submitted} : null;
}
function val(id){ var e=$("#"+id); return e?e.value.trim():""; }
function chip(label,on,attr){ return '<button type="button" class="chipb" '+attr+' aria-pressed="'+(on?"true":"false")+'">'+esc(label)+'</button>'; }
function recipeName(f){ return f.replace(/\.ya?ml$/,""); }
function runInfo(id){ return (state&&state.runs||[]).filter(function(r){return r.run_id===id})[0]||{}; }
function runList(){
  var ok={}; (L.opts.runs||[]).forEach(function(r){ ok[r.run_id]=1; });
  var rs=(state&&state.runs||[]).filter(function(r){return ok[r.run_id]}), multi=L.kind==="eval";
  if(!rs.length) return '<div class="sub">No finished run has a saved adapter yet. Train with <code>logging.mlflow_log_adapter: true</code> in the recipe.</div>';
  if(!L.run||!ok[L.run]) L.run=rs[0].run_id;
  L.evalSel=L.evalSel.filter(function(id){return ok[id]}); if(multi&&!L.evalSel.length) L.evalSel=[L.run];
  var on=function(id){ return multi?L.evalSel.indexOf(id)>=0:id===L.run; };
  var h='<div class="rlist">'+rs.map(function(r){
    return '<button type="button" class="ritem" data-run="'+esc(r.run_id)+'" aria-pressed="'+(on(r.run_id)?"true":"false")+'"><b>'+esc(r.name)+'</b><span class="muted">'+ago(r.started)+'</span>'+
      '<span class="m">'+(r.eval_loss!=null?"eval loss "+fx(r.eval_loss,3):"")+(r.lm_eval&&Object.keys(r.lm_eval).length?" · benchmarked":"")+'</span></button>';
  }).join("")+'</div>';
  var refs=[]; (L.opts.models||[]).forEach(function(m){ (m.refs||[]).forEach(function(x){ if(ok[x.run_id]) refs.push({ref:x.ref,run:x.run_id,aliases:x.aliases||[],name:m.name}); }); });
  if(L.kind!=="register"&&refs.length) h+='<div class="sub" style="margin-top:8px">Or a registered model:</div><div class="chips" style="margin-top:4px">'+refs.map(function(x){
    return chip(x.ref+(x.aliases.length?" @"+x.aliases.join(" @"):""),on(x.run),'data-run="'+esc(x.run)+'"'); }).join("")+'</div>';
  if(multi) h+='<div class="sub" style="margin-top:8px">'+(L.evalSel.length>1?L.evalSel.length+" models chosen: one benchmark job each, then put them side by side under Compare models.":"Choose more than one to compare models: each gets its own job.")+'</div>';
  return h+'<input type="hidden" id="f-run" value="'+esc(L.run)+'">';
}
function hoursRow(opts,def){
  var cur=L.hours===""?def:L.hours;
  var inChips=opts.some(function(h){ return String(h)===String(cur); });
  return '<div class="chips">'+opts.map(function(h){ return chip(h===""?"Longest allowed":h+" h", String(cur)===String(h), 'data-h="'+h+'"'); }).join("")+
    '<span class="muted" style="margin:0 4px 0 8px">or</span><input id="f-hours" type="number" min="1" max="'+L.opts.max_hours+'" value="'+(inChips||cur===""?"":cur)+'" placeholder="hours"></div>';
}
function renderStart(){
  var o=L.opts, k=L.kind, h="";
  L.confirm=false;
  Array.prototype.forEach.call($("#startkind").querySelectorAll(".choice"), function(b){ b.setAttribute("aria-pressed", b.getAttribute("data-k")===k?"true":"false"); });
  var ds=$("#startkind .choice[data-k="+k+"] span"); $("#startdesc").textContent=ds?ds.textContent:"";
  var note=$("#startnote"); note.hidden=o.enabled; note.textContent=o.reason||"";
  $("#startout").hidden=true; $("#startgo").classList.remove("confirm"); $("#startgo").textContent=goLabel(k);
  if(k==="recipe"){
    h+=recipeForm(o);
  } else if(k==="train"){
    var rs=o.recipes||[], ready=rs.filter(function(r){return r.ready}), blocked=rs.filter(function(r){return !r.ready&&!r.tokenize});
    if(ready.length&&!ready.some(function(r){return r.file===L.recipe})) L.recipe=ready[0].file;
    h+='<div class="srow"><label class="lbl" for="f-recipe">Recipe</label>'+(ready.length ?
        '<select id="f-recipe">'+ready.map(function(r){return '<option value="'+esc(r.file)+'"'+(r.file===L.recipe?" selected":"")+'>'+esc(recipeName(r.file))+'</option>'}).join("")+'</select><div class="sub" id="f-recipe-info"></div>' :
        '<div class="sub">No recipe is ready. Put one in the <code>recipes</code> folder next to submit.sh, or in a folder listed in $BHASKERA_RECIPES.</div>')+
      (blocked.length?'<div class="sub">'+blocked.length+' more '+(blocked.length>1?"need":"needs")+' editing first (they still have <code>&lt;project&gt;/&lt;username&gt;</code> placeholders): '+blocked.slice(0,3).map(function(r){return esc(recipeName(r.file))}).join(", ")+(blocked.length>3?", …":"")+'</div>':'')+'</div>'+
      '<div class="srow"><label class="lbl">Nodes</label><div class="chips">'+[1,2,4,8,16,32].filter(function(n){return n<=o.max_nodes}).map(function(n){return chip(n,L.nodes===n,'data-n="'+n+'"')}).join("")+
        '<span class="muted" style="margin:0 4px 0 8px">or</span><input id="f-nodes" type="number" min="1" max="'+o.max_nodes+'" value="'+L.nodes+'"></div></div>'+
      '<div class="srow"><label class="lbl">How long</label>'+hoursRow(["",1,2,6,12,24],"")+'</div>';
  } else if(k==="resume"){
    h+='<div class="srow"><label class="lbl">Stopped run</label>'+stoppedList()+'</div><div id="f-resume"></div>';
  } else if(k==="register"){
    h+='<div class="srow"><label class="lbl">Trained model</label>'+runList()+'</div>';
    if(!L.regname||L.regnameFor!==L.run){ L.regname=slug(runInfo(L.run).name||"model"); L.regnameFor=L.run; }
    var known=(o.models||[]);
    h+='<div class="srow"><label class="lbl" for="f-name">Registry name</label><input id="f-name" type="text" value="'+esc(L.regname)+'" maxlength="64">'+
       (known.length?'<div class="chips" style="margin-top:8px">'+known.map(function(m){return chip(m.name+" ("+m.versions+(m.versions===1?" version":" versions")+")",L.regname===m.name,'data-name="'+esc(m.name)+'"')}).join("")+'</div><div class="sub">Pick one to add a new version to it.</div>':'<div class="sub">The first version creates the model.</div>')+'</div>'+
       '<div class="srow"><label class="lbl" for="f-alias">Alias (optional)</label><input id="f-alias" type="text" list="f-aliases" value="'+esc(L.alias)+'" maxlength="32" placeholder="a name of your choice, for example the stage it is in">'+
       '<datalist id="f-aliases">'+aliasNames().map(function(a){return '<option value="'+esc(a)+'">'}).join("")+'</datalist>'+
       '<div class="sub">An alias is a label you choose that points at one version; move it to a newer version later. '+(aliasNames().length?"Already in use: "+esc(aliasNames().join(", "))+". ":"")+'Chat and benchmark accept <code>name@alias</code>.</div></div>';
  } else {
    h+='<div class="srow"><label class="lbl">Trained model</label>'+runList()+'</div>';
    if(k==="eval") h+='<div class="srow"><label class="lbl">Tests</label><div class="chips">'+COMMON_TASKS.map(function(t){return chip(t,!!L.tasks[t],'data-t="'+t+'"')}).join("")+'</div>'+
        (o.own_tests&&o.own_tests.length?'<div class="sub" style="margin-top:8px">Your own tests:</div><div class="chips" style="margin-top:4px">'+o.own_tests.map(function(t){return chip(t,!!L.tasks[t],'data-t="'+esc(t)+'"')}).join("")+'</div>':"")+
        '<div style="margin-top:8px"><input id="f-more" type="text" placeholder="other test names, separated by commas"></div>'+
        '<div style="margin-top:8px"><details data-own="1"><summary>Add a test of your own</summary><div style="margin-top:8px"><div class="sub">A JSONL file with one question and its right answer per line. It becomes a test you can tick above and use on any model.</div>'+
          '<input id="f-tname" type="text" value="'+esc(L.own.name)+'" placeholder="name of the test, for example my_support_questions" style="margin-top:8px">'+
          '<input id="f-tdata" type="text" value="'+esc(L.own.data)+'" placeholder="/scratch/<project>/<username>/data/my_test.jsonl" style="margin-top:6px">'+
          '<div class="chips" style="margin-top:8px"><label class="muted">Question field <input id="f-tq" type="text" value="'+esc(L.own.q)+'" style="width:110px"></label><label class="muted">Answer field <input id="f-ta" type="text" value="'+esc(L.own.a)+'" style="width:110px"></label>'+
          '<button class="btn small" id="f-tgo" type="button">Add the test</button></div></div></details></div>'+
        '<div class="sub">The test data must be downloaded once: <code>python -m bhaskera.launcher.evaluate prepare --tasks &lt;names&gt;</code></div></div>'+
      '<div class="srow"><label class="lbl">How much</label><div class="chips">'+chip("Quick check: 200 examples per test",L.quick,'data-q="1"')+chip("Full test",!L.quick,'data-q="0"')+'</div></div>';
    h+='<div class="srow"><label class="lbl">Stop after</label>'+hoursRow([1,2,4,8],o.default_hours||2)+'</div>'+
       '<div class="srow"><details'+(k==="eval"?' open':'')+'><summary>More options</summary>'+(k==="eval"?'<label><input id="f-samples" type="checkbox" checked> Keep every answer, to review the mistakes afterwards</label>':'')+'<label><input id="f-4bit" type="checkbox"> Load the model in 4-bit (as QLoRA trained it)</label>'+(k==="eval"?'<label><input id="f-skipbase" type="checkbox"'+(L.skipBase?" checked":"")+'> Only the tuned model: skip the base model (quicker, but nothing to compare with)</label><label>Examples shown in each question <input id="f-shots" type="number" min="0" max="50" value="'+esc(L.shots)+'" placeholder="test default" style="width:110px"></label>':"")+'</details></div>';
  }
  $("#startfields").innerHTML=h; fillInfo(); showSum();
  var none = k==="recipe" ? false : k==="train" ? !(o.recipes||[]).some(function(r){return r.ready}) : k==="resume" ? !(o.stopped||[]).length : !(o.runs||[]).length;
  $("#startcheck").disabled=$("#startgo").disabled=!o.enabled||L.busy||none;
  if(k==="resume") loadResume();
}
function cacheLabel(c){ return c.name+" · "+(c.model||"?")+" · sequence "+(c.seq_len||"?")+" · "+(c.rows!=null?Number(c.rows).toLocaleString()+" rows":""); }
function cacheOptions(list,sel,none){
  return (none?'<option value="">none (no evaluation during training)</option>':'')+list.map(function(c){ return '<option value="'+esc(c.path)+'"'+(c.path===sel?" selected":"")+'>'+esc(cacheLabel(c))+'</option>'; }).join("");
}
function aliasNames(){ var out=[]; (L.opts.models||[]).forEach(function(m){ (m.aliases||[]).forEach(function(a){ if(out.indexOf(a)<0) out.push(a); }); }); return out; }
function methodOk(o,m,kind){ var x=(o.methods||{})[m]; return !!x&&(x.kinds||["text","multimodal"]).indexOf(kind)>=0; }
function det(r,key,title,body){ return '<div class="srow"><details data-od="'+key+'"'+(r.open[key]?" open":"")+'><summary>'+title+'</summary>'+body+'</details></div>'; }
function settingRow(f,v){
  var h=f.help?esc(f.help):"", d=f.default===null||f.default===undefined?"":(typeof f.default==="object"?JSON.stringify(f.default):String(f.default)), id="s-"+f.key, ctl;
  if(f.type==="bool") ctl='<select data-set="'+esc(f.key)+'"><option value="">default ('+esc(d)+')</option><option value="true"'+(v==="true"?" selected":"")+'>true</option><option value="false"'+(v==="false"?" selected":"")+'>false</option></select>';
  else ctl='<input type="text" data-set="'+esc(f.key)+'" value="'+esc(v||"")+'" placeholder="'+(d===""?(f.optional?"not set":"empty"):"default "+esc(d))+'">';
  return '<div class="setrow" data-find="'+esc((f.key+" "+(f.help||"")).toLowerCase())+'"><div class="setkey"><code>'+esc(f.name)+'</code><span class="muted"> '+esc(f.type==="text-or-list"?"text or list":f.type)+'</span></div>'+ctl+(h?'<div class="sub">'+h+'</div>':"")+'</div>';
}
function allSettings(o,r){
  if(!r.open.all) return '<div class="sub">Every option of a recipe, with what it does. Leave a box empty to keep the value above.</div>';
  var n=Object.keys(r.settings).length;
  return '<div class="sub">Every option of a recipe. Leave a box empty to keep the value the recipe above sets; what you type here wins. <span id="f-setcount">'+(n?n+" changed":"")+'</span></div>'+
    '<input id="f-rfind" type="search" placeholder="Search the options, for example learning, offload, video" style="margin:8px 0">'+
    '<div id="f-setlist">'+(o.schema||[]).map(function(g){
      return '<div class="setgroup" data-section="'+esc(g.section)+'"><div class="settitle">'+esc(g.section)+'</div>'+g.fields.map(function(f){ return settingRow(f,r.settings[f.key]); }).join("")+'</div>';
    }).join("")+'</div>';
}
function customForm(o,r){
  var st=o.starters||[], rs=(o.recipes||[]).filter(function(x){return !x.tokenize});
  var opts='<option value="">Start from …</option>'+(st.length?'<optgroup label="Starters shipped with Bhaskera">'+st.filter(function(x){return !x.tokenize}).map(function(x){return '<option value="starter:'+esc(x.file)+'"'+(r.base==="starter:"+x.file?" selected":"")+'>'+esc(x.label)+'</option>'}).join("")+'</optgroup>':"")+
    (rs.length?'<optgroup label="Your recipes">'+rs.map(function(x){return '<option value="recipe:'+esc(x.file)+'"'+(r.base==="recipe:"+x.file?" selected":"")+'>'+esc(recipeName(x.file))+'</option>'}).join("")+'</optgroup>':"");
  return '<div class="srow"><label class="lbl" for="f-rbase">The recipe</label><div class="chips"><select id="f-rbase">'+opts+'</select><button class="btn small" id="f-rload" type="button">Load it here</button></div>'+
    '<textarea id="f-ryaml" rows="16" spellcheck="false" style="width:100%;margin-top:8px;font-family:var(--mono,monospace);font-size:12px" placeholder="model:\n  name: /path/to/model\ndata:\n  tokenized_path: /path/to/cache\ntraining:\n  max_steps: 300 …">'+esc(r.yaml)+'</textarea>'+
    '<div class="sub">The whole recipe as YAML. A starter or recipe loaded here is saved under your name above, with its own run name and checkpoint folder. Fill every <code>&lt;project&gt;</code> placeholder before saving.</div></div>';
}
function recipeForm(o){
  var r=L.rc, caches=o.caches||[], tr=caches.filter(function(c){return c.side!=="val"}), va=caches.filter(function(c){return c.side!=="train"});
  if(!r.train||!caches.some(function(c){return c.path===r.train})) r.train=(tr[0]||{}).path||"";
  if(r.val!==""&&!caches.some(function(c){return c.path===r.val})) r.val=(va.filter(function(c){return c.path!==r.train})[0]||{}).path||"";
  if(r.val===""&&L.rcValTouched!==true) r.val=(va.filter(function(c){return c.path!==r.train})[0]||{}).path||"";
  var methods=Object.keys(o.methods||{}).filter(function(m){return methodOk(o,m,r.kind)});
  if(methods.indexOf(r.method)<0) r.method=methods[0]||"fsdp";
  var multi=r.kind==="multimodal", custom=r.method==="custom";
  var h='<div class="sec">The model</div><div class="srow half"><label class="lbl" for="f-rname">Name of the recipe</label><input id="f-rname" type="text" value="'+esc(r.name)+'" maxlength="61"><div class="sub">Saved as <code>recipes/'+esc(r.name||"name")+'.yaml</code>; it also names the run in MLflow.</div></div>'+
    '<div class="srow half"><label class="lbl">Kind of model</label><div class="chips">'+chip("Text",!multi,'data-kind="text"')+chip("Multimodal (images, video, audio)",multi,'data-kind="multimodal"')+'</div></div>'+
    (multi?'<div class="srow"><label class="lbl">The data holds</label><div class="chips">'+["image","video","audio"].map(function(m){return chip(m,r.mods.indexOf(m)>=0,'data-mod="'+m+'"')}).join("")+'<span class="muted">besides text</span></div></div>':"")+
    '<div class="srow"><label class="lbl">Method</label><div class="chips">'+methods.map(function(m){return chip(o.methods[m].label,r.method===m,'data-method="'+m+'"')}).join("")+'</div><div class="sub">'+esc((o.methods[r.method]||{}).about||"")+'</div></div>';
  if(custom){ h+='<div class="sec">The recipe</div>'+customForm(o,r); }
  else {
    h+='<div class="srow"><label class="lbl" for="f-rmodel">Model folder</label><input id="f-rmodel" type="text" list="f-rmodels" value="'+esc(r.model)+'" placeholder="/scratch/<project>/<username>/models/'+(multi?"Qwen3-VL-8B-Instruct":"Qwen3-14B")+'">'+
      '<datalist id="f-rmodels">'+(o.model_folders||[]).map(function(m){return '<option value="'+esc(m)+'">'}).join("")+'</datalist><div class="sub">A downloaded Hugging Face model folder (compute nodes have no internet).</div></div>';
    h+='<div class="sec">The data</div>';
    if(caches.length){
      h+='<div class="srow half"><label class="lbl" for="f-rtrain">Training data</label><select id="f-rtrain">'+cacheOptions(tr,r.train,false)+'</select></div>'+
         '<div class="srow half"><label class="lbl" for="f-rval">Validation data</label><select id="f-rval">'+cacheOptions(va,r.val,true)+'</select></div>';
    } else {
      h+='<div class="srow"><div class="sub">No tokenised data was found yet. Open <b>My data is not tokenised yet</b> below to prepare your files; they appear here when that job has finished.</div></div>';
    }
    h+='<div class="sec">The training</div><div class="srow"><label class="lbl">How long</label><div class="chips">'+["100","300","1000","3000"].map(function(s){return chip(s+" steps",r.steps===s,'data-steps="'+s+'"')}).join("")+
       '<span class="muted" style="margin:0 4px 0 8px">or</span><input id="f-rsteps" type="number" min="1" value="'+(["100","300","1000","3000"].indexOf(r.steps)<0?esc(r.steps):"")+'" placeholder="steps"></div></div>'+
       '<div class="srow half"><label class="lbl">Learning rate</label><div class="chips">'+[["5e-5","5e-5"],["1e-4","1e-4 (usual)"],["2e-4","2e-4"]].map(function(x){return chip(x[1],r.lr===x[0],'data-lr="'+x[0]+'"')}).join("")+'</div></div>'+
       '<div class="srow half"><label class="lbl">LoRA rank</label><div class="chips">'+["8","16","32"].map(function(x){return chip(x,r.rank===x,'data-rank="'+x+'"')}).join("")+'</div><div class="sub">Larger learns more and uses a little more memory.</div></div>'+
       det(r,"more","More options",'<div class="chips" style="margin-top:8px">'+
         '<label class="muted">Batch size <input id="f-rbatch" type="number" min="1" value="'+esc(r.batch)+'" placeholder="auto"></label>'+
         '<label class="muted">Gradient accumulation <input id="f-raccum" type="number" min="1" value="'+esc(r.accum)+'" placeholder="1"></label>'+
         '<label class="muted">Save every <input id="f-revery" type="number" min="1" value="'+esc(r.every)+'" placeholder="auto"> steps</label>'+
         '<label class="muted">Evaluate every <input id="f-revals" type="number" min="0" value="'+esc(r.evals)+'" placeholder="auto"> steps</label>'+
         (r.method==="hsdp"?'<label class="muted">GPUs per shard group <input id="f-rhsdp" type="number" min="0" value="'+esc(r.hsdp)+'" placeholder="one node"></label>':"")+
         (r.method==="pipeline"?'<label class="muted">Stages <input id="f-rstages" type="number" min="2" value="'+esc(r.stages)+'" placeholder="4"></label><label class="muted">Micro-batches <input id="f-rmicro" type="number" min="1" value="'+esc(r.micro)+'" placeholder="2 x stages"></label>':"")+'</div>'+
         (r.method==="hsdp"||r.method==="pipeline"?'<div class="sub">'+(r.method==="hsdp"?"The shard group must divide the number of GPUs you start with.":"The number of GPUs must divide by the stages.")+' The check on the Train page tells you if it does not.</div>':""));
  }
  h+='<div class="sec">'+(custom?"Settings on top":"More")+'</div>'+det(r,"all","All settings ("+((o.schema||[]).reduce(function(n,g){return n+g.fields.length},0))+" options)",allSettings(o,r));
  if(!custom) h+=det(r,"raw","My data is not tokenised yet",'<div style="margin-top:8px">'+
       '<div class="sub">Give the JSONL files (one conversation per line'+(multi?", with the paths of the images, video or audio in each row":"")+'). A CPU job tokenises them once; then they show up as training data above.</div>'+
       '<input id="f-rawtrain" type="text" value="'+esc(r.raw.train)+'" placeholder="training file: /scratch/<project>/<username>/data/train.jsonl" style="margin-top:8px">'+
       '<input id="f-rawval" type="text" value="'+esc(r.raw.val)+'" placeholder="validation file (optional)" style="margin-top:6px">'+
       (multi?'<input id="f-rawmedia" type="text" value="'+esc(r.raw.media)+'" placeholder="folder the media paths in the file are relative to (optional)" style="margin-top:6px">':"")+
       '<div class="chips" style="margin-top:8px">'+["chatml","alpaca","sharegpt"].map(function(f){return chip(f,r.raw.format===f,'data-fmt="'+f+'"')}).join("")+
       '<label class="muted">Sequence length <input id="f-rawseq" type="number" min="128" value="'+esc(r.raw.seq)+'" placeholder="'+(multi?"1280":"4096")+'"></label>'+
       '<button class="btn small" id="f-rawgo" type="button">Prepare data</button></div></div>');
  return h;
}
function goLabel(k){ return k==="register"?"Register":k==="recipe"?"Create recipe":"Start"; }
function slug(t){ return String(t).toLowerCase().replace(/[^a-z0-9_.-]+/g,"-").replace(/^[^a-z0-9]+/,"").slice(0,64)||"model"; }
function stoppedList(){
  var rs=L.opts.stopped||[];
  if(!rs.length) return '<div class="sub">No run has stopped half way.</div>';
  if(!rs.some(function(r){return r.run_id===L.run})) L.run=rs[0].run_id;
  return '<div class="rlist">'+rs.map(function(r){
    var full=runInfo(r.run_id);
    return '<button type="button" class="ritem" data-run="'+esc(r.run_id)+'" aria-pressed="'+(r.run_id===L.run?"true":"false")+'"><b>'+esc(r.name)+'</b><span class="muted">'+(full.started?ago(full.started):"")+'</span><span class="m">'+esc(r.status.toLowerCase())+'</span></button>';
  }).join("")+'</div>';
}
function loadResume(){
  var run=L.run, box=$("#f-resume"); if(!box||!run) return;
  box.innerHTML='<div class="sub">Looking at what the run left behind …</div>'; L.rinfo=null; L.step=0;
  api("/api/resume?run="+encodeURIComponent(run),{headers:{"X-Requested-With":"bhaskera"}}).then(function(r){
    if(L.run!==run||L.kind!=="resume") return; L.rinfo=r.j; showResume();
  }).catch(function(){ if($("#f-resume")) $("#f-resume").innerHTML='<div class="sub">The page could not reach the hub.</div>'; });
}
function showResume(){
  var box=$("#f-resume"), i=L.rinfo; if(!box||!i) return;
  if(i.error){ box.innerHTML='<div class="sub" style="color:var(--bad)">'+esc(i.error)+'</div>'; return; }
  if(!i.resumable){ box.innerHTML='<div class="sub" style="color:var(--bad)">Cannot be resumed: '+esc(i.why)+'</div>'; $("#startcheck").disabled=$("#startgo").disabled=true; showSum(); return; }
  var steps=(i.steps||[]).slice(-6).reverse();
  box.innerHTML='<div class="srow"><div class="sub">'+esc(i.recipe)+' · '+i.nodes+(i.nodes===1?" node":" nodes")+' · '+i.hours+' h · checkpoints at step'+(i.steps.length>1?"s ":" ")+esc(i.steps.join(", "))+(i.max_steps?" of "+esc(i.max_steps):"")+'</div></div>'+
    '<div class="srow"><label class="lbl">Continue from</label><div class="chips">'+steps.map(function(s,k){ return chip(k===0?"Latest (step "+s+")":"Step "+s, (L.step||i.latest)===s, 'data-step="'+s+'"'); }).join("")+'</div>'+
    '<div class="sub">An earlier step is for going back before a problem such as a loss spike. The checkpoints after it are set aside, not deleted.</div></div>'+
    '<div class="srow"><label class="lbl">Nodes and time</label><div class="chips"><input id="f-nodes" type="number" min="1" max="'+L.opts.max_nodes+'" value="'+i.nodes+'"><span class="muted">nodes</span> <input id="f-hours" type="number" min="1" max="'+L.opts.max_hours+'" value="'+i.hours+'"><span class="muted">hours</span></div>'+
    '<div class="sub">The recipe is read again when the job starts: change it first to change something, for example a lower learning rate.</div></div>';
  $("#startcheck").disabled=$("#startgo").disabled=!L.opts.enabled; showSum();
}
function fillInfo(){
  var el=$("#f-recipe-info"); if(!el) return;
  var r=(L.opts.recipes||[]).filter(function(x){return x.file===val("f-recipe")})[0];
  el.textContent=r?[r.model,r.method,r.strategy].filter(Boolean).join(" · "):"";
}
function showSum(){
  var el=$("#startsum"); if(!el) return; var o=L.opts, g=o.gpus_per_node, bits=[], warn=false;
  if(L.kind==="recipe"){
    var rc=L.rc; el.className="ssum"; var lab=(o.methods&&o.methods[rc.method]?o.methods[rc.method].label:"")+(rc.kind==="multimodal"?" · multimodal":"");
    el.textContent=rc.method==="custom"?lab+" · the recipe is saved as you wrote it":lab+" · "+rc.steps+" steps · learning rate "+rc.lr+" · LoRA rank "+rc.rank+" · the adapter is saved, with a checkpoint every few hundred steps"; return;
  }
  if(L.kind==="register"){
    var rr=runInfo(L.run); el.className="ssum"; el.textContent=rr.name?"Registers the adapter of "+rr.name+" as a new version"+(L.alias?", and points "+L.alias+" at it":""):""; return;
  }
  if(L.kind==="resume"){
    var ri=L.rinfo; if(!ri||!ri.resumable){ el.className="ssum"; el.textContent=""; return; }
    var rn=parseInt(val("f-nodes"),10)||ri.nodes, rp=planFor(rn);
    bits.push("from step "+(L.step||ri.latest)); bits.push(rn+(rn===1?" node":" nodes")+(g?" · "+(rn*g)+" GPUs":""));
    if(rp){ bits.push(rp.partition); var rs=slotsFor(rp.partition); if(rs){ bits.push(rs.free+" of "+rs.max+" job slots free"); if(!rs.free) warn=true; } }
    el.className="ssum"+(warn?" warn":""); el.textContent=bits.join("  ·  "); return;
  }
  var n=L.kind==="train"?(parseInt(val("f-nodes"),10)||0):1;
  if(!n){ el.className="ssum"; el.textContent=""; return; }
  var p=planFor(n);
  bits.push(n+(n===1?" node":" nodes")+(g?" · "+(n*g)+" GPUs":""));
  if(p){ bits.push(p.partition); var sl=slotsFor(p.partition); if(sl){ bits.push(sl.free+" of "+sl.max+" job slots free"); if(!sl.free) warn=true; } }
  var hrs=L.hours===""?(p&&p.max_hours?p.max_hours:""):L.hours;
  if(L.kind==="train") bits.push(hrs?"up to "+hrs+" h":"time limit of the partition");
  else bits.push((L.hours||o.default_hours||2)+" h at most"+(L.kind==="serve"?"; loading takes a few minutes":""));
  el.className="ssum"+(warn?" warn":""); el.textContent=bits.join("  ·  ")+(warn?"  ·  no free slot: the job will be refused until one ends":"");
}
function tasksValue(){
  var t=Object.keys(L.tasks).filter(function(k){return L.tasks[k]}), more=val("f-more").split(",").map(function(x){return x.trim()}).filter(Boolean);
  return t.concat(more).join(",");
}
function formData(action){
  var f={action:action,kind:L.kind};
  if(L.kind==="recipe"){
    var r=L.rc; f.name=r.name; f.method=r.method; f.model_kind=r.kind; if(r.kind==="multimodal") f.modalities=r.mods;
    f.settings=r.settings;
    if(r.method==="custom"){ f.yaml=r.yaml; f.copy=r.copy; }
    else {
      f.model=r.model; f.train_cache=r.train; f.val_cache=r.val; f.steps=r.steps; f.lr=r.lr; f.rank=r.rank;
      if(r.batch) f.batch_size=r.batch; if(r.accum) f.grad_accum=r.accum; if(r.every) f.save_every_steps=r.every; if(r.evals!=="") f.eval_steps=r.evals;
      if(r.method==="hsdp"&&r.hsdp!=="") f.hsdp_shard_size=r.hsdp; if(r.method==="pipeline"){ if(r.stages) f.stages=r.stages; if(r.micro) f.microbatches=r.micro; }
    }
  }
  else if(L.kind==="register"){ f.run=val("f-run"); f.name=val("f-name"); f.alias=L.alias; }
  else if(L.kind==="resume"){ f.run=L.run; f.step=L.step||""; f.nodes=val("f-nodes"); f.hours=val("f-hours"); }
  else if(L.kind==="train"){ f.recipe=val("f-recipe"); f.nodes=val("f-nodes"); f.hours=val("f-hours")||L.hours; }
  else { f.run=val("f-run"); f.hours=val("f-hours")||L.hours||String(L.opts.default_hours||2); f.four_bit=$("#f-4bit").checked;
    if(L.kind==="eval"){ f.tasks=tasksValue(); f.limit=L.quick?"200":""; f.samples=$("#f-samples").checked; f.skip_base=$("#f-skipbase").checked; if($("#f-shots").value.trim()!=="") f.num_fewshot=$("#f-shots").value.trim(); } }
  return f;
}
function showResult(cls,title,body){
  var o=$("#startout"); o.hidden=false; o.className="result "+cls; o.innerHTML="<b>"+esc(title)+"</b>"+(body?"\n"+esc(body):""); o.scrollIntoView({block:"nearest"});
}
function toast(msg){ var t=$("#toast"); t.textContent=msg; t.hidden=false; clearTimeout(toast.t); toast.t=setTimeout(function(){ t.hidden=true; },6000); }
function send(action){
  if(L.busy) return;
  if(L.kind==="eval"&&L.evalSel.length>1){ sendEvalMany(action); return; }
  var f=formData(action);
  if(action==="start"&&(L.kind==="train"||L.kind==="resume")&&!L.confirm){
    L.confirm=true; var g=L.opts.gpus_per_node, n=parseInt(f.nodes,10)||0;
    $("#startgo").textContent="Yes, start on "+n+(n===1?" node":" nodes")+(g?" ("+n*g+" GPUs)":""); $("#startgo").classList.add("confirm");
    setTimeout(function(){ if(L.confirm){ L.confirm=false; $("#startgo").textContent="Start"; $("#startgo").classList.remove("confirm"); } },6000);
    return;
  }
  L.confirm=false; $("#startgo").textContent=goLabel(L.kind); $("#startgo").classList.remove("confirm");
  L.busy=true; $("#startcheck").disabled=$("#startgo").disabled=true; showResult("", action==="check"?"Checking …":"Submitting …", "");
  api("/api/launch",{method:"POST",headers:{"Content-Type":"application/json","X-Requested-With":"bhaskera"},body:JSON.stringify(f)}).then(function(r){
    if(r.j.error){ showResult("bad","Not started",r.j.error); return; }
    if(r.j.dry&&L.kind==="recipe"){ showResult("ok","This is the recipe that would be written. Nothing was saved yet.",r.j.output); return; }
    if(r.j.ok&&L.kind==="recipe"){
      L.recipe=r.j.recipe; toast("Created "+r.j.recipe+". It is selected under Train a model: choose the nodes and press Start.");
      L.kind="train"; L.step=0; api("/api/launch",{headers:{"X-Requested-With":"bhaskera"}}).then(function(o){ L.opts=o.j; renderStart(); }); return;
    }
    if(r.j.dry){ showResult(r.j.ok?"ok":"bad", r.j.ok?"Everything checks out. Nothing was submitted.":"A check failed. Nothing was submitted.", r.j.output); return; }
    if(r.j.ok&&L.kind==="register"){
      toast(r.j.output.split("\n")[0]+". It is under Models in the header."); $("#startbox").close(); return;
    }
    if(r.j.ok){
      var what=L.kind==="train"?"Training":L.kind==="serve"?"Chat server":L.kind==="resume"?"Resumed run":"Benchmark";
      toast(what+" submitted as job #"+r.j.job+(L.kind==="serve"?". The chat opens on this page when the model is loaded.":"."));
      $("#startbox").close(); refresh();
      setTimeout(function(){ var c=document.querySelector('[data-job-card="'+r.j.job+'"]'); if(c){ c.scrollIntoView({block:"center",behavior:"smooth"}); c.classList.add("flash"); } },1500);
    } else showResult("bad","Not submitted",r.j.output);
  }).catch(function(){ showResult("bad","Not started","The page could not reach the hub."); }).then(function(){ L.busy=false; if(L.opts) $("#startcheck").disabled=$("#startgo").disabled=!L.opts.enabled; });
}
function addOwnTest(){
  var o=L.own, f={kind:"evaltask",action:"start",name:o.name,data:o.data,question_field:o.q||"question",answer_field:o.a||"answer"};
  showResult("","Writing the test …","");
  api("/api/launch",{method:"POST",headers:{"Content-Type":"application/json","X-Requested-With":"bhaskera"},body:JSON.stringify(f)}).then(function(x){
    if(x.j.error){ showResult("bad","Not added",x.j.error); return; }
    if(!x.j.ok){ showResult("bad","Not added",x.j.output); return; }
    L.tasks[o.name]=1; L.own={name:"",data:"",q:"question",a:"answer"};
    api("/api/launch",{headers:{"X-Requested-With":"bhaskera"}}).then(function(r){ L.opts=r.j; renderStart(); toast("Added the test "+o.name+" and ticked it."); });
  }).catch(function(){ showResult("bad","Not added","The page could not reach the hub."); });
}
function sendEvalMany(action){
  var runs=L.evalSel.slice(), base=formData(action), jobs=[], notes=[];
  L.busy=true; $("#startcheck").disabled=$("#startgo").disabled=true; showResult("",action==="check"?"Checking …":"Submitting "+runs.length+" jobs …","");
  var step=function(i){
    if(i>=runs.length) return Promise.resolve();
    var f=Object.assign({},base,{run:runs[i]});
    return api("/api/launch",{method:"POST",headers:{"Content-Type":"application/json","X-Requested-With":"bhaskera"},body:JSON.stringify(f)}).then(function(r){
      var nm=runInfo(runs[i]).name||runs[i].slice(0,8);
      if(r.j.error){ notes.push(nm+": "+r.j.error); return; }
      if(r.j.ok&&!r.j.dry) jobs.push(r.j.job);
      notes.push(nm+": "+(r.j.ok?(r.j.dry?"checks out":"job #"+r.j.job):"not submitted"));
      if(!r.j.ok&&r.j.output) notes.push("  "+r.j.output.split("\n").slice(-2).join(" "));
    }).then(function(){ return new Promise(function(res){ setTimeout(res,2200); }); }).then(function(){ return step(i+1); });
  };
  step(0).then(function(){
    if(action==="check") showResult(notes.some(function(n){return /not submitted|^[^:]+: [A-Z]/.test(n)&&!/checks out/.test(n)})?"bad":"ok","Checked "+runs.length+" models. Nothing was submitted.",notes.join("\n"));
    else if(jobs.length===runs.length){ toast(jobs.length+" benchmark jobs submitted. Compare them on this page when they finish."); $("#startbox").close(); refresh(); }
    else showResult("bad",jobs.length+" of "+runs.length+" submitted",notes.join("\n"));
  }).catch(function(){ showResult("bad","Not started","The page could not reach the hub."); }).then(function(){ L.busy=false; if(L.opts) $("#startcheck").disabled=$("#startgo").disabled=!L.opts.enabled; });
}
function loadBase(){
  var base=val("f-rbase"); if(!base){ showResult("bad","Pick something to start from",""); return; }
  api("/api/launch",{method:"POST",headers:{"Content-Type":"application/json","X-Requested-With":"bhaskera"},body:JSON.stringify({kind:"recipe",action:"show",base:base})}).then(function(x){
    if(x.j.error){ showResult("bad","Could not load it",x.j.error); return; }
    L.rc.yaml=x.j.output; L.rc.base=base; L.rc.copy=true; $("#startout").hidden=true; renderStart();
  });
}
function prepareData(){
  var r=L.rc, f={action:"start",kind:"tokenize",name:r.name,model:r.model,train_path:r.raw.train,val_path:r.raw.val,format:r.raw.format,seq_len:r.raw.seq,model_kind:r.kind};
  if(r.kind==="multimodal"){ f.modalities=r.mods; f.media_root=r.raw.media; }
  showResult("","Submitting the tokenising job …","");
  api("/api/launch",{method:"POST",headers:{"Content-Type":"application/json","X-Requested-With":"bhaskera"},body:JSON.stringify(f)}).then(function(x){
    if(x.j.error){ showResult("bad","Not started",x.j.error); return; }
    if(x.j.ok){ showResult("ok","Tokenising job #"+x.j.job+" submitted. It takes a few minutes; when it has finished, reopen this form and the data is listed above.",x.j.output); refresh(); }
    else showResult("bad","Not submitted",x.j.output);
  }).catch(function(){ showResult("bad","Not started","The page could not reach the hub."); });
}
function openStart(kind,run){
  api("/api/launch",{headers:{"X-Requested-With":"bhaskera"}}).then(function(r){ L.opts=r.j; }).catch(function(){ L.opts={enabled:false,reason:"The page could not reach the hub.",recipes:[],runs:[],partitions:[]}; }).then(function(){
    L.kind=kind||L.kind; if(run) L.run=run; renderStart(); if(!$("#startbox").open) $("#startbox").showModal();
  });
}
$("#startbtn").addEventListener("click", function(){ openStart("train"); });
$("#startclose").addEventListener("click", function(){ $("#startbox").close(); });
$("#startkind").addEventListener("click", function(e){ var b=e.target.closest(".choice"); if(b){ L.kind=b.getAttribute("data-k"); L.hours=""; renderStart(); } });
$("#startfields").addEventListener("click", function(e){
  var b=e.target.closest("button"); if(!b) return;
  if(b.hasAttribute("data-n")){ L.nodes=parseInt(b.getAttribute("data-n"),10); renderStart(); }
  else if(b.hasAttribute("data-h")){ L.hours=b.getAttribute("data-h"); renderStart(); }
  else if(b.hasAttribute("data-t")){ var t=b.getAttribute("data-t"); L.tasks[t]=!L.tasks[t]; renderStart(); }
  else if(b.hasAttribute("data-q")){ L.quick=b.getAttribute("data-q")==="1"; renderStart(); }
  else if(b.hasAttribute("data-method")){ L.rc.method=b.getAttribute("data-method"); renderStart(); }
  else if(b.hasAttribute("data-kind")){ L.rc.kind=b.getAttribute("data-kind"); renderStart(); }
  else if(b.hasAttribute("data-mod")){ var mm=b.getAttribute("data-mod"), ix=L.rc.mods.indexOf(mm); if(ix>=0){ if(L.rc.mods.length>1) L.rc.mods.splice(ix,1); } else L.rc.mods.push(mm); renderStart(); }
  else if(b.id==="f-rload"){ loadBase(); }
  else if(b.hasAttribute("data-steps")){ L.rc.steps=b.getAttribute("data-steps"); renderStart(); }
  else if(b.hasAttribute("data-lr")){ L.rc.lr=b.getAttribute("data-lr"); renderStart(); }
  else if(b.hasAttribute("data-rank")){ L.rc.rank=b.getAttribute("data-rank"); renderStart(); }
  else if(b.hasAttribute("data-fmt")){ L.rc.raw.format=b.getAttribute("data-fmt"); renderStart(); }
  else if(b.id==="f-rawgo"){ prepareData(); }
  else if(b.hasAttribute("data-name")){ L.regname=b.getAttribute("data-name"); L.regnameFor=L.run; renderStart(); }
  else if(b.hasAttribute("data-step")){ L.step=parseInt(b.getAttribute("data-step"),10); showResume(); }
  else if(b.hasAttribute("data-run")){
    var rid=b.getAttribute("data-run");
    if(L.kind==="eval"){ var at=L.evalSel.indexOf(rid); if(at>=0){ if(L.evalSel.length>1) L.evalSel.splice(at,1); } else L.evalSel.push(rid); L.run=L.evalSel[L.evalSel.length-1]; }
    else L.run=rid;
    renderStart();
  }
  else if(b.id==="f-tgo"){ addOwnTest(); }
});
$("#startfields").addEventListener("input", function(e){
  if(e.target.id==="f-nodes"){ var n=parseInt(e.target.value,10); L.nodes=n||L.nodes; showSum(); }
  else if(e.target.id==="f-hours"){ L.hours=e.target.value.trim(); showSum(); }
  if(L.kind==="resume") showSum();
  else if(e.target.id==="f-recipe"){ L.recipe=e.target.value; fillInfo(); }
  else if(e.target.id==="f-alias"){ L.alias=e.target.value.trim(); showSum(); }
  else if(e.target.id==="f-name"){ L.regname=e.target.value.trim(); L.regnameFor=L.run; }
  else if(e.target.id==="f-rname"){ L.rc.name=e.target.value.trim(); }
  else if(e.target.id==="f-rmodel"){ L.rc.model=e.target.value.trim(); }
  else if(e.target.id==="f-rtrain"){ L.rc.train=e.target.value; }
  else if(e.target.id==="f-rval"){ L.rc.val=e.target.value; L.rcValTouched=true; }
  else if(e.target.id==="f-rsteps"){ if(e.target.value.trim()) L.rc.steps=e.target.value.trim(); }
  else if(e.target.id==="f-rbatch"){ L.rc.batch=e.target.value.trim(); }
  else if(e.target.id==="f-raccum"){ L.rc.accum=e.target.value.trim(); }
  else if(e.target.id==="f-revery"){ L.rc.every=e.target.value.trim(); }
  else if(e.target.id==="f-revals"){ L.rc.evals=e.target.value.trim(); }
  else if(e.target.id==="f-rawtrain"){ L.rc.raw.train=e.target.value.trim(); }
  else if(e.target.id==="f-rawval"){ L.rc.raw.val=e.target.value.trim(); }
  else if(e.target.id==="f-rawseq"){ L.rc.raw.seq=e.target.value.trim(); }
  else if(e.target.id==="f-tname"){ L.own.name=e.target.value.trim(); }
  else if(e.target.id==="f-tdata"){ L.own.data=e.target.value.trim(); }
  else if(e.target.id==="f-tq"){ L.own.q=e.target.value.trim(); }
  else if(e.target.id==="f-ta"){ L.own.a=e.target.value.trim(); }
  else if(e.target.id==="f-skipbase"){ L.skipBase=e.target.checked; }
  else if(e.target.id==="f-shots"){ L.shots=e.target.value.trim(); }
  else if(e.target.id==="f-rawmedia"){ L.rc.raw.media=e.target.value.trim(); }
  else if(e.target.id==="f-rhsdp"){ L.rc.hsdp=e.target.value.trim(); }
  else if(e.target.id==="f-rstages"){ L.rc.stages=e.target.value.trim(); }
  else if(e.target.id==="f-rmicro"){ L.rc.micro=e.target.value.trim(); }
  else if(e.target.id==="f-ryaml"){ L.rc.yaml=e.target.value; L.rc.copy=L.rc.copy&&!!L.rc.base; }
  else if(e.target.id==="f-rbase"){ L.rc.base=e.target.value; }
  else if(e.target.id==="f-rfind"){ filterSettings(e.target.value); }
  else if(e.target.hasAttribute("data-set")){ setSetting(e.target); }
});
function setSetting(el){
  var k=el.getAttribute("data-set"), v=el.value.trim(); if(v==="") delete L.rc.settings[k]; else L.rc.settings[k]=v;
  var n=Object.keys(L.rc.settings).length, c=$("#f-setcount"); if(c) c.textContent=n?n+" changed":"";
}
function filterSettings(q){
  q=q.trim().toLowerCase(); var groups=document.querySelectorAll("#f-setlist .setgroup");
  Array.prototype.forEach.call(groups,function(g){ var any=false;
    Array.prototype.forEach.call(g.querySelectorAll(".setrow"),function(r){ var on=!q||r.getAttribute("data-find").indexOf(q)>=0; r.hidden=!on; if(on) any=true; });
    g.hidden=!any; });
}
$("#startfields").addEventListener("toggle", function(e){
  var k=e.target.getAttribute&&e.target.getAttribute("data-od"); if(!k||L.kind!=="recipe") return;
  var was=!!L.rc.open[k]; L.rc.open[k]=e.target.open; if(k==="all"&&e.target.open&&!was) renderStart();
}, true);
$("#startcheck").addEventListener("click", function(){ send("check"); });
$("#startgo").addEventListener("click", function(){ send("start"); });

refresh(); setInterval(refresh, 8000); setInterval(clock, 1000);
})();
</script>
</body>
</html>
"""
