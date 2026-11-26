"""The Bhaskera dashboard: one self-contained HTML document (no outside fonts, scripts or images), because it is
served through an SSH tunnel and must work on any laptop. It reads /api/state every few seconds (see hub_state.py)."""

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bhaskera</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 36 36'%3E%3Crect width='36' height='36' rx='10' fill='%23a8381f'/%3E%3Cpath d='M9.5 19.5a8.5 8.5 0 0 1 17 0z' fill='white'/%3E%3Crect x='8' y='22' width='20' height='2.4' rx='1.2' fill='white'/%3E%3Crect x='11.5' y='26' width='13' height='2.4' rx='1.2' fill='white' opacity='.85'/%3E%3Crect x='15' y='30' width='6' height='2.4' rx='1.2' fill='white' opacity='.7'/%3E%3C/svg%3E">
<style>
:root{
  --bg:#f4f1e9; --surface:#fffdf8; --surface-2:#efe9db; --border:#e2dac7; --border-strong:#c9bfa6; --text:#1f1c17; --muted:#6b6454;
  --radius:10px; --radius-sm:6px; --shadow-1:0 1px 2px rgba(60,48,20,.06); --shadow-2:0 12px 32px rgba(40,30,10,.18);
  --ring:0 0 0 3px rgba(168,56,31,.22);
  --ink:#1f1c17; --paper:#f3efe6;
  --accent:#a8381f; --accent-ink:#8a2c17; --accent-soft:#e9dccf;
  --ok:#2b6648; --ok-soft:#e1efe6; --warn:#8a5d08; --warn-soft:#f5e9c6; --bad:#a8381f; --bad-soft:#f7e2dc;
  --info:#2c4a7a; --info-soft:#e2e9f4;
  --serif:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;     /* headings use the same family as the text */
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
:root[data-theme="dark"]{
  --bg:#14120d; --surface:#1c1913; --surface-2:#26221a; --border:#322e23; --border-strong:#4a4435; --text:#ece6d6; --muted:#a39b86; --shadow-1:0 1px 2px rgba(0,0,0,.4); --shadow-2:0 12px 32px rgba(0,0,0,.6); --ring:0 0 0 3px rgba(224,118,90,.3);
  --ink:#ece6d6; --paper:#16140f; --accent:#e0765a; --accent-ink:#ee9077; --accent-soft:#2f2018;
  --ok:#7fc29f; --ok-soft:#18281f; --warn:#e0b04c; --warn-soft:#33290f; --bad:#e0765a; --bad-soft:#32201a; --info:#8fb0e0; --info-soft:#1b2433;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#14120d; --surface:#1c1913; --surface-2:#26221a; --border:#322e23; --border-strong:#4a4435; --text:#ece6d6; --muted:#a39b86; --shadow-1:0 1px 2px rgba(0,0,0,.4); --shadow-2:0 12px 32px rgba(0,0,0,.6); --ring:0 0 0 3px rgba(224,118,90,.3);
  --ink:#ece6d6; --paper:#16140f; --accent:#e0765a; --accent-ink:#ee9077; --accent-soft:#2f2018;
  --ok:#7fc29f; --ok-soft:#18281f; --warn:#e0b04c; --warn-soft:#33290f; --bad:#e0765a; --bad-soft:#32201a; --info:#8fb0e0; --info-soft:#1b2433;
}}
*{box-sizing:border-box} html{-webkit-text-size-adjust:100%}
[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.55 var(--sans)}
a{color:inherit;text-decoration:none}
button{font:inherit;color:inherit;cursor:pointer}
.wrap{max-width:1120px;margin:0 auto;padding:0 24px}
.top{position:sticky;top:0;z-index:30;background:color-mix(in srgb,var(--surface) 90%,transparent);-webkit-backdrop-filter:saturate(1.3) blur(12px);backdrop-filter:saturate(1.3) blur(12px);border-bottom:1px solid var(--border)}
.top .wrap{display:flex;align-items:center;gap:8px;height:64px}
.brand{display:flex;align-items:center;gap:11px;margin-right:22px;border-radius:10px;flex:none}
.brand .logo{width:34px;height:34px;flex:none;display:block;filter:drop-shadow(0 1px 2px rgba(120,40,20,.28))}
.brand b{display:block;font:650 19px/1.05 var(--sans);letter-spacing:-.015em;color:var(--text)} .brand small{display:block;color:var(--muted);font-size:11.5px;font-weight:500;letter-spacing:.02em;margin-top:3px;white-space:nowrap}
.spacer{flex:1}
#updated{display:inline-block;width:16ch;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-variant-numeric:tabular-nums}
.pill{font-size:12.5px;color:var(--muted);display:inline-flex;align-items:center;gap:7px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--ok);display:inline-block;box-shadow:0 0 0 3px var(--ok-soft)}
.dot.stale{background:var(--bad)}
.icon-btn{width:32px;height:32px;display:grid;place-items:center;border:1px solid var(--border);background:transparent;border-radius:var(--radius-sm);transition:background .12s,border-color .12s}
.icon-btn:hover{border-color:var(--border-strong);background:var(--surface-2)}
.icon-btn svg{width:16px;height:16px}
main{padding:6px 0 60px}
h2{font:500 19px/1.2 var(--serif);margin:30px 0 12px;display:flex;align-items:baseline;gap:10px}
h2 .count{font:12px var(--mono);color:var(--muted)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow-1);padding:0 20px;margin-bottom:12px}
.job{padding:16px 0 18px}
.job-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 14px}
.job-name{font:500 19px/1.2 var(--serif)}
.muted{color:var(--muted)}
.mono{font-family:var(--mono)}
.badge{display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:600;padding:2px 9px 2px 8px;border-radius:999px;background:var(--surface-2)}
.badge::before{content:"";width:6px;height:6px;border-radius:50%;background:currentColor}
.b-ok{color:var(--ok);background:var(--ok-soft)} .b-warn{color:var(--warn);background:var(--warn-soft)} .b-bad{color:var(--bad);background:var(--bad-soft)} .b-info{color:var(--info);background:var(--info-soft)} .b-mute{color:var(--muted)}
.chip{font:12px var(--mono);color:var(--muted);padding:0}
.chip+.chip::before{content:"/";margin:0 8px;color:var(--border)}
.meta{display:flex;flex-wrap:wrap;gap:0;margin-left:auto}
.bar{height:6px;border-radius:99px;background:var(--surface-2);overflow:hidden}
.bar>i{display:block;height:100%;background:var(--ink);transition:width .6s ease}
.warnbar{background:var(--warn-soft);color:var(--warn);border-radius:12px;padding:10px 14px;margin:14px 0 0;font-size:13.5px;line-height:1.5}
.runfoot{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:10px 14px;border-top:1px solid var(--border);font-size:13px}.runfoot .spacer{flex:1}
.dstat{display:inline-flex;align-items:center;font-size:12px;font-weight:600;padding:2px 9px;border-radius:999px}.dstat.ok{color:var(--ok);background:var(--ok-soft)}.dstat.changed{color:var(--warn);background:var(--warn-soft)}.dstat.missing{color:var(--bad);background:var(--bad-soft)}
.dlcard{border:1px solid var(--border);border-radius:12px;background:var(--surface);padding:12px 14px;margin-bottom:14px}.dlcard b{font-weight:650}
.dlrow{display:flex;flex-direction:column;gap:5px;margin-bottom:12px}.dlrow>label{font-size:12px;font-weight:650;letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}.dlrow input[type=text],.dlrow input[type=password],.dlrow textarea{width:100%}
.dlprog{margin:10px 0 2px}.dlprog .row{display:flex;justify-content:space-between;gap:12px;font-size:13px;margin-bottom:6px}
.bar.thin{height:4px} .bar.thin>i{background:var(--muted)} .bar.hot>i{background:var(--warn)}
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
.btn{display:inline-flex;align-items:center;justify-content:center;gap:7px;min-height:32px;border:1px solid var(--border-strong);background:var(--surface);color:var(--text);border-radius:var(--radius-sm);padding:5px 13px;font-weight:550;font-size:13px;transition:background .12s,border-color .12s,color .12s}
.btn:hover{background:var(--surface-2);border-color:var(--text)}
.btn.primary{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.btn.primary:hover{background:var(--accent);border-color:var(--accent);color:#fff}
.btn[aria-disabled=true]{opacity:.4;pointer-events:none}
.btn:disabled{opacity:.4;pointer-events:none}
.btn svg{width:14px;height:14px}
.btn.small{min-height:26px;padding:2px 10px;font-size:12px}
.empty{padding:28px 0}
.empty h3{margin:4px 0;font:500 17px var(--serif)} .empty p{margin:4px 0;color:var(--muted);max-width:560px}
code,.code{font:12.5px var(--mono);background:var(--surface-2);border-radius:4px;padding:1px 6px}
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
th{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);font-weight:600;text-align:left;padding:11px 12px 9px 0;white-space:nowrap;border-bottom:1px solid var(--border)}
td{padding:12px 12px 12px 0;border-bottom:1px solid var(--border);white-space:nowrap}
tr:last-child td{border-bottom:0}
tbody tr:hover td{background:var(--surface-2)}
td.num,th.num{text-align:right;font-family:var(--mono);font-size:12.5px}
td a.run{font-weight:600;display:inline-block;max-width:240px;overflow:hidden;text-overflow:ellipsis;vertical-align:bottom;white-space:nowrap}
td a.run:hover{text-decoration:underline;text-underline-offset:3px}
.note-card{padding:12px 0;color:var(--muted)}
.more{margin-top:24px;border:1px solid var(--border);border-radius:var(--radius);background:var(--surface);padding:0 16px}
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
.runact .btn.primary{background:transparent;color:var(--accent);border-color:var(--accent)}
.runact .btn.primary:hover{background:var(--accent);color:#fff}
.quiet{padding:26px 22px;color:var(--muted);background:var(--surface);border:1px dashed var(--border-strong);border-radius:var(--radius);display:flex;flex-wrap:wrap;gap:12px 18px;align-items:center;justify-content:space-between}
.quiet .actions{gap:8px}
footer{color:var(--muted);font-size:12.5px;margin-top:40px;padding-top:14px;border-top:1px solid var(--border)}
footer a{text-decoration:underline;text-underline-offset:3px}
dialog{width:min(980px,96vw);height:min(860px,92vh);display:none;flex-direction:column;overflow:hidden;padding:0;border:1px solid var(--border-strong);border-radius:14px;background:var(--surface);color:var(--text);box-shadow:var(--shadow-2)}
dialog[open]{display:flex}
dialog::backdrop{background:rgba(31,28,23,.5);backdrop-filter:blur(2px)}
.dlg-head{display:flex;align-items:center;gap:10px;padding:12px 22px;border-bottom:1px solid var(--border)}
.dlg-head b{font:500 18px var(--serif)}
#logtext{margin:0;padding:14px 18px;background:#14120d;color:#d9d2c0;font:12px/1.55 var(--mono);height:62vh;overflow:auto;white-space:pre-wrap;word-break:break-word}
.skeleton{height:90px;background:var(--surface-2)}
/* chat */
.chat-head{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:center;justify-content:space-between;padding:14px 0;border-bottom:1px solid var(--border)}
.chat-controls{display:flex;flex-wrap:wrap;gap:14px;align-items:center}
.chat-controls select,.chat-controls input[type=range]{accent-color:var(--accent)}
.chat-controls select{border:1px solid var(--border);background:transparent;color:var(--text);border-radius:var(--radius-sm);padding:4px 6px;font:inherit}
.chat-controls label{display:flex;align-items:center;gap:6px;font-size:12.5px}
.msgs{padding:16px 0;min-height:160px;max-height:460px;overflow:auto;display:flex;flex-direction:column;gap:16px}
.msg{max-width:78ch;white-space:pre-wrap;word-break:break-word;line-height:1.6}
.msg.user{align-self:flex-end;background:var(--ink);color:var(--paper);padding:8px 13px;border-radius:var(--radius-sm)}
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
.composer textarea{flex:1;resize:vertical;min-height:44px;max-height:180px;border:1px solid var(--border);background:var(--surface);color:var(--text);border-radius:var(--radius-sm);padding:9px 11px;font:inherit}
.composer textarea:focus{outline:0;border-color:var(--ink)}
.cmds td,.cmds th{padding:9px 12px 9px 0}
.fgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:16px;padding:18px 22px 6px}
.fld label{display:block;font-size:11px;color:var(--muted);margin-bottom:4px;font-weight:600;text-transform:uppercase;letter-spacing:.07em}
.fld select,.fld input[type=text],.fld input[type=number]{width:100%;border:1px solid var(--border);background:var(--surface);color:var(--text);border-radius:var(--radius-sm);padding:8px 10px;font:inherit}
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
.opt{position:relative}.opt>div{position:absolute;right:0;z-index:2;display:flex;flex-direction:column;gap:10px;padding:12px 14px;margin-top:6px;background:var(--bg);border:1px solid var(--ink);border-radius:var(--radius-sm);min-width:230px}
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
.srow select,.srow input[type=number],.srow input[type=text]{border:1px solid var(--border);background:var(--surface);color:var(--text);border-radius:var(--radius-sm);padding:7px 10px;font:inherit}
.srow select{width:100%}.srow input[type=number]{width:90px}.srow input[type=text]{width:100%}
.srow select:focus,.srow input:focus{outline:0;border-color:var(--ink)}
.srow .sub{font-size:12.5px;color:var(--muted);margin-top:5px}
.chips{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.chipb{border:1px solid var(--border);background:transparent;border-radius:var(--radius-sm);padding:4px 11px;font-size:13px}
.chipb:hover{border-color:var(--ink)}
.chipb[aria-pressed=true]{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.rlist{border:1px solid var(--border);border-radius:var(--radius-sm);max-height:210px;overflow:auto}
.ritem{display:flex;gap:10px;align-items:baseline;width:100%;text-align:left;border:0;border-bottom:1px solid var(--border);background:transparent;padding:8px 11px}
.ritem:last-child{border-bottom:0}
.ritem:hover{background:var(--surface)}
.ritem[aria-pressed=true]{background:var(--surface);box-shadow:inset 3px 0 0 var(--accent)}
.ritem b{font-weight:600}.ritem .m{margin-left:auto;font:12px var(--mono);color:var(--muted);white-space:nowrap}
.srow details summary{cursor:pointer;color:var(--muted);font-size:13px}
#f-setlist{max-height:340px;overflow:auto;border:1px solid var(--border);border-radius:var(--radius-sm);padding:4px 10px}
.setgroup{margin:10px 0}.settitle{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px}
.setrow{display:grid;grid-template-columns:minmax(150px,1fr) minmax(120px,1fr);gap:4px 12px;align-items:center;padding:5px 0;border-top:1px solid var(--border)}
.setrow .sub{grid-column:1/-1;margin:0}.setrow input,.setrow select{width:100%}
.srow details label{display:flex;gap:8px;align-items:center;margin-top:8px;font-size:13px}
.ssum{margin:14px 22px 0;padding:10px 13px;border-left:3px solid var(--ink);background:var(--surface);font-size:13px;min-height:20px}
.ssum.warn{border-left-color:var(--warn);background:var(--warn-soft)}
.ssum:empty{display:none}
.toast{position:fixed;left:50%;top:78px;transform:translateX(-50%);z-index:20;background:var(--ink);color:var(--paper);padding:10px 18px;border-radius:var(--radius-sm);font-size:13.5px;box-shadow:4px 4px 0 rgba(31,28,23,.2)}
.job.flash{animation:fl 2.4s ease-out}@keyframes fl{0%{background:var(--warn-soft)}100%{background:transparent}}
.btn.confirm{background:var(--accent);border-color:var(--accent);color:#fff}
.attached{display:flex;flex-wrap:wrap;gap:8px;padding:10px 0 0;align-items:center}
.att{display:inline-flex;align-items:center;gap:8px;border:1px solid var(--border);border-radius:var(--radius-sm);padding:3px 4px 3px 8px;font-size:12.5px;background:var(--surface)}
.att img{height:34px;width:34px;object-fit:cover;border-radius:4px}
.att button{border:0;background:transparent;font-size:15px;line-height:1;padding:2px 6px;color:var(--muted)}
.att button:hover{color:var(--bad)}
.msg .files{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:6px}
.msg .files img{max-height:130px;max-width:200px;border-radius:4px;border:1px solid rgba(255,255,255,.25)}
.msg .files span{font:12px var(--mono);opacity:.85;border:1px solid rgba(255,255,255,.35);padding:1px 7px;border-radius:4px}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
.linkbtn{border:0;background:transparent;color:var(--muted);font-size:12.5px;text-decoration:underline;text-underline-offset:3px;padding:2px 4px}.linkbtn:hover{color:var(--text)}
.intro{padding:11px 14px;background:var(--info-soft);border-radius:var(--radius-sm);font-size:13px;line-height:1.55}
dialog.small{width:min(640px,94vw);height:auto;max-height:86vh}
.gloss{margin:0}.gloss dt{font-weight:600;margin-top:14px}.gloss dd{margin:2px 0 0;color:var(--muted);line-height:1.55}
/* header: hover menus */
.gnav{display:flex;align-items:center;gap:2px;flex:none}
.gi{position:relative}
.gb{display:inline-flex;align-items:center;gap:6px;border:0;background:transparent;border-radius:9px;padding:9px 13px;font-size:14px;font-weight:550;color:var(--muted);transition:background .14s,color .14s}
.gb svg{width:12px;height:12px;fill:none;stroke:currentColor;stroke-width:2;stroke-linecap:round;stroke-linejoin:round;transition:transform .18s;opacity:.7}
.gi:hover>.gb,.gi.open>.gb,.gb:focus-visible{color:var(--text);background:var(--surface-2)}
.gi.open>.gb svg{transform:rotate(180deg)}
.gp{position:absolute;left:-8px;top:100%;margin-top:10px;z-index:40;min-width:300px;padding:8px;display:flex;gap:6px;background:var(--surface);border:1px solid var(--border);border-radius:16px;box-shadow:var(--shadow-2);
  opacity:0;visibility:hidden;transform:translateY(-4px);transition:opacity .15s,transform .15s,visibility 0s .15s}
.gp::before{content:"";position:absolute;left:0;right:0;top:-12px;height:12px}
.gi.open>.gp{opacity:1;visibility:visible;transform:none;transition:opacity .15s,transform .15s,visibility 0s}
.gi.right>.gp{left:auto;right:-8px}
.mcol{display:flex;flex-direction:column;min-width:270px;flex:1}
.mh{padding:8px 10px 4px;font-size:11px;font-weight:650;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.mi{display:flex;align-items:center;gap:12px;width:100%;text-align:left;border:0;background:transparent;border-radius:11px;padding:9px 10px;color:var(--text);transition:background .12s}
.mi:hover,.mi:focus-visible{background:var(--surface-2);outline:0}
.mi.off{opacity:.45;pointer-events:none}
.mic{flex:none;width:38px;height:38px;border-radius:10px;display:grid;place-items:center;background:var(--accent-soft);color:var(--accent)}
.mic svg{width:20px;height:20px;fill:none;stroke:currentColor;stroke-width:1.7;stroke-linecap:round;stroke-linejoin:round}
.mt{display:flex;flex-direction:column;min-width:0}.mt b{font-size:14px;font-weight:600;line-height:1.25}.mt span{font-size:12.5px;color:var(--muted);line-height:1.35;overflow:hidden;text-overflow:ellipsis}
.mnone{padding:8px 12px 10px;font-size:12.5px;color:var(--muted)}
.gcta{display:inline-flex;align-items:center;gap:7px;height:38px;padding:0 16px;border-radius:10px;font-weight:600;font-size:13.5px;background:var(--ink);color:var(--paper);border:0;margin:0 6px 0 10px;flex:none;white-space:nowrap;transition:background .14s,transform .14s}
.gcta:hover{background:var(--accent);color:#fff}.gcta svg{width:14px;height:14px;fill:none;stroke:currentColor;stroke-width:2.2;stroke-linecap:round}
.gb.ub{padding:5px 10px 5px 6px;gap:8px;color:var(--text);font-weight:600}.gp.acct{min-width:260px}.mu{padding:10px 12px 8px;border-bottom:1px solid var(--border);margin:-2px -2px 6px}.mu b{display:block;font-size:14px}.mu span{font-size:12.5px;color:var(--muted)}
.byepage{max-width:520px;margin:20vh auto 0;padding:0 24px;text-align:center}.byepage h1{margin:18px 0 8px;font:650 26px/1.2 var(--sans);letter-spacing:-.02em}.byepage p{color:var(--muted);line-height:1.6}.byepage code{display:inline-block;margin-top:6px}
.av{width:26px;height:26px;border-radius:50%;display:inline-grid;place-items:center;background:var(--accent-soft);color:var(--accent-ink);font-size:12px;font-weight:700}
.pill.status{flex:none;padding:5px 11px;border:1px solid var(--border);border-radius:999px;background:var(--surface)}
section,details.more,#chat-live,#cmpsec,#modsec{scroll-margin-top:84px}
@media (max-width:900px){
  .top .wrap{flex-wrap:wrap;height:auto;padding-top:8px;padding-bottom:8px;row-gap:4px}
  .gnav{order:5;width:100%;overflow-x:auto}.gp{position:fixed;left:12px;right:12px;top:118px;flex-direction:column;max-height:70vh;overflow:auto}.gi.right>.gp{left:12px}
  #updated{display:none}
}
/* pages */
.phead{padding:30px 0 4px}.phead h1{margin:0;font:650 28px/1.15 var(--sans);letter-spacing:-.02em}.phead p{margin:7px 0 0;color:var(--muted);font-size:14.5px;max-width:72ch}
.agrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(236px,1fr));gap:14px;margin:20px 0 10px}
.mi.acard{align-items:flex-start;gap:14px;padding:16px;border:1px solid var(--border);border-radius:14px;background:var(--surface);box-shadow:var(--shadow-1);transition:border-color .14s,box-shadow .14s,transform .14s}
.mi.acard:hover,.mi.acard:focus-visible{border-color:var(--border-strong);box-shadow:var(--shadow-2);transform:translateY(-1px);background:var(--surface)}
.mi.acard .mt span{white-space:normal}.mi.acard .mt b{font-size:15px}
.sumrow{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:20px 0 4px}
.sumtile{display:flex;flex-direction:column;gap:2px;text-align:left;border:1px solid var(--border);border-radius:14px;background:var(--surface);padding:14px 16px;box-shadow:var(--shadow-1);transition:border-color .14s,transform .14s}
.sumtile:hover{border-color:var(--border-strong);transform:translateY(-1px)}
.sumtile .k{font-size:11.5px;font-weight:650;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}.sumtile .v{font:650 26px/1.2 var(--sans);letter-spacing:-.02em}.sumtile .v small{font-size:12.5px;font-weight:500;color:var(--muted);letter-spacing:0}
.gb{position:relative}
.gi.here>.gb{color:var(--text)}.gi.here>.gb::after{content:"";position:absolute;left:13px;right:13px;bottom:3px;height:2px;border-radius:2px;background:var(--accent)}
.only-touch{display:none}@media (hover:none) and (pointer:coarse){.only-touch{display:flex}}
#cmpempty{margin-top:14px}
/* polish: focus, motion, small screens */
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
input:focus-visible,select:focus-visible,textarea:focus-visible{outline:0;box-shadow:var(--ring)}
.srow select:focus,.srow input:focus,.composer textarea:focus,.tools input:focus{border-color:var(--accent);box-shadow:var(--ring)}
.chipb{border-color:var(--border-strong);transition:background .12s,border-color .12s}
.chipb:hover{border-color:var(--text)}
.ritem{border-radius:var(--radius-sm)}
.toast{border-radius:var(--radius-sm);box-shadow:var(--shadow-2)}
.skeleton{border-radius:var(--radius);animation:pulse 1.4s ease-in-out infinite}
@keyframes pulse{50%{opacity:.55}}
.h2row{align-items:center}
@media (max-width:640px){.wrap{padding:0 14px}.top .wrap{height:auto;min-height:56px;flex-wrap:wrap;padding-top:8px;padding-bottom:8px}.card{padding:0 14px}#updated{width:auto}}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style>
</head>
<body>
<header class="top"><div class="wrap">
  <a class="brand" href="#" id="brand" aria-label="Bhaskera, back to the top">
    <svg class="logo" viewBox="0 0 36 36" aria-hidden="true"><rect width="36" height="36" rx="10" fill="var(--accent)"/><path d="M9.5 19.5a8.5 8.5 0 0 1 17 0z" fill="#fff"/><rect x="8" y="22" width="20" height="2.4" rx="1.2" fill="#fff"/><rect x="11.5" y="26" width="13" height="2.4" rx="1.2" fill="#fff" opacity=".85"/><rect x="15" y="30" width="6" height="2.4" rx="1.2" fill="#fff" opacity=".7"/></svg>
    <span><b>Bhaskera</b><small id="tag"></small></span>
  </a>
  <nav class="gnav" id="gnav" aria-label="Main">
    <div class="gi" data-menu="train" data-page="train"><button type="button" class="gb" aria-haspopup="true" aria-expanded="false">Train<svg viewBox="0 0 12 12"><path d="M2.5 4.5l3.5 3.5 3.5-3.5"/></svg></button><div class="gp" id="mn-train" role="menu"></div></div>
    <div class="gi" data-menu="serve" data-page="serve"><button type="button" class="gb" aria-haspopup="true" aria-expanded="false">Serve<svg viewBox="0 0 12 12"><path d="M2.5 4.5l3.5 3.5 3.5-3.5"/></svg></button><div class="gp" id="mn-serve" role="menu"></div></div>
    <div class="gi" data-menu="eval" data-page="evaluate"><button type="button" class="gb" aria-haspopup="true" aria-expanded="false">Evaluate<svg viewBox="0 0 12 12"><path d="M2.5 4.5l3.5 3.5 3.5-3.5"/></svg></button><div class="gp" id="mn-eval" role="menu"></div></div>
    <div class="gi" data-menu="monitor" data-page="monitor"><button type="button" class="gb" aria-haspopup="true" aria-expanded="false">Monitor<svg viewBox="0 0 12 12"><path d="M2.5 4.5l3.5 3.5 3.5-3.5"/></svg></button><div class="gp" id="mn-monitor" role="menu"></div></div>
    <div class="gi right" data-menu="downloads"><button type="button" class="gb" aria-haspopup="true" aria-expanded="false">Downloads<svg viewBox="0 0 12 12"><path d="M2.5 4.5l3.5 3.5 3.5-3.5"/></svg></button><div class="gp" id="mn-dl" role="menu"></div></div>
    <div class="gi right" data-menu="docs"><button type="button" class="gb" aria-haspopup="true" aria-expanded="false">Docs<svg viewBox="0 0 12 12"><path d="M2.5 4.5l3.5 3.5 3.5-3.5"/></svg></button><div class="gp" id="mn-docs" role="menu"></div></div>
  </nav>
  <div class="spacer"></div>
  <button class="gcta" id="startbtn" type="button"><svg viewBox="0 0 14 14"><path d="M7 2v10M2 7h10"/></svg>New run</button>
  <nav class="gnav" id="unav" aria-label="Account"><div class="gi right" data-menu="user"><button type="button" class="gb ub" id="who" aria-haspopup="true" aria-expanded="false" hidden></button><div class="gp" id="mn-user" role="menu"></div></div></nav>
  <span class="pill status" title="The page refreshes by itself"><span class="dot" id="dot"></span><span id="updated">connecting…</span></span>
  <button class="icon-btn" id="theme" aria-label="Switch between light and dark" title="Light / dark"></button>
</div></header>

<main class="wrap">
  <div class="view" data-view="home">
    <div class="phead"><h1 id="home-title">Welcome</h1><p id="home-sub"></p></div>
    <div class="warnbar" id="diskwarn" hidden></div>
    <div class="sumrow" id="sumrow"></div>
    <div class="agrid" id="ag-home"></div>
  <div class="h2row"><h2>Jobs <span class="count" id="jobcount">0</span></h2><span class="muted" id="stats"></span></div>
  <div id="jobs"><div class="skeleton"></div></div>

  </div>
  <div class="view" data-view="train" hidden>
    <div class="phead"><h1>Train a model</h1><p>Create a recipe, run it on the cluster, and continue a run that stopped.</p></div>
    <div class="agrid" id="ag-train"></div>
  <section id="runsec">
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
      <div class="runfoot" id="runfoot" hidden></div>
    </div>
  </section>
  <section id="datasec">
    <h2>Data <span class="count" id="datacount">0</span></h2>
    <div class="card tablebox">
      <div class="scroll"><table>
        <thead><tr><th>Dataset</th><th>Version</th><th>Content</th><th>Source</th><th>Status</th><th></th></tr></thead>
        <tbody id="datarows"></tbody>
      </table></div>
      <div class="tools" style="display:flex;align-items:center;gap:12px;padding:10px 14px"><button class="btn small" id="data-add" type="button">Register data</button><span id="data-prog" class="muted"></span></div>
    </div>
  </section>
  </div>
  <div class="view" data-view="serve" hidden>
    <div class="phead"><h1>Serve and chat</h1><p>Talk to a trained model, with pictures or sound if it reads them, and keep the ones you like under a name.</p></div>
    <div class="agrid" id="ag-serve"></div>
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

  <section id="modsec">
    <h2>Models <span class="count" id="modcount">0</span></h2>
    <div class="card tablebox"><div class="scroll"><table>
      <thead><tr><th>Model</th><th>Version</th><th>Trained from</th><th>Alias</th><th class="num">Eval loss</th><th></th></tr></thead>
      <tbody id="models"></tbody>
    </table></div></div>
  </section>
  </div>
  <div class="view" data-view="evaluate" hidden>
    <div class="phead"><h1>Evaluate</h1><p>Test models on standard benchmarks or on your own questions, and put the results side by side.</p></div>
    <div class="agrid" id="ag-evaluate"></div>
  <section id="cmpsec" hidden>
    <h2>Compare models</h2>
    <div class="card" id="cmp" style="padding:14px 16px"></div>
  </section>
    <div id="cmpempty" class="card note-card">Nothing to compare yet. Benchmark two models and they appear here, side by side.</div>
  </div>
  <div class="view" data-view="monitor" hidden>
    <div class="phead"><h1>Monitor</h1><p>Open MLflow and the Ray Dashboard, look at jobs and runs, and check the cluster.</p></div>
    <div class="agrid" id="ag-monitor"></div>
  <details class="more" id="morebox" open><summary>Cluster limits and node health <span id="morewarn"></span></summary><div id="cluster"></div></details>
  </div>
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
<dialog id="dlbox">
  <div class="dlg-head"><b>Download from Hugging Face</b><span class="spacer" style="flex:1"></span><button class="btn" id="dlclose" type="button">Close</button></div>
  <div class="dlgbody" id="dlbody" style="padding:16px 22px 18px"></div>
</dialog>
<dialog id="databox" class="small">
  <div class="dlg-head"><b>Register data</b><span class="spacer" style="flex:1"></span><button class="btn" id="dataclose" type="button">Close</button></div>
  <div class="dlgbody" id="databody" style="padding:16px 22px 18px"></div>
  <div class="dlg-foot"><button class="btn primary" id="datago" type="button">Register</button></div>
</dialog>
<dialog id="bye" class="small">
  <div class="dlg-head"><b>Log out</b></div>
  <div class="dlgbody" style="padding:14px 22px 6px"><p id="bye-text" style="margin:0 0 10px;line-height:1.6"></p></div>
  <div class="dlg-foot"><button class="btn" id="bye-cancel" type="button">Stay</button><button class="btn primary" id="bye-ok" type="button">Log out</button></div>
</dialog>
<dialog id="guide" class="small">
  <div class="dlg-head"><b>Words used on this page</b><span class="spacer" style="flex:1"></span><button class="btn" id="guideclose" type="button">Close</button></div>
  <div class="dlgbody" style="padding:6px 22px 18px"><dl class="gloss">
    <dt>Run</dt><dd>One training session. Each one is a row under Runs.</dd>
    <dt>Recipe</dt><dd>The saved list of choices for one training run: which model, which examples, how long. Make one under Create recipe.</dd>
    <dt>Fine-tuning</dt><dd>Teaching a model that already exists more about your subject or your style, instead of building one from nothing.</dd>
    <dt>Tokenised data</dt><dd>Your examples turned into the numbers a model reads. It is done once, with Prepare data, and reused by every run.</dd>
    <dt>LoRA and adapter</dt><dd>LoRA trains a small add-on instead of the whole model. That add-on is the adapter: it is what a run produces, and what you chat with, test and register.</dd>
    <dt>QLoRA</dt><dd>LoRA with the original model squeezed to 4 bits, so it needs much less memory.</dd>
    <dt>Method (DDP, FSDP2, HSDP, Pipeline)</dt><dd>How the model is spread over the GPUs. QLoRA is the usual start. The Create recipe page says when each of the others fits.</dd>
    <dt>Step</dt><dd>One update of the model from one batch of examples. A run is a number of steps.</dd>
    <dt>Validation data and eval loss</dt><dd>Examples kept aside to check progress. Eval loss is the score on them: lower is better.</dd>
    <dt>Checkpoint</dt><dd>A snapshot saved during training. If a run stops, Resume continues from the last one.</dd>
    <dt>Node and GPU</dt><dd>A node is one machine; a GPU is one of the chips in it that does the work.</dd>
    <dt>Benchmark</dt><dd>A set of questions with known answers used to score a model, so you can compare it with its base model or with another one.</dd>
    <dt>Registry and alias</dt><dd>The registry is a named, numbered list of your trained models. An alias is a label you choose for one version, such as the one you use.</dd>
  </dl></div>
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
var PROXY=false;
var PORT = location.port || "8265";
var MLFLOW = PROXY ? "mlflow/" : "http://mlflow.localhost:"+PORT+"/";
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
  if(PROXY&&/\.localhost:/.test(href||"")) return "";
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
    var er=(state.runs||[]).filter(function(r){return String(r.job)===String(j.id)&&r.energy_kwh!=null})[0];
    if(er) tl.push(tile("GPU energy so far",kwh(er.energy_kwh)));
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
  return '<div class="quiet"><div><b style="color:var(--text)">Nothing is running</b><div>Start a training run, or chat with or benchmark a model you trained.</div></div></div>';
}

/* ---- runs ---- */
function kwh(v){ if(v==null) return "–"; return v<0.1 ? fx(v*1000,0)+'<small> Wh</small>' : fx(v,v<10?2:1)+'<small> kWh</small>'; }
function renderRuns(){
  var rows=(state.runs||[]).filter(function(r){
    if(filter && r.status!==filter) return false;
    if(query && (r.name+" "+r.job+" "+(r.group||"")).toLowerCase().indexOf(query.toLowerCase())<0) return false;
    return true;
  });
  $("#runcount").textContent=rows.length;
  $("#runtools").hidden=(state.runs||[]).length<=6;
  var all=state.runs||[], hasEval=all.some(function(r){return r.eval_loss!=null}), hasBench=all.some(function(r){return r.lm_eval&&Object.keys(r.lm_eval).length}), hasEnergy=all.some(function(r){return r.energy_kwh!=null});
  $("#runhead").innerHTML='<tr><th>Run</th><th>Status</th><th>Started</th>'+(hasEval?'<th class="num">Eval loss</th>':'')+(hasEnergy?'<th class="num" title="Energy the GPUs of the run used, worked out from the power they drew. CPUs, fans and the network are not counted.">GPU energy</th>':'')+(hasBench?'<th>Benchmarks</th>':'')+'<th></th></tr>';
  var shown=rows.slice(0,runShow);
  $("#runs").innerHTML = rows.length ? shown.map(function(r){
    var act=(r.has_adapter&&r.status!=="RUNNING") ? '<div class="runact"><button class="btn small primary" data-start="serve" data-run="'+esc(r.run_id)+'">Chat</button><button class="btn small" data-start="eval" data-run="'+esc(r.run_id)+'">Benchmark</button></div>' :
          ((r.status==="FAILED"||r.status==="KILLED"||r.stale)&&r.job ? '<div class="runact"><button class="btn small" data-start="resume" data-run="'+esc(r.run_id)+'">Resume</button></div>' : "");
    var rm=(r.status!=="RUNNING"||r.stale)?'<button class="btn small" data-rm="'+esc(r.run_id)+'" title="Take this run off the list (MLflow keeps it hidden, so it can be restored)">Remove</button>':"";
    act=act?act.replace(/<\/div>$/,rm+"</div>"):(rm?'<div class="runact">'+rm+'</div>':"");
    return '<tr><td>'+(PROXY&&!(state.portal&&state.portal.mlflow)?'<span class="run">'+esc(r.name)+'</span>':'<a class="run" href="'+esc(runUrl(r))+'" target="_blank" rel="noopener">'+esc(r.name)+'</a>')+(r.group?' <span class="chip">'+esc(r.group)+'</span>':"")+(r.dataset?' <span class="chip" title="The data this run trained on'+(r.dataset.status==="changed"?" (the files differed from the registered version when it started)":"")+'">'+esc(r.dataset.name)+' v'+esc(r.dataset.version)+(r.dataset.status==="changed"?" ⚠":"")+'</span>':"")+'</td>'+
      '<td>'+runBadge(r)+'</td><td class="muted">'+ago(r.started)+'</td>'+(hasEval?'<td class="num">'+fx(r.eval_loss,3)+'</td>':'')+(hasEnergy?'<td class="num">'+kwh(r.energy_kwh)+'</td>':'')+(hasBench?'<td>'+bench(r)+'</td>':'')+'<td>'+act+'</td></tr>';
  }).join("") : '<tr><td colspan="6" class="muted" style="text-align:center;padding:26px">'+(all.length?"No runs match.":"No runs yet.")+'</td></tr>';
  runFoot(rows,all);
}
var runShow=10;
function runFoot(rows,all){
  var f=$("#runfoot"), bad=all.filter(function(r){return r.status==="FAILED"||r.status==="KILLED"||r.stale}), h="";
  if(rows.length>10){
    h+='<span class="muted">Showing '+Math.min(runShow,rows.length)+' of '+rows.length+'</span>';
    if(runShow<rows.length) h+='<button class="btn small" data-rows="more">Show 10 more</button><button class="btn small" data-rows="all">Show all '+rows.length+'</button>';
    if(runShow>10) h+='<button class="btn small" data-rows="less">Show fewer</button>';
  }
  h+='<span class="spacer"></span>';
  if(bad.length) h+='<button class="btn small" data-clear="bad" data-ids="'+bad.map(function(r){return r.run_id}).join(",")+'" title="Take the failed, killed and stalled runs off the list (MLflow keeps them hidden)">Clear failed and stalled ('+bad.length+')</button>';
  f.innerHTML=h; f.hidden=!h.replace(/<span class="spacer"><\/span>/,"");
}
function removeRuns(ids,btn){
  if(btn) btn.disabled=true;
  lpost({kind:"runs",action:"remove",runs:ids}).then(function(r){
    if(!r.j||r.j.error){ toast((r.j&&r.j.error)||"Could not remove those runs."); if(btn) btn.disabled=false; return; }
    var gone={}; ids.forEach(function(i){ gone[i]=1; }); (r.j.skipped||[]).forEach(function(x){ delete gone[x.run_id]; });
    state.runs=(state.runs||[]).filter(function(x){ return !gone[x.run_id]; }); renderRuns();
    toast("Removed "+r.j.removed+(r.j.removed===1?" run":" runs")+" from the list."+((r.j.skipped||[]).length?" "+r.j.skipped.length+" could not be removed ("+r.j.skipped[0].reason+").":"")+" They are only hidden in MLflow.");
    refresh();
  }).catch(function(){ toast("The page could not reach the hub."); if(btn) btn.disabled=false; });
}
$("#runs").addEventListener("click",function(e){
  var b=e.target.closest("button[data-rm]"); if(!b) return;
  if(b.getAttribute("data-sure")!=="1"){ b.setAttribute("data-sure","1"); b.textContent="Sure? Remove"; setTimeout(function(){ if(b.isConnected){ b.removeAttribute("data-sure"); b.textContent="Remove"; } },4000); return; }
  removeRuns([b.getAttribute("data-rm")],b);
});
$("#runfoot").addEventListener("click",function(e){
  var b=e.target.closest("button"); if(!b) return;
  var m=b.getAttribute("data-rows");
  if(m){ runShow=m==="more"?runShow+10:m==="all"?100000:10; renderRuns(); return; }
  if(b.hasAttribute("data-clear")){
    if(b.getAttribute("data-sure")!=="1"){ b.setAttribute("data-sure","1"); var t=b.textContent; b.textContent="Sure? "+t; setTimeout(function(){ if(b.isConnected) renderRuns(); },4000); return; }
    removeRuns(b.getAttribute("data-ids").split(","),b);
  }
});


/* ---- data versions, downloads and Hugging Face access ---- */
function lpost(body){ return api("api/launch",{method:"POST",headers:{"Content-Type":"application/json","X-Requested-With":"bhaskera"},body:JSON.stringify(body)}); }
function hb(n){ if(n==null) return "–"; return n>=2**30?(n/2**30).toFixed(n>=10*2**30?0:1)+" GB":n>=2**20?(n/2**20).toFixed(0)+" MB":n>=1024?(n/1024).toFixed(0)+" KB":n+" B"; }
var TASK={state:"idle"}, taskTimer=null;
function taskLine(t){
  if(!t||t.state==="idle") return "";
  var pr=t.total?Math.min(100,Math.round(100*(t.done||0)/t.total)):null, running=t.state==="running"||t.state==="hashing";
  return '<div class="dlprog"><div class="row"><span>'+esc(t.message||t.state)+'</span><span>'+(running&&pr!=null?pr+"% · "+hb(t.done)+" of "+hb(t.total):"")+'</span></div>'+
    (running?'<div class="bar"><i style="width:'+(pr==null?8:pr)+'%"></i></div>':"")+'</div>';
}
function showTask(){
  var t=TASK, box=$("#dl-prog"); if(box) box.innerHTML=taskLine(t)+((t.state==="running"||t.state==="hashing")&&t.kind==="download"?'<button class="btn small" id="dl-cancel" type="button">Cancel</button>':"");
  $("#data-prog").textContent=(t.kind==="hash"&&t.state!=="idle")?(t.state==="hashing"?(t.message+(t.total?" · "+Math.min(100,Math.round(100*(t.done||0)/t.total))+"%":"")):t.message):"";
}
function pollTask(){
  lpost({kind:"download",action:"status"}).then(function(r){
    var prev=TASK.state, t=(r.j&&r.j.task)||{state:"idle"}; TASK=t; showTask();
    var running=t.state==="running"||t.state==="hashing";
    if(running){ clearTimeout(taskTimer); taskTimer=setTimeout(pollTask,2000); return; }
    if(prev==="running"||prev==="hashing"){
      toast(t.state==="done"?t.message:(t.state==="cancelled"?"Cancelled.":"Did not finish: "+t.message));
      loadModels();
    }
  }).catch(function(){ clearTimeout(taskTimer); taskTimer=setTimeout(pollTask,5000); });
}
function watchTask(){ clearTimeout(taskTimer); taskTimer=setTimeout(pollTask,400); }

var DL={what:"model",repo:"",rev:"",target:"",preview:null,msg:"",confirm:false,hf:null,hfmsg:"",busy:false};
function dlHf(){
  var h=DL.hf;
  if(h===null) return '<div class="dlcard"><b>Hugging Face access</b><div class="muted">Checking …</div></div>';
  if(h.signed_in) return '<div class="dlcard"><b>Hugging Face access</b><div>Signed in'+(h.user?' as <b>'+esc(h.user)+'</b>':"")+(h.source==="environment"?' <span class="muted">(from the environment)</span>':"")+'. <button class="btn small" id="hf-out" type="button" style="margin-left:8px">Sign out</button></div></div>';
  return '<div class="dlcard"><b>Hugging Face access</b>'+(h.problem?'<div class="muted" style="color:var(--warn)">'+esc(h.problem)+'</div>':"")+
    '<div class="muted" style="margin:4px 0 8px">Not signed in. A key is only needed for private repositories and for models that ask you to accept a licence first (Llama, for example).</div>'+
    '<div style="display:flex;gap:8px"><input id="hf-key" type="password" autocomplete="off" spellcheck="false" placeholder="hf_…" style="flex:1"><button class="btn" id="hf-in" type="button">Sign in</button></div>'+
    '<div class="sub" style="margin-top:6px">Create a key with read access at huggingface.co/settings/tokens. It is checked with Hugging Face, kept in a private file in your own folders and never shown again.</div>'+
    (DL.hfmsg?'<div style="color:var(--bad);margin-top:6px">'+esc(DL.hfmsg)+'</div>':"")+'</div>';
}
function renderDl(){
  var pv=DL.preview, running=TASK.state==="running"||TASK.state==="hashing";
  var h=dlHf()+
   '<div class="dlrow"><label>What</label><div class="chips">'+chip("Model",DL.what==="model",'data-dk="model"')+chip("Dataset",DL.what==="dataset",'data-dk="dataset"')+'</div></div>'+
   '<div class="dlrow"><label for="dl-repo">Repository</label><input id="dl-repo" type="text" value="'+esc(DL.repo)+'" placeholder="'+(DL.what==="model"?"Qwen/Qwen3-14B":"owner/dataset-name")+'" spellcheck="false" autocomplete="off"><div class="sub">The name as it appears on huggingface.co (owner/name). Only Hugging Face is contacted.</div></div>'+
   '<div class="dlrow"><label for="dl-rev">Revision (optional)</label><input id="dl-rev" type="text" value="'+esc(DL.rev)+'" placeholder="main" spellcheck="false" autocomplete="off"></div>'+
   '<div class="dlrow"><label for="dl-target">Save in this folder</label><input id="dl-target" type="text" value="'+esc(DL.target)+'" placeholder="/scratch/<project>/<you>/models/Qwen3-14B" spellcheck="false" autocomplete="off"><div class="sub">A full path you can write to. It is made if it does not exist. Starting the same download again continues where it stopped.</div></div>'+
   '<div style="display:flex;gap:10px;align-items:center"><button class="btn" id="dl-prev" type="button"'+(DL.busy||running?" disabled":"")+'>Check size</button>'+
   '<button class="btn primary" id="dl-go" type="button"'+(!pv||DL.busy||running||(pv.large&&!DL.confirm)?" disabled":"")+'>Download</button></div>';
  if(DL.msg) h+='<div style="margin-top:12px;color:var(--bad)">'+esc(DL.msg)+'</div>';
  if(pv){
    h+='<div class="dlcard" style="margin-top:14px"><b>'+esc(pv.repo)+'</b> <span class="muted">('+esc(pv.revision)+(pv.sha?" · "+esc(pv.sha):"")+')</span><div>'+hb(pv.bytes)+' in '+pv.files+' file'+(pv.files===1?"":"s")+(pv.license?' · licence '+esc(pv.license):"")+'</div>'+
      (pv.gated?'<div style="color:var(--warn)">This repository asks you to accept its licence on huggingface.co first, and to sign in above.</div>':"")+
      (pv.private?'<div class="muted">A private repository.</div>':"")+
      '<div class="muted" style="margin-top:4px">Largest: '+pv.largest.slice(0,3).map(function(f){return esc(f.name)+" ("+hb(f.bytes)+")"}).join(", ")+'</div>'+
      (pv.large?'<label style="display:flex;gap:8px;align-items:center;margin-top:8px"><input type="checkbox" id="dl-confirm"'+(DL.confirm?" checked":"")+'> This is a large download on this machine. Go ahead.</label>':"")+'</div>';
  }
  h+='<div id="dl-prog"></div>';
  $("#dlbody").innerHTML=h; showTask();
}
function dlHfLoad(){ lpost({kind:"hf",action:"status"}).then(function(r){ DL.hf=r.j&&!r.j.error?r.j:{signed_in:false}; if($("#dlbox").open) renderDl(); }).catch(function(){ DL.hf={signed_in:false}; if($("#dlbox").open) renderDl(); }); }
function openDl(what){
  if(what==="model"||what==="dataset"){ if(DL.what!==what){ DL.what=what; DL.preview=null; DL.msg=""; } }
  DL.hf=null; DL.hfmsg=""; renderDl(); $("#dlbox").showModal(); dlHfLoad();
  if(what==="access"){ setTimeout(function(){ var k=$("#hf-key"); if(k) k.focus(); },300); }
  pollTask();
}
$("#dlclose").addEventListener("click",function(){ $("#dlbox").close(); });
$("#dlbody").addEventListener("input",function(e){
  var id=e.target.id; if(id==="dl-repo"){ DL.repo=e.target.value; DL.preview=null; DL.msg=""; $("#dl-go")&&($("#dl-go").disabled=true); }
  else if(id==="dl-rev"){ DL.rev=e.target.value; DL.preview=null; $("#dl-go")&&($("#dl-go").disabled=true); }
  else if(id==="dl-target"){ DL.target=e.target.value; }
  else if(id==="dl-confirm"){ DL.confirm=e.target.checked; $("#dl-go").disabled=!(DL.preview&&(!DL.preview.large||DL.confirm)); }
});
$("#dlbody").addEventListener("click",function(e){
  var t=e.target.closest("button"); if(!t) return;
  if(t.hasAttribute("data-dk")){ DL.what=t.getAttribute("data-dk"); DL.preview=null; DL.msg=""; renderDl(); return; }
  if(t.id==="hf-in"){
    var key=($("#hf-key").value||"").trim(); DL.hfmsg=""; t.disabled=true;
    lpost({kind:"hf",action:"login",key:key}).then(function(r){ $("#hf-key")&&($("#hf-key").value=""); if(r.j&&r.j.signed_in){ DL.hf=r.j; DL.hfmsg=""; toast("Signed in to Hugging Face"+(r.j.user?" as "+r.j.user:"")+"."); } else DL.hfmsg=(r.j&&r.j.error)||"Could not sign in."; renderDl(); }).catch(function(){ DL.hfmsg="The page could not reach the hub."; renderDl(); });
    return;
  }
  if(t.id==="hf-out"){ lpost({kind:"hf",action:"logout"}).then(function(r){ DL.hf={signed_in:false}; DL.hfmsg=""; renderDl(); toast("Signed out of Hugging Face."+((r.j&&r.j.note)?" "+r.j.note:"")); }); return; }
  if(t.id==="dl-prev"){
    DL.msg=""; DL.preview=null; DL.busy=true; t.disabled=true; t.textContent="Checking …";
    lpost({kind:"download",action:"preview",what:DL.what,repo:DL.repo.trim(),revision:DL.rev.trim()}).then(function(r){ DL.busy=false; if(r.j&&r.j.preview){ DL.preview=r.j.preview; DL.confirm=false; } else DL.msg=(r.j&&r.j.error)||"Could not check that repository."; renderDl(); }).catch(function(){ DL.busy=false; DL.msg="The page could not reach the hub."; renderDl(); });
    return;
  }
  if(t.id==="dl-go"){
    DL.msg=""; t.disabled=true;
    lpost({kind:"download",action:"start",what:DL.what,repo:DL.repo.trim(),revision:DL.rev.trim(),target:DL.target.trim(),confirm_large:DL.confirm}).then(function(r){
      if(r.j&&r.j.task){ TASK=r.j.task; renderDl(); watchTask(); } else { DL.msg=(r.j&&r.j.error)||"Could not start the download."; renderDl(); }
    }).catch(function(){ DL.msg="The page could not reach the hub."; renderDl(); });
    return;
  }
  if(t.id==="dl-cancel"){ lpost({kind:"download",action:"cancel"}).then(function(){ watchTask(); }); return; }
});

var DV={name:"",path:"",note:"",msg:""};
function dataRows(){
  var ds=(pathOpts&&pathOpts.datasets)||[], rows=[], n=0;
  ds.forEach(function(d){ d.versions.forEach(function(v,i){
    n++;
    var src=v.source&&v.source.type==="huggingface"?"Hugging Face "+esc(v.source.repo)+(v.source.revision&&v.source.revision!=="main"?" @ "+esc(v.source.revision):""):'<span class="mono" title="'+esc(v.path)+'">'+esc((v.path||"").split("/").slice(-2).join("/"))+'</span>';
    var content=[v.kind==="tokenized"?"tokenised":"raw",v.rows!=null?Number(v.rows).toLocaleString()+" rows":"",v.seq_len?"seq "+v.seq_len:"",v.tokenizer||"",hb(v.bytes)].filter(Boolean).join(" · ");
    var st=v.status==="ok"?'<span class="dstat ok">unchanged</span>':v.status==="changed"?'<span class="dstat changed" title="The files differ from what was registered">changed</span>':'<span class="dstat missing">folder missing</span>';
    rows.push('<tr><td>'+(i===0?'<b>'+esc(d.name)+'</b>':'')+'</td><td>v'+v.version+(i===0?' <span class="chip">latest</span>':'')+'<div class="muted" style="font-size:12px">'+ago(v.created)+'</div></td><td class="muted">'+esc(content)+(v.note?'<div title="'+esc(v.note)+'" style="font-size:12px">'+esc(v.note.length>70?v.note.slice(0,70)+"…":v.note)+'</div>':"")+'</td><td class="muted">'+src+'</td><td>'+st+'</td>'+
      '<td><div class="runact"><button class="btn small" data-dv="verify" data-n="'+esc(d.name)+'" data-v="'+v.version+'">Verify</button>'+(i===0?'<button class="btn small" data-dv="new" data-n="'+esc(d.name)+'" data-p="'+esc(v.path)+'">New version</button>':"")+'<button class="btn small" data-dv="forget" data-n="'+esc(d.name)+'" data-v="'+v.version+'">Forget</button></div></td></tr>');
  }); });
  $("#datacount").textContent=n;
  return rows.length?rows.join(""):'<tr><td colspan="6" class="muted" style="text-align:center;padding:22px">No data is registered yet. Register a tokenised cache or a raw file to give it a name and a version, so every run records exactly what it trained on.</td></tr>';
}
function renderData(){ $("#datarows").innerHTML=dataRows(); showTask(); }
function renderDataDlg(){
  var names=((pathOpts&&pathOpts.datasets)||[]).map(function(d){return d.name}), paths={};
  ((pathOpts&&pathOpts.caches)||[]).forEach(function(c){ paths[c.path]=1; }); ((pathOpts&&pathOpts.datasets)||[]).forEach(function(d){ d.versions.forEach(function(v){ paths[v.path]=1; }); });
  $("#databody").innerHTML='<div class="dlrow"><label for="dv-name">Name</label><input id="dv-name" type="text" list="dv-names" value="'+esc(DV.name)+'" placeholder="support-tickets" spellcheck="false" autocomplete="off"><datalist id="dv-names">'+names.map(function(n){return '<option value="'+esc(n)+'">'}).join("")+'</datalist><div class="sub">Letters, digits, . _ - . A name that exists gets its next version when the content differs.</div></div>'+
    '<div class="dlrow"><label for="dv-path">Folder or file</label><input id="dv-path" type="text" list="dv-paths" value="'+esc(DV.path)+'" placeholder="/scratch/<project>/<you>/data/train.jsonl" spellcheck="false" autocomplete="off"><datalist id="dv-paths">'+Object.keys(paths).map(function(x){return '<option value="'+esc(x)+'">'}).join("")+'</datalist><div class="sub">A tokenised cache folder, or your raw data. Every file is read once to fingerprint it, in the background.</div></div>'+
    '<div class="dlrow"><label for="dv-note">Note (optional)</label><textarea id="dv-note" rows="2" maxlength="2000" placeholder="What changed in this version">'+esc(DV.note)+'</textarea></div>'+
    (DV.msg?'<div style="color:var(--bad)">'+esc(DV.msg)+'</div>':"");
  $("#datago").disabled=false;
}
function openData(name,path){ DV={name:name||"",path:path||"",note:"",msg:""}; renderDataDlg(); $("#databox").showModal(); }
$("#data-add").addEventListener("click",function(){ openData(); });
$("#dataclose").addEventListener("click",function(){ $("#databox").close(); });
$("#databody").addEventListener("input",function(e){ var id=e.target.id; if(id==="dv-name") DV.name=e.target.value; else if(id==="dv-path") DV.path=e.target.value; else if(id==="dv-note") DV.note=e.target.value; });
$("#datago").addEventListener("click",function(){
  $("#datago").disabled=true; DV.msg="";
  lpost({kind:"dataset",action:"add",name:DV.name.trim(),path:DV.path.trim(),note:DV.note}).then(function(r){
    if(r.j&&r.j.task){ TASK=r.j.task; $("#databox").close(); showTask(); watchTask(); } else { DV.msg=(r.j&&r.j.error)||"Could not register that."; renderDataDlg(); }
  }).catch(function(){ DV.msg="The page could not reach the hub."; renderDataDlg(); });
});
$("#datarows").addEventListener("click",function(e){
  var t=e.target.closest("button[data-dv]"); if(!t) return;
  var a=t.getAttribute("data-dv"), n=t.getAttribute("data-n");
  if(a==="new"){ openData(n,t.getAttribute("data-p")); return; }
  if(a==="verify"){ lpost({kind:"dataset",action:"verify",name:n,version:t.getAttribute("data-v")}).then(function(r){ if(r.j&&r.j.task){ TASK=r.j.task; showTask(); watchTask(); } else toast((r.j&&r.j.error)||"Could not check that."); }); return; }
  if(a==="forget"){
    if(t.getAttribute("data-sure")!=="1"){ t.setAttribute("data-sure","1"); t.textContent="Sure? Forget"; setTimeout(function(){ t.removeAttribute("data-sure"); t.textContent="Forget"; },4000); return; }
    lpost({kind:"dataset",action:"remove",name:n,version:t.getAttribute("data-v")}).then(function(r){ toast(r.j&&r.j.removed?"Forgot "+n+" v"+r.j.version+". The files were not touched.":((r.j&&r.j.error)||"Could not forget that.")); loadModels(); });
  }
});

/* ---- registered models ---- */
var regModels=null, pathOpts=null;
function loadModels(){
  fetch("api/launch",{cache:"no-store"}).then(function(r){ return r.json(); }).then(function(o){ regModels=o.models||[]; pathOpts=o; renderModels(); renderMenus(); renderData(); }).catch(function(){});
}
$("#guideclose").addEventListener("click", function(){ $("#guide").close(); });
function renderModels(){
  var rows=[]; (regModels||[]).forEach(function(m){ (m.refs||[]).forEach(function(v){ rows.push({name:m.name,v:v}); }); });
  $("#modcount").textContent=(regModels||[]).length;
  var ok={}; (state&&state.runs||[]).forEach(function(r){ if(r.has_adapter) ok[r.run_id]=1; });
  $("#models").innerHTML=rows.length?rows.map(function(x){
    var v=x.v, act=ok[v.run_id]?'<div class="runact"><button class="btn small primary" data-start="serve" data-run="'+esc(v.run_id)+'">Chat</button><button class="btn small" data-start="eval" data-run="'+esc(v.run_id)+'">Benchmark</button></div>':'<span class="muted">run not in this store</span>';
    return '<tr'+(v.note?' title="'+esc(v.note)+'"':"")+'><td><b>'+esc(x.name)+'</b>'+(v.note?' <span class="muted" title="A hand-off note is kept with this version. Hover over the row to read it.">&#9998;</span>':"")+'</td><td>'+esc(v.version)+'</td><td class="muted">'+esc(v.run_name||v.run_id.slice(0,8))+'</td><td>'+(v.aliases||[]).map(function(a){return '<span class="chip">'+esc(a)+'</span>'}).join(" ")+'</td><td class="num">'+esc(v.eval_loss||"–")+'</td><td>'+act+'</td></tr>';
  }).join(""):'<tr><td colspan="6" class="muted" style="text-align:center;padding:22px">'+(regModels===null?"Loading …":'No model is registered yet. Once a run has finished, register it to keep it under a name. <button class="btn small" data-start="register" style="margin-left:8px">Register a model</button>')+'</td></tr>';
}

/* ---- pages: Home, Train, Serve, Evaluate, Monitor (the address after # chooses one) ---- */
var PAGES={home:"Home",train:"Train",serve:"Serve",evaluate:"Evaluate",monitor:"Monitor"}, pendingScroll=null;
function routeName(){ var m=/^#\/(\w+)/.exec(location.hash||""); return m&&PAGES[m[1]]?m[1]:"home"; }
function setTitle(){ var r=routeName(); document.title=(state&&state.cluster?"Bhaskera · "+state.cluster:"Bhaskera")+(r==="home"?"":" · "+PAGES[r]); }
function route(){
  var r=routeName();
  Array.prototype.forEach.call(document.querySelectorAll(".view"),function(v){ v.hidden=v.getAttribute("data-view")!==r; });
  Array.prototype.forEach.call(document.querySelectorAll("#gnav .gi"),function(g){ g.classList.toggle("here",g.getAttribute("data-page")===r); });
  setTitle();
  var sc=pendingScroll; pendingScroll=null, t=sc&&document.querySelector(sc);
  if(t&&!t.hidden){ if(t.tagName==="DETAILS") t.open=true; t.scrollIntoView({block:"start"}); } else window.scrollTo(0,0);
}
function go(page,scroll){ pendingScroll=scroll||null; var h=page==="home"?"#/":"#/"+page; if((location.hash||"#/")===h) route(); else location.hash=h; }
window.addEventListener("hashchange",route);

/* ---- what the menus and the page cards offer: one definition for both ---- */
var MI={
 text:'<svg viewBox="0 0 24 24"><path d="M5 6h14M5 11h14M5 16h9"/></svg>',
 image:'<svg viewBox="0 0 24 24"><rect x="4" y="5" width="16" height="14" rx="2"/><circle cx="9" cy="10" r="1.6"/><path d="M5 17l5-4 4 3 3-2 3 3"/></svg>',
 data:'<svg viewBox="0 0 24 24"><ellipse cx="12" cy="6.5" rx="6.5" ry="2.5"/><path d="M5.5 6.5v11c0 1.4 2.9 2.5 6.5 2.5s6.5-1.1 6.5-2.5v-11M5.5 12c0 1.4 2.9 2.5 6.5 2.5s6.5-1.1 6.5-2.5"/></svg>',
 code:'<svg viewBox="0 0 24 24"><path d="M9 8l-4 4 4 4M15 8l4 4-4 4"/></svg>',
 play:'<svg viewBox="0 0 24 24"><path d="M8 5.5v13l11-6.5z"/></svg>',
 resume:'<svg viewBox="0 0 24 24"><path d="M4 12a8 8 0 1 0 3-6.2M4 4v4h4"/></svg>',
 chat:'<svg viewBox="0 0 24 24"><path d="M5 6h14a1 1 0 0 1 1 1v8a1 1 0 0 1-1 1h-8l-4 3v-3H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1z"/></svg>',
 box:'<svg viewBox="0 0 24 24"><path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M4 7.5l8 4.5 8-4.5M12 12v9"/></svg>',
 bench:'<svg viewBox="0 0 24 24"><path d="M5 20V10M12 20V4M19 20v-7"/></svg>',
 compare:'<svg viewBox="0 0 24 24"><rect x="4" y="5" width="7" height="14" rx="1.5"/><rect x="13" y="9" width="7" height="10" rx="1.5"/></svg>',
 save:'<svg viewBox="0 0 24 24"><path d="M6 4h12v16l-6-4-6 4z"/></svg>',
 chart:'<svg viewBox="0 0 24 24"><path d="M4 19h16M6 15l4-5 3 3 5-7"/></svg>',
 nodes:'<svg viewBox="0 0 24 24"><circle cx="12" cy="5.5" r="2"/><circle cx="5.5" cy="18" r="2"/><circle cx="18.5" cy="18" r="2"/><path d="M12 7.5v4M12 11.5l-5 5M12 11.5l5 5"/></svg>',
 chip:'<svg viewBox="0 0 24 24"><rect x="7" y="7" width="10" height="10" rx="1.5"/><path d="M10 3v4M14 3v4M10 17v4M14 17v4M3 10h4M3 14h4M17 10h4M17 14h4"/></svg>',
 list:'<svg viewBox="0 0 24 24"><path d="M9 6h11M9 12h11M9 18h11M4.5 6h.01M4.5 12h.01M4.5 18h.01"/></svg>',
 book:'<svg viewBox="0 0 24 24"><path d="M5 5a2 2 0 0 1 2-2h12v16H7a2 2 0 0 0-2 2z"/><path d="M5 19V5"/></svg>',
 help:'<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.5"/><path d="M9.7 9.5a2.4 2.4 0 1 1 3.4 2.2c-.7.4-1.1.9-1.1 1.8M12 16.6h.01"/></svg>',
 folder:'<svg viewBox="0 0 24 24"><path d="M4 7a1 1 0 0 1 1-1h4l2 2h8a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1z"/></svg>',
 wrench:'<svg viewBox="0 0 24 24"><path d="M14.5 6.5a4 4 0 0 0 5 5L11 20a2.1 2.1 0 0 1-3-3z"/></svg>',
 logout:'<svg viewBox="0 0 24 24"><path d="M10 5H6a1 1 0 0 0-1 1v12a1 1 0 0 0 1 1h4M14 8l4 4-4 4M18 12H9"/></svg>'
};
function mi(o){
  var inner='<span class="mic">'+MI[o.ic]+'</span><span class="mt"><b>'+esc(o.t)+'</b>'+(o.d?'<span>'+esc(o.d)+'</span>':"")+'</span>', cls="mi"+(o.cls?" "+o.cls:"")+(o.off?" off":"");
  if(o.href) return '<a class="'+cls+'" role="menuitem" href="'+esc(o.href)+'"'+(o.href.charAt(0)==="#"?"":' target="_blank" rel="noopener"')+'>'+inner+'</a>';
  return '<button type="button" class="'+cls+'" role="menuitem" '+(o.attrs||"")+(o.off?" disabled":"")+'>'+inner+'</button>';
}
function mcol(title,items){ return '<div class="mcol">'+(title?'<div class="mh">'+esc(title)+'</div>':"")+(Array.isArray(items)?items.map(mi).join(""):items)+'</div>'; }
function cards(list){ return list.map(function(o){ return mi(Object.assign({cls:"acard"},o)); }).join(""); }
var DOCS_SOURCE="https://github.com/kishoryd/Bhaskera";
function offer(){
  var s=state||{}, o=pathOpts||{}, pl=s.portal||{}, jobs=s.jobs||[], I={};
  var ready=(o.recipes||[]).filter(function(r){return r.ready}).length;
  var run=jobs.filter(function(j){return j.linked&&(j.kind||"train")==="train"})[0];
  var refs=[]; (regModels||[]).forEach(function(m){ (m.refs||[]).forEach(function(v){ refs.push({m:m.name,v:v}); }); });
  I.text={ic:"text",t:"Text model",d:"Fine-tune a language model",attrs:'data-open="recipe" data-ckind="text"'};
  I.image={ic:"image",t:"Multimodal model",d:"Teach a model to read pictures, clips or speech",attrs:'data-open="recipe" data-ckind="multimodal"'};
  I.prep={ic:"data",t:"Prepare my data",d:"Turn your own files into training data",attrs:'data-open="recipe" data-mode="raw"'};
  I.custom={ic:"code",t:"Custom recipe",d:"Write or paste the whole recipe",attrs:'data-open="recipe" data-method="custom"'};
  I.train={ic:"play",t:"Train from a recipe",d:ready?ready+(ready===1?" recipe is ready":" recipes are ready"):"Pick a recipe and the nodes",attrs:'data-open="train"'};
  I.resume={ic:"resume",t:"Resume a stopped run",d:"Continue from the last checkpoint",attrs:'data-open="resume"'};
  I.chat={ic:"chat",t:"Chat with a model",d:"Ask a trained model; attach pictures or audio",attrs:'data-open="serve"'};
  I.serving=jobs.filter(function(j){return j.kind==="serve"&&j.state==="RUNNING"}).map(function(j){ return {ic:"chat",t:"Chat server #"+j.id,d:"Running now: open the chat",attrs:'data-go="serve" data-scroll="#chat-live"'}; });
  I.saved=refs.slice(0,5).map(function(x){
    var base=(x.v.base_model||"").split("/").filter(Boolean).pop()||"", al=(x.v.aliases||[]).join(", ");
    return {ic:"box",t:x.m+":"+x.v.version,d:[base,al].filter(Boolean).join(" · ")||"saved model",attrs:'data-open="serve" data-run="'+esc(x.v.run_id)+'"'};
  });
  I.models={ic:"list",t:"All saved models",d:"Versions, aliases and notes",attrs:'data-go="serve" data-scroll="#modsec"'};
  I.bench={ic:"bench",t:"Benchmark models",d:"Standard tests or your own, on one or several models",attrs:'data-open="eval"'};
  I.compare={ic:"compare",t:"Compare models",d:"Benchmarked models side by side",attrs:'data-go="evaluate" data-scroll="#cmpsec"'};
  I.register={ic:"save",t:"Save a model",d:"Keep a trained model under a name, with a note",attrs:'data-open="register"'};
  I.mlflow={ic:"chart",t:"MLflow",d:"Experiments, runs and the model registry",href:MLFLOW,off:PROXY&&!pl.mlflow};
  I.ray=PROXY?null:{ic:"nodes",t:"Ray Dashboard",d:run?"Workers, nodes and logs of job #"+run.id:"Opens when a training job is running",href:run?"http://job-"+run.id+".localhost:"+PORT+"/":"#",off:!run};
  I.gpu=PROXY?null:{ic:"chip",t:"GPU charts",d:run?"GPU and node charts of job #"+run.id:"Opens when a training job is running",href:run?"http://grafana-"+run.id+".localhost:"+PORT+"/d/rayDefaultDashboard?orgId=1&from=now-1h&to=now&refresh=10s":"#",off:!run};
  I.jobs={ic:"list",t:"Jobs",d:"Running and queued, with live numbers",attrs:'data-go="home" data-scroll="#jobs"'};
  I.runs={ic:"chart",t:"Runs",d:"Every training run and its results",attrs:'data-go="train" data-scroll="#runsec"'};
  I.cluster={ic:"wrench",t:"Cluster limits and node health",d:"Your job slots and faulty nodes",attrs:'data-go="monitor" data-scroll="#morebox"'};
  I.files=PROXY&&pl.files?{ic:"folder",t:"Files",d:"Your run folder in the portal",href:pl.files}:null;
  I.active=PROXY&&pl.jobs?{ic:"list",t:"Active jobs",d:"The portal's own job list",href:pl.jobs}:null;
  I.refs=refs.length;
  return I;
}
function renderMenus(){
  var I=offer(), s=state||{}, ok=function(a){ return a.filter(Boolean); };
  var go=function(page,label){ return {ic:"play",t:"Open the "+label+" page",d:"",cls:"only-touch",attrs:'data-go="'+page+'"'}; };
  $("#mn-train").innerHTML=mcol("Create a recipe",[go("train","Train"),I.text,I.image,I.prep,I.custom])+mcol("Run",[I.train,I.resume]);
  $("#mn-serve").innerHTML=mcol("Chat",[go("serve","Serve"),I.chat].concat(I.serving,[I.models]))+
    mcol("Saved models",I.saved.length?I.saved:'<div class="mnone">No model saved yet. Register a trained run to keep it here.</div>');
  $("#mn-eval").innerHTML=mcol("",[go("evaluate","Evaluate"),I.bench,I.compare,I.register]);
  $("#mn-monitor").innerHTML=mcol("Tools",[go("monitor","Monitor")].concat(ok([I.mlflow,I.ray,I.gpu])))+mcol("Look at",ok([I.jobs,I.runs,I.cluster,I.files,I.active]));
  var base=/^https?:\/\//.test(s.docs_url||"")?s.docs_url:"";
  $("#mn-dl").innerHTML=mcol("From Hugging Face",[
      {ic:"box",t:"Download a model",d:"Into a folder you choose",attrs:'data-act="dl-model"'},
      {ic:"data",t:"Download a dataset",d:"Registered as a dataset version",attrs:'data-act="dl-dataset"'},
      {ic:"save",t:"Hugging Face access",d:"Sign in with your key for private and gated models",attrs:'data-act="dl-access"'}])+
    mcol("Your data",[
      {ic:"list",t:"Datasets and versions",d:"What each run trained on",attrs:'data-go="train" data-scroll="#datasec"'},
      {ic:"data",t:"Register data",d:"Give a folder a name and a version",attrs:'data-act="data-register"'}]);
  $("#mn-docs").innerHTML=mcol("Learn",[
      {ic:"book",t:"User manual",d:"Everything, with commands",href:base||"#",off:!base},
      {ic:"play",t:"Getting started",d:"From data to a model you can use",href:base?base+"workflow/":"#",off:!base},
      {ic:"help",t:"Words used on this page",d:"What a recipe, an adapter or a step is",attrs:'data-dialog="guide"'}])+
    mcol("Reference",[
      {ic:"text",t:"Recipes and methods",d:"QLoRA, DDP, FSDP2, HSDP, Pipeline",href:base?base+"recipes/":"#",off:!base},
      {ic:"box",t:"Models and the registry",d:"Save, name and find your models",href:base?base+"registry/":"#",off:!base},
      {ic:"wrench",t:"Troubleshooting",d:"Where it breaks and what to do",href:base?base+"troubleshooting/":"#",off:!base},
      {ic:"code",t:"Source code",d:"The project on GitHub",href:DOCS_SOURCE}]);
  $("#mn-user").className="gp acct";
  $("#mn-user").innerHTML='<div class="mu"><b>'+esc(s.user||"")+'</b><span>'+esc(s.cluster||"")+'</span></div>'+mcol("",[{ic:"help",t:"Words used on this page",d:"What a recipe, an adapter or a step is",attrs:'data-dialog="guide"'},
    {ic:"logout",t:"Log out",d:PROXY?"Sign out of the portal":"Stop this page; your jobs keep running",attrs:'data-act="logout"'}]);
  $("#ag-home").innerHTML=cards([I.image,I.chat,I.bench,I.register]);
  $("#ag-train").innerHTML=cards([I.text,I.image,I.prep,I.custom,I.train,I.resume]);
  $("#ag-serve").innerHTML=cards([I.chat].concat(I.serving,[I.register]));
  $("#ag-evaluate").innerHTML=cards([I.bench,I.register]);
  $("#ag-monitor").innerHTML=cards(ok([I.mlflow,I.ray,I.gpu,I.files,I.active,I.jobs,I.runs]));
  // Home: who, and four numbers that lead to the pages
  $("#home-title").textContent=s.user?"Welcome, "+s.user:"Welcome";
  $("#home-sub").textContent="Train, chat with and test models, all in one place.";
  var jobs=s.jobs||[], running=jobs.filter(function(j){return j.state==="RUNNING"}).length, queued=jobs.filter(function(j){return j.state==="PENDING"}).length;
  var runs=s.runs||[], benched=runs.filter(function(r){return r.lm_scores&&Object.keys(r.lm_scores).length}).length;
  var dk=s.disk, dw=$("#diskwarn");
  if(dk&&dk.low){ dw.hidden=false; dw.textContent="The disk with your run folder ("+dk.path+") has only "+dk.free_gb+" GB free of "+dk.total_gb+" GB. Checkpoints can be large (about 1 GB for a LoRA run of a 14B model, but around 140 GB for a full fine-tune of it): save them on a disk with more room (New run, Create recipe, Checkpoints, Save them in), or clear old runs and checkpoints."; } else { dw.hidden=true; }
  var energyTotal=null; runs.forEach(function(r){ if(r.energy_kwh!=null) energyTotal=(energyTotal||0)+r.energy_kwh; });
  var tile=function(k,v,attrs){ return '<button type="button" class="sumtile" '+attrs+'><span class="k">'+esc(k)+'</span><span class="v">'+v+'</span></button>'; };
  $("#sumrow").innerHTML=tile("Running now",running+(queued?' <small>· '+queued+' queued</small>':""),'data-go="home" data-scroll="#jobs"')+tile("Training runs",runs.length,'data-go="train" data-scroll="#runsec"')+
    tile("Saved models",(regModels||[]).length,'data-go="serve" data-scroll="#modsec"')+tile("Benchmarked",benched,'data-go="evaluate"')+
    (energyTotal!=null?tile("GPU energy used",kwh(energyTotal),'data-go="train" data-scroll="#runsec"'):"");
}
var menuOpen=null, menuOpenT=null, menuCloseT=null, TOUCH=window.matchMedia&&window.matchMedia("(hover: none) and (pointer: coarse)").matches;
function setMenu(el,on){ el.classList.toggle("open",on); var b=el.querySelector(".gb"); if(b) b.setAttribute("aria-expanded",on?"true":"false"); }
function closeMenus(){ clearTimeout(menuOpenT); clearTimeout(menuCloseT); document.querySelectorAll(".gnav .gi.open").forEach(function(e){ setMenu(e,false); }); menuOpen=null; }
Array.prototype.forEach.call(document.querySelectorAll("#gnav .gi, #unav .gi"),function(el){
  el.addEventListener("mouseenter",function(){ if(TOUCH) return; clearTimeout(menuCloseT); clearTimeout(menuOpenT); menuOpenT=setTimeout(function(){ if(menuOpen&&menuOpen!==el) setMenu(menuOpen,false); setMenu(el,true); menuOpen=el; },60); });
  el.addEventListener("mouseleave",function(){ clearTimeout(menuOpenT); menuCloseT=setTimeout(function(){ setMenu(el,false); if(menuOpen===el) menuOpen=null; },170); });
  el.querySelector(".gb").addEventListener("click",function(e){
    e.stopPropagation(); var page=el.getAttribute("data-page");
    if(page&&!TOUCH){ closeMenus(); go(page); return; }                                  // with a mouse the name is a link to its page; the menu opens on hover
    var on=!el.classList.contains("open"); closeMenus(); if(on){ setMenu(el,true); menuOpen=el; }
  });
});
document.addEventListener("click",function(e){ if(!e.target.closest(".gnav")) closeMenus(); });
document.addEventListener("keydown",function(e){ if(e.key==="Escape") closeMenus(); });
$("#brand").addEventListener("click",function(e){ e.preventDefault(); closeMenus(); go("home"); });
$("#startbtn").addEventListener("click",function(){ L.hours=""; openStart("recipe"); });
document.addEventListener("click",function(e){
  var it=e.target.closest(".mi,.sumtile"); if(!it||it.classList.contains("off")) return;
  closeMenus();
  var kind=it.getAttribute("data-open"), gp=it.getAttribute("data-go"), sc=it.getAttribute("data-scroll"), dg=it.getAttribute("data-dialog"), act=it.getAttribute("data-act");
  if(act==="logout"){ askLogout(); return; }
  if(act&&act.indexOf("dl-")===0){ openDl(act.slice(3)); return; }
  if(act==="data-register"){ openData(); return; }
  if(kind){
    L.hours="";
    if(kind==="recipe"){
      var ck=it.getAttribute("data-ckind"), mode=it.getAttribute("data-mode"), meth=it.getAttribute("data-method");
      if(ck){ var cm=((L.opts&&L.opts.model_catalog)||[]).filter(function(m){return m.path===L.rc.model})[0]; L.rc.kind=ck; if(cm&&cm.kind!==ck){ L.rc.model=""; L.rc.modelOther=false; } }
      if(meth) L.rc.method=meth;
      else if(ck&&(L.rc.method==="custom"||(ck==="multimodal"&&L.rc.method==="qlora"))) L.rc.method=ck==="multimodal"?"fsdp":"qlora";
      if(mode) L.rc.dataMode=mode; else if(ck) L.rc.dataMode="";
    }
    openStart(kind, it.getAttribute("data-run")); return;
  }
  if(dg){ $("#"+dg).showModal(); return; }
  if(gp){ go(gp, sc); return; }
  if(sc){ var t=document.querySelector(sc); if(t&&!t.hidden){ if(t.tagName==="DETAILS") t.open=true; t.scrollIntoView({behavior:"smooth",block:"start"}); } }
});

/* ---- log out ---- */
function askLogout(){
  $("#bye-text").textContent=PROXY?"This signs you out of the portal. Jobs you started keep running, and you can open Bhaskera again from the portal.":
    "This stops the page on the login node and closes its links to your jobs. Jobs you started keep running. To open the page again, run the same command you used to start it.";
  $("#bye-ok").disabled=false; $("#bye").showModal();
}
$("#bye-cancel").addEventListener("click",function(){ $("#bye").close(); });
$("#bye-ok").addEventListener("click",function(){
  $("#bye-ok").disabled=true;
  api("api/logout",{method:"POST",headers:{"Content-Type":"application/json","X-Requested-With":"bhaskera"},body:"{}"}).then(function(r){
    $("#bye").close();
    if(!r.j||!r.j.ok){ toast("Could not log out: "+((r.j&&r.j.error)||"the hub did not answer")); return; }
    if(r.j.mode==="portal"){ var pl=(state&&state.portal)||{}; location.href=pl.logout||"/logout"; return; }
    document.body.innerHTML='<div class="byepage"><svg class="logo" viewBox="0 0 36 36" style="width:56px;height:56px" aria-hidden="true"><rect width="36" height="36" rx="10" fill="var(--accent)"/><path d="M9.5 19.5a8.5 8.5 0 0 1 17 0z" fill="#fff"/><rect x="8" y="22" width="20" height="2.4" rx="1.2" fill="#fff"/><rect x="11.5" y="26" width="13" height="2.4" rx="1.2" fill="#fff" opacity=".85"/><rect x="15" y="30" width="6" height="2.4" rx="1.2" fill="#fff" opacity=".7"/></svg>'+
      '<h1>Bhaskera has stopped</h1><p>You are logged out. Your jobs keep running on the cluster.<br>To open the page again, run the same command you used to start it, then reload this tab.</p></div>';
  }).catch(function(){ $("#bye").close(); toast("The page could not reach the hub."); });
});

/* ---- compare benchmarked models ---- */
var cmpSel={};
function renderCompare(){
  var runs=(state.runs||[]).filter(function(r){return r.lm_scores&&Object.keys(r.lm_scores).length});
  $("#cmpsec").hidden=!runs.length; $("#cmpempty").hidden=runs.length>0; if(!runs.length) return;
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
  $("#who").hidden=false; $("#who").innerHTML='<span class="av">'+esc((s.user||"?").charAt(0).toUpperCase())+'</span>'+esc(s.user)+'<svg viewBox="0 0 12 12"><path d="M2.5 4.5l3.5 3.5 3.5-3.5"/></svg>';
  $("#tag").textContent = s.cluster || ""; $("#tag").hidden = !s.cluster;
  setTitle();
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
  renderRuns(); renderCompare(); renderModels(); renderSide(); renderMenus(); renderChat();
  $("#foot").innerHTML='Bhaskera '+esc(s.version||"")+' · <a href="'+esc(safe(s.docs_url)||"#")+'" target="_blank" rel="noopener">documentation</a> · <a href="'+GITHUB+'" target="_blank" rel="noopener">source</a>';
}
function clock(){
  var dot=$("#dot"), el=$("#updated");
  if(!lastOk){ return; }
  var age=Math.round((Date.now()-lastOk)/1000);
  dot.classList.toggle("stale", age>30);
  el.title = age>30 ? "The page lost its connection and is retrying." : ""; el.textContent = age>30 ? "reconnecting …" : (age<3 ? "live" : "updated "+age+" s ago");
}
function refresh(){
  fetch("api/state",{cache:"no-store"}).then(function(r){ if(!r.ok) throw new Error(r.status); return r.json(); })
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
  fetch("api/log?job="+encodeURIComponent(logJob)+"&lines=400",{cache:"no-store"}).then(function(r){return r.json()}).then(function(d){
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
    live.hidden=true; info.hidden=!sj.length; chat.loading=chat.loading||{}; sj.forEach(function(j){ chat.loading[j.id]=true; });
    if(sj.length) info.innerHTML='Loading the model for chat (job #'+esc(sj[0].id)+'). This takes a few minutes; the chat opens here by itself.';
    return;
  }
  info.hidden=true; live.hidden=false;
  var cur=ready.filter(function(j){return j.id===chat.job})[0]||ready[0];
  chat.job=cur.id;
  if(chat.loading&&chat.loading[cur.id]&&!(chat.announced||{})[cur.id]){ (chat.announced=chat.announced||{})[cur.id]=true; if(routeName()!=="serve"){ toast("The chat for job #"+cur.id+" is ready."); go("serve","#chat-live"); } }
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
  return fetch("api/chat?job="+encodeURIComponent(chat.job),{method:"POST",signal:ctrl.signal,
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
function startClick(e){ var s=e.target.closest("[data-start]"); if(s) openStart(s.getAttribute("data-start"), s.getAttribute("data-run")); }
["#runs","#jobs","#models"].forEach(function(id){ $(id).addEventListener("click", startClick); });      // the buttons of the Runs table, the empty Jobs box and the Models list

/* ---- cancel a job, close a chat server ---- */
function doCancel(id,label,btn){
  var now=Date.now();
  if(!(armed.id===String(id)&&now<armed.until)){            // the first click only arms the button
    armed={id:String(id),until:now+5000}; if(btn){ btn.classList.add("confirm"); btn.textContent="Click again to "+label.toLowerCase(); }
    setTimeout(function(){ if(armed.id===String(id)&&Date.now()>=armed.until){ armed={id:null,until:0}; refresh(); } },5200); return;
  }
  armed={id:null,until:0}; if(btn){ btn.disabled=true; btn.textContent="Cancelling …"; }
  api("api/launch",{method:"POST",headers:{"Content-Type":"application/json","X-Requested-With":"bhaskera"},body:JSON.stringify({action:"start",kind:"cancel",job:String(id)})}).then(function(r){
    if(r.j.error){ toast(r.j.error); if(btn){ btn.disabled=false; btn.textContent=label; } return; }
    toast(r.j.ok?("Job #"+id+" cancelled. It disappears from this page in a few seconds."):("Could not cancel job #"+id+": "+r.j.output)); refresh();
  }).catch(function(){ toast("The page could not reach the hub."); if(btn){ btn.disabled=false; btn.textContent=label; } });
}
$("#jobs").addEventListener("click", function(e){ var b=e.target.closest("[data-cancel]"); if(b) doCancel(b.getAttribute("data-cancel"),b.getAttribute("data-label")||"Cancel job",b); });
$("#chat-close").addEventListener("click", function(){ if(chat.job) doCancel(chat.job,"Close server",$("#chat-close")); });

/* ---- start a job ---- */
var L = {opts:null, kind:"train", run:"", note:"", evalSel:[], own:{name:"",data:"",q:"question",a:"answer"}, shots:"", skipBase:false, rinfo:null, step:0, regname:"", alias:"",
  rc:{name:"my-run",kind:"text",mods:["image"],model:"",method:"qlora",train:"",val:"",steps:"300",lr:"1e-4",rank:"16",batch:"",accum:"",every:"",ckpt:"on",keep:"",ckdir:"",tuning:"lora",evals:"",hsdp:"",stages:"",micro:"",yaml:"",copy:false,base:"",settings:{},open:{},raw:{train:"",val:"",format:"chatml",seq:"",media:""},modelOther:false,dataMode:"",trainOther:false,valOther:false,trainPath:"",valPath:""}, nodes:2, hours:"", tasks:{arc_easy:1,hellaswag:1}, quick:true, confirm:false, busy:false};
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
        '<select id="f-recipe">'+ready.map(function(r){return '<option value="'+esc(r.file)+'"'+(r.file===L.recipe?" selected":"")+'>'+esc(recipeName(r.file))+'</option>'}).join("")+'</select><div class="sub" id="f-recipe-info"></div><div class="sub">A recipe is the saved list of choices for one training run. Make a new one under Create recipe.</div>' :
        '<div class="sub">No recipe is ready yet. Use <b>Create recipe</b> in the Start menu to make one.</div>')+
      (blocked.length?'<div class="sub">'+blocked.length+' more '+(blocked.length>1?"need":"needs")+' editing first (they still have <code>&lt;project&gt;/&lt;username&gt;</code> placeholders): '+blocked.slice(0,3).map(function(r){return esc(recipeName(r.file))}).join(", ")+(blocked.length>3?", …":"")+'</div>':'')+'</div>'+
      '<div class="srow"><label class="lbl">Nodes</label><div class="chips">'+[1,2,4,8,16,32].filter(function(n){return n<=o.max_nodes}).map(function(n){return chip(n,L.nodes===n,'data-n="'+n+'"')}).join("")+
        '<span class="muted" style="margin:0 4px 0 8px">or</span><input id="f-nodes" type="number" min="1" max="'+o.max_nodes+'" value="'+L.nodes+'"></div><div class="sub">A node is one machine'+(o.gpus_per_node?" with "+o.gpus_per_node+" GPUs":"")+'. More nodes train faster but may wait longer in the queue. One or two is plenty for a first run.</div></div>'+
      '<div class="srow"><label class="lbl">Time limit</label>'+hoursRow(["",1,2,6,12,24],"")+'<div class="sub">The job is stopped when this runs out. Checkpoints are saved along the way, so a stopped run can be resumed.</div></div>';
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
       '<div class="sub">An alias is a label you choose that points at one version; move it to a newer version later. '+(aliasNames().length?"Already in use: "+esc(aliasNames().join(", "))+". ":"")+'Chat and benchmark accept <code>name@alias</code>.</div></div>'+
       '<div class="srow"><label class="lbl" for="f-note">Hand-off note (optional)</label><textarea id="f-note" rows="4" maxlength="2000" style="width:100%" placeholder="What is this model for? Which data was it trained on? Anything the next person should know, such as limits or things it gets wrong.">'+esc(L.note)+'</textarea>'+
       '<div class="sub">Kept with the version in the registry, together with the base model, settings and scores that are filled in for you, so whoever picks it up knows what it is and how to use it.</div></div>';
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
function setTuningLr(t){ var full=t==='full'; if(full&&['5e-5','1e-4','2e-4'].indexOf(L.rc.lr)>=0) L.rc.lr='1e-5'; if(!full&&['5e-6','1e-5','2e-5'].indexOf(L.rc.lr)>=0) L.rc.lr='1e-4'; }
function cacheLabel(c){ return (c.data_version?c.data_version.name+" v"+c.data_version.version+(c.data_version.status==="changed"?" (changed) · ":" · "):"")+c.name+" · "+(c.model||"?")+" · sequence "+(c.seq_len||"?")+" · "+(c.rows!=null?Number(c.rows).toLocaleString()+" rows":""); }
function cacheOptions(list,sel,none,other){
  var group=function(items,label){ return items.length?'<optgroup label="'+label+'">'+items.map(function(c){ return '<option value="'+esc(c.path)+'"'+(!other&&c.path===sel?" selected":"")+'>'+esc(cacheLabel(c))+'</option>'; }).join("")+'</optgroup>':""; };
  var sh=list.filter(function(c){return c.shared}), mine=list.filter(function(c){return !c.shared});
  return (none?'<option value="">none (no evaluation during training)</option>':'')+(sh.length&&mine.length?group(sh,"Shared on this cluster")+group(mine,"Yours"):list.map(function(c){ return '<option value="'+esc(c.path)+'"'+(!other&&c.path===sel?" selected":"")+'>'+esc(cacheLabel(c))+'</option>'; }).join(""))+
    '<option value="__other"'+(other?" selected":"")+'>Another folder (type its path) …</option>';
}
function modelOptions(cat,sel,other){
  var item=function(m){ return '<option value="'+esc(m.path)+'"'+(!other&&m.path===sel?" selected":"")+'>'+esc(m.name+" · "+(m.kind==="multimodal"?"multimodal":"text")+(m.size_gb?" · "+m.size_gb+" GB":"")+(m.custom_code?" · own code":""))+'</option>'; };
  var sh=cat.filter(function(m){return m.group==="shared"}), mine=cat.filter(function(m){return m.group!=="shared"});
  return (sh.length?'<optgroup label="Shared on this cluster">'+sh.map(item).join("")+'</optgroup>':"")+(mine.length?'<optgroup label="Yours">'+mine.map(item).join("")+'</optgroup>':"")+
    '<option value="__other"'+(other?" selected":"")+'>Another folder (type its path) …</option>';
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
  if(!r.trainOther&&(!r.train||!caches.some(function(c){return c.path===r.train}))) r.train=(tr[0]||{}).path||"";
  if(!r.valOther&&r.val!==""&&!caches.some(function(c){return c.path===r.val})) r.val=(va.filter(function(c){return c.path!==r.train})[0]||{}).path||"";
  if(!r.valOther&&r.val===""&&L.rcValTouched!==true) r.val=(va.filter(function(c){return c.path!==r.train})[0]||{}).path||"";
  var cat=o.model_catalog||[];
  if(!r.model&&!r.modelOther&&cat.length){ var first=cat.filter(function(m){return m.kind===r.kind})[0]||cat[0]; r.model=first.path; r.kind=first.kind; }
  var mode=r.dataMode||(caches.length?"ready":"raw");
  var methods=Object.keys(o.methods||{}).filter(function(m){return methodOk(o,m,r.kind)});
  if(methods.indexOf(r.method)<0) r.method=methods[0]||"fsdp";
  var multi=r.kind==="multimodal", custom=r.method==="custom";
  var h='<div class="srow intro"><b>What is a recipe?</b> It is the saved list of choices for one training run: which model to teach, which examples to teach it with, and for how long. You make it once here, then <b>Train</b> runs it on the cluster. Unsure about a choice? Leave it as it is: the defaults suit a first run.</div><div class="sec">The model</div><div class="srow half"><label class="lbl" for="f-rname">Name of the recipe</label><input id="f-rname" type="text" value="'+esc(r.name)+'" maxlength="61"><div class="sub">A short name for this run. It is the file name (<code>recipes/'+esc(r.name||"name")+'.yaml</code>) and the name you will see in MLflow.</div></div>'+
    '<div class="srow half"><label class="lbl">Kind of model</label><div class="chips">'+chip("Text",!multi,'data-kind="text"')+chip("Multimodal (images, video, audio)",multi,'data-kind="multimodal"')+'</div><div class="sub">Text models read and write words. Multimodal models also look at images, video or audio.</div></div>'+
    (multi?'<div class="srow"><label class="lbl">The data holds</label><div class="chips">'+["image","video","audio"].map(function(m){return chip(m,r.mods.indexOf(m)>=0,'data-mod="'+m+'"')}).join("")+'<span class="muted">besides text</span></div></div>':"")+
    '<div class="srow"><label class="lbl">Method</label><div class="chips">'+methods.map(function(m){return chip(o.methods[m].label+(m===(multi?"fsdp":"qlora")?" · start here":""),r.method===m,'data-method="'+m+'"')}).join("")+'</div><div class="sub">'+esc((o.methods[r.method]||{}).about||"")+'</div><div class="sub">The method decides how the model is spread over the GPUs. If you are new, keep the one marked start here.</div></div>'+
    (r.method==="custom"?"":'<div class="srow"><label class="lbl">What to train</label><div class="chips">'+chip("A small add-on (LoRA)",r.tuning!=="full",'data-tuning="lora"')+chip("The whole model (full fine-tune)",r.tuning==="full",'data-tuning="full"')+'</div>'+
      '<div class="sub">'+(r.tuning==="full"?"Trains every weight of the model. It can learn more, but it needs far more GPU memory (every weight, its gradient and the optimizer state) and disk for the checkpoints, so use FSDP2 or HSDP on several nodes. The result stays in the checkpoints: Chat, Benchmark and Register on this page need an adapter, so they are not offered for a full fine-tune yet. Needs a smaller learning rate, which is set for you.":
        "LoRA keeps the model as it is and trains a small add-on on top: far less memory and disk, and the usual choice. QLoRA does the same on a model squeezed to 4 bits.")+'</div></div>');
  if(custom){ h+='<div class="sec">The recipe</div>'+customForm(o,r); }
  else {
    var typed=r.modelOther||!cat.length||(r.model&&!cat.some(function(m){return m.path===r.model}));
    h+='<div class="srow"><label class="lbl" for="f-rmodelsel">Base model</label>'+(cat.length?'<select id="f-rmodelsel">'+modelOptions(cat,r.model,typed)+'</select>':"")+
      (typed?'<input id="f-rmodel" type="text" value="'+esc(r.model)+'" placeholder="/path/to/'+(multi?"Qwen3-VL-8B-Instruct":"Qwen3-14B")+'"'+(cat.length?' style="margin-top:8px"':"")+'>':"")+
      '<div class="sub">The model you want to teach. Pick one shared on this cluster, or type the path of your own Hugging Face model folder. It has to be on disk already: the machines that run training have no internet.</div>'+
      ((cat.filter(function(m){return m.path===r.model&&m.custom_code})[0])?'<div class="sub"><b>This model comes with its own code</b> (the Param models do), so the recipe turns on <code>trust_remote_code</code>. Only use model folders you trust.</div>':"")+'</div>';
    h+='<div class="sec">The data</div><div class="srow"><label class="lbl">Which data</label><div class="chips">'+chip("Data that is already prepared (tokenised)",mode==="ready",'data-dmode="ready"')+chip("My own files: prepare them here",mode==="raw",'data-dmode="raw"')+'</div>'+
       '<div class="sub">The model reads numbers, not text. Prepared data is your examples converted to numbers once; a new set of files needs that step first.</div></div>';
    if(mode==="ready"){
      var trOther=r.trainOther||!tr.length, vaOther=r.valOther;
      h+='<div class="srow half"><label class="lbl" for="f-rtrain">Training data</label>'+(tr.length?'<select id="f-rtrain">'+cacheOptions(tr,r.train,false,trOther)+'</select>':"")+
         (trOther?'<input id="f-rtrain-path" type="text" value="'+esc(r.trainPath)+'" placeholder="/path/to/the/tokenised/training/folder"'+(tr.length?' style="margin-top:8px"':"")+'>':"")+
         '<div class="sub">The examples the model learns from, already turned into numbers.'+(tr.length?"":" None was found in the usual places: type the folder, or prepare your own files with the other choice above.")+'</div></div>'+
         '<div class="srow half"><label class="lbl" for="f-rval">Validation data</label><select id="f-rval">'+cacheOptions(va,r.val,true,vaOther)+'</select>'+
         (vaOther?'<input id="f-rval-path" type="text" value="'+esc(r.valPath)+'" placeholder="/path/to/the/tokenised/validation/folder" style="margin-top:8px">':"")+
         '<div class="sub">Examples kept aside to check progress. The model never learns from them, so a better score here means it is really improving. Optional.</div></div>';
    } else {
      h+='<div class="srow"><div class="sub">Give the JSONL files (one conversation per line'+(multi?", with the paths of the images, video or audio in each row":"")+'). A short CPU job converts them once. When it has finished, choose <b>Data that is already prepared</b> above: the result is listed there.</div>'+
         '<input id="f-rawtrain" type="text" value="'+esc(r.raw.train)+'" placeholder="training file: /path/to/train.jsonl" style="margin-top:8px">'+
         '<input id="f-rawval" type="text" value="'+esc(r.raw.val)+'" placeholder="validation file (optional)" style="margin-top:6px">'+
         (multi?'<input id="f-rawmedia" type="text" value="'+esc(r.raw.media)+'" placeholder="folder the media paths in the file are relative to (optional)" style="margin-top:6px">':"")+
         '<div class="chips" style="margin-top:8px">'+["chatml","alpaca","sharegpt","prompt_completion","dolly","qa","scored_response","tool_calling","reasoning"].map(function(f){return chip(f,r.raw.format===f,'data-fmt="'+f+'"')}).join("")+
         '<label class="muted">Sequence length <input id="f-rawseq" type="number" min="128" value="'+esc(r.raw.seq)+'" placeholder="'+(multi?"1280":"4096")+'"></label>'+
         '<button class="btn small" id="f-rawgo" type="button">Prepare data</button></div></div>';
    }
    h+='<div class="sec">The training</div><div class="srow"><label class="lbl">How long</label><div class="chips">'+["100","300","1000","3000"].map(function(s){return chip(s+" steps",r.steps===s,'data-steps="'+s+'"')}).join("")+
       '<span class="muted" style="margin:0 4px 0 8px">or</span><input id="f-rsteps" type="number" min="1" value="'+(["100","300","1000","3000"].indexOf(r.steps)<0?esc(r.steps):"")+'" placeholder="steps"></div><div class="sub">One step teaches the model one batch of examples. 100 to 300 steps is a quick trial; use 1000 or more for a real run.</div></div>'+
       '<div class="srow half"><label class="lbl">Learning rate</label><div class="chips">'+(r.tuning==="full"?[["5e-6","5e-6"],["1e-5","1e-5 (usual)"],["2e-5","2e-5"]]:[["5e-5","5e-5"],["1e-4","1e-4 (usual)"],["2e-4","2e-4"]]).map(function(x){return chip(x[1],r.lr===x[0],'data-lr="'+x[0]+'"')}).join("")+'</div><div class="sub">How big each learning step is. The usual value suits most runs; try a lower one if the loss jumps around.</div></div>'+
       (r.tuning==="full"?"":'<div class="srow half"><label class="lbl">LoRA rank</label><div class="chips">'+["8","16","32"].map(function(x){return chip(x,r.rank===x,'data-rank="'+x+'"')}).join("")+'</div><div class="sub">Instead of changing the whole model, LoRA trains a small add-on. A larger rank can learn more and uses a little more memory.</div></div>')+
       '<div class="srow"><label class="lbl">Checkpoints</label><div class="chips">'+chip("Save them",r.ckpt!=="off"||r.tuning==="full",'data-ckpt="on"')+(r.tuning==="full"?"":chip("Do not save",r.ckpt==="off",'data-ckpt="off"'))+
         ((r.ckpt==="off"&&r.tuning!=="full")?"":'<span class="muted" style="margin:0 4px 0 8px"></span><label class="muted">every <input id="f-revery" type="number" min="1" value="'+esc(r.every)+'" placeholder="auto" style="width:90px"> steps</label>'+
           '<label class="muted">keep the newest <input id="f-rkeep" type="number" min="0" value="'+esc(r.keep)+'" placeholder="3" style="width:70px"></label>')+'</div>'+
         ((r.ckpt==="off"&&r.tuning!=="full")?"":'<div class="chips" style="margin-top:8px"><label class="muted" style="flex:1">Save them in <input id="f-rckdir" type="text" value="'+esc(r.ckdir)+'" placeholder="default: checkpoints/ in your run folder" spellcheck="false" autocomplete="off" style="width:100%"></label></div>')+
         '<div class="sub">'+((r.ckpt==="off"&&r.tuning!=="full")?"Nothing is saved while training, so a run that stops cannot be continued with Resume; it has to start again. Only the final adapter is kept.":
           "A checkpoint is the saved state of the run, so one that stops can continue from it with Resume. Older ones are deleted so that only the newest few stay. "+(r.tuning==="full"?"Each one holds every weight and the optimizer state, about five times the size of the model (an estimate: around 140 GB for a 14B model), so keep few and use a large disk.":"Each one holds only the trained add-on and its optimizer state (about 1 GB for a 14B model).")+" (0 keeps all.) Home folders fill up fast on a cluster: give a folder on a large disk (for example under /scratch) in Save them in. The recipe gets its own folder inside it, and Check first tells you whether there is room.")+'</div></div>'+
       det(r,"more","More options",'<div class="chips" style="margin-top:8px">'+
         '<label class="muted">Batch size <input id="f-rbatch" type="number" min="1" value="'+esc(r.batch)+'" placeholder="auto"></label>'+
         '<label class="muted">Gradient accumulation <input id="f-raccum" type="number" min="1" value="'+esc(r.accum)+'" placeholder="1"></label>'+
         '<label class="muted">Evaluate every <input id="f-revals" type="number" min="0" value="'+esc(r.evals)+'" placeholder="auto"> steps</label>'+
         (r.method==="hsdp"?'<label class="muted">GPUs per shard group <input id="f-rhsdp" type="number" min="0" value="'+esc(r.hsdp)+'" placeholder="one node"></label>':"")+
         (r.method==="pipeline"?'<label class="muted">Stages <input id="f-rstages" type="number" min="2" value="'+esc(r.stages)+'" placeholder="4"></label><label class="muted">Micro-batches <input id="f-rmicro" type="number" min="1" value="'+esc(r.micro)+'" placeholder="2 x stages"></label>':"")+'</div>'+
         '<div class="sub">Batch size: examples each GPU handles per step. Gradient accumulation: gather several batches before each update.  Evaluate every: how often to check against the validation data.</div>'+
         (r.method==="hsdp"||r.method==="pipeline"?'<div class="sub">'+(r.method==="hsdp"?"The shard group must divide the number of GPUs you start with.":"The number of GPUs must divide by the stages.")+' The check on the Train page tells you if it does not.</div>':""));
  }
  h+='<div class="sec">'+(custom?"Settings on top":"More")+'</div>'+det(r,"all","All settings ("+((o.schema||[]).reduce(function(n,g){return n+g.fields.length},0))+" options)",allSettings(o,r));
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
  api("api/resume?run="+encodeURIComponent(run),{headers:{"X-Requested-With":"bhaskera"}}).then(function(r){
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
    var tc=(o.caches||[]).filter(function(c){return c.path===rc.train})[0], mname=(rc.model||"").split("/").filter(Boolean).pop()||"the model";
    el.textContent=rc.method==="custom"?"Saves the recipe exactly as you wrote it. Nothing runs until you start it under Train.":"Saves a recipe that teaches "+mname+(tc&&tc.rows!=null?" with "+Number(tc.rows).toLocaleString()+" examples":"")+" for "+rc.steps+" steps ("+lab+(rc.tuning==="full"?", all weights":"")+"). Nothing runs yet: start it afterwards under Train."; return;
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
      f.model=r.model; f.train_cache=r.trainOther?r.trainPath:r.train; f.val_cache=r.valOther?r.valPath:r.val; f.steps=r.steps; f.lr=r.lr; f.rank=r.rank; f.tuning=r.tuning;
      if(r.batch) f.batch_size=r.batch; if(r.accum) f.grad_accum=r.accum; if(r.every) f.save_every_steps=r.every; f.checkpoints=r.tuning==="full"?true:r.ckpt!=="off"; if(r.keep!=="") f.keep_last_n=r.keep; if(r.ckdir.trim()) f.checkpoint_dir=r.ckdir.trim(); if(r.evals!=="") f.eval_steps=r.evals;
      if(r.method==="hsdp"&&r.hsdp!=="") f.hsdp_shard_size=r.hsdp; if(r.method==="pipeline"){ if(r.stages) f.stages=r.stages; if(r.micro) f.microbatches=r.micro; }
    }
  }
  else if(L.kind==="register"){ f.run=val("f-run"); f.name=val("f-name"); f.alias=L.alias; f.note=L.note; }
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
  api("api/launch",{method:"POST",headers:{"Content-Type":"application/json","X-Requested-With":"bhaskera"},body:JSON.stringify(f)}).then(function(r){
    if(r.j.error){ showResult("bad","Not started",r.j.error); return; }
    if(r.j.dry&&L.kind==="recipe"){ showResult("ok","This is the recipe that would be written. Nothing was saved yet.",r.j.output); return; }
    if(r.j.ok&&L.kind==="recipe"){
      L.recipe=r.j.recipe; toast("Saved the recipe "+r.j.recipe+". Next: choose how many nodes and press Start to train it.");
      L.kind="train"; L.step=0; api("api/launch",{headers:{"X-Requested-With":"bhaskera"}}).then(function(o){ L.opts=o.j; renderStart(); }); return;
    }
    if(r.j.dry){ showResult(r.j.ok?"ok":"bad", r.j.ok?"Everything checks out. Nothing was submitted.":"A check failed. Nothing was submitted.", r.j.output); return; }
    if(r.j.ok&&L.kind==="register"){
      toast(r.j.output.split("\n")[0]+". It is listed under Models."); $("#startbox").close(); loadModels(); return;
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
  api("api/launch",{method:"POST",headers:{"Content-Type":"application/json","X-Requested-With":"bhaskera"},body:JSON.stringify(f)}).then(function(x){
    if(x.j.error){ showResult("bad","Not added",x.j.error); return; }
    if(!x.j.ok){ showResult("bad","Not added",x.j.output); return; }
    L.tasks[o.name]=1; L.own={name:"",data:"",q:"question",a:"answer"};
    api("api/launch",{headers:{"X-Requested-With":"bhaskera"}}).then(function(r){ L.opts=r.j; renderStart(); toast("Added the test "+o.name+" and ticked it."); });
  }).catch(function(){ showResult("bad","Not added","The page could not reach the hub."); });
}
function sendEvalMany(action){
  var runs=L.evalSel.slice(), base=formData(action), jobs=[], notes=[];
  L.busy=true; $("#startcheck").disabled=$("#startgo").disabled=true; showResult("",action==="check"?"Checking …":"Submitting "+runs.length+" jobs …","");
  var step=function(i){
    if(i>=runs.length) return Promise.resolve();
    var f=Object.assign({},base,{run:runs[i]});
    return api("api/launch",{method:"POST",headers:{"Content-Type":"application/json","X-Requested-With":"bhaskera"},body:JSON.stringify(f)}).then(function(r){
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
  api("api/launch",{method:"POST",headers:{"Content-Type":"application/json","X-Requested-With":"bhaskera"},body:JSON.stringify({kind:"recipe",action:"show",base:base})}).then(function(x){
    if(x.j.error){ showResult("bad","Could not load it",x.j.error); return; }
    L.rc.yaml=x.j.output; L.rc.base=base; L.rc.copy=true; $("#startout").hidden=true; renderStart();
  });
}
function prepareData(){
  var r=L.rc, f={action:"start",kind:"tokenize",name:r.name,model:r.model,train_path:r.raw.train,val_path:r.raw.val,format:r.raw.format,seq_len:r.raw.seq,model_kind:r.kind};
  if(r.kind==="multimodal"){ f.modalities=r.mods; f.media_root=r.raw.media; }
  showResult("","Submitting the tokenising job …","");
  api("api/launch",{method:"POST",headers:{"Content-Type":"application/json","X-Requested-With":"bhaskera"},body:JSON.stringify(f)}).then(function(x){
    if(x.j.error){ showResult("bad","Not started",x.j.error); return; }
    if(x.j.ok){ showResult("ok","Tokenising job #"+x.j.job+" submitted. It takes a few minutes; when it has finished, reopen this form and the data is listed above.",x.j.output); refresh(); }
    else showResult("bad","Not submitted",x.j.output);
  }).catch(function(){ showResult("bad","Not started","The page could not reach the hub."); });
}
function openStart(kind,run){
  api("api/launch",{headers:{"X-Requested-With":"bhaskera"}}).then(function(r){ L.opts=r.j; }).catch(function(){ L.opts={enabled:false,reason:"The page could not reach the hub.",recipes:[],runs:[],partitions:[]}; }).then(function(){
    L.kind=kind||L.kind; if(run) L.run=run; renderStart(); if(!$("#startbox").open) $("#startbox").showModal();
  });
}
$("#startclose").addEventListener("click", function(){ $("#startbox").close(); });
$("#startkind").addEventListener("click", function(e){ var b=e.target.closest(".choice"); if(b){ L.kind=b.getAttribute("data-k"); L.hours=""; renderStart(); } });
$("#startfields").addEventListener("click", function(e){
  var b=e.target.closest("button"); if(!b) return;
  if(b.hasAttribute("data-n")){ L.nodes=parseInt(b.getAttribute("data-n"),10); renderStart(); }
  else if(b.hasAttribute("data-h")){ L.hours=b.getAttribute("data-h"); renderStart(); }
  else if(b.hasAttribute("data-t")){ var t=b.getAttribute("data-t"); L.tasks[t]=!L.tasks[t]; renderStart(); }
  else if(b.hasAttribute("data-q")){ L.quick=b.getAttribute("data-q")==="1"; renderStart(); }
  else if(b.hasAttribute("data-method")){ L.rc.method=b.getAttribute("data-method"); if(L.rc.tuning==="full"&&(L.rc.method==="qlora"||L.rc.method==="pipeline")){ L.rc.tuning="lora"; setTuningLr("lora"); } renderStart(); }
  else if(b.hasAttribute("data-tuning")){
    var tn=b.getAttribute("data-tuning"); L.rc.tuning=tn; setTuningLr(tn);
    if(tn==="full"){ L.rc.ckpt="on"; if(L.rc.method==="qlora"||L.rc.method==="pipeline") L.rc.method=L.rc.kind==="multimodal"?"fsdp":"fsdp"; }
    renderStart();
  }
  else if(b.hasAttribute("data-dmode")){ L.rc.dataMode=b.getAttribute("data-dmode"); renderStart(); }
  else if(b.hasAttribute("data-kind")){ L.rc.kind=b.getAttribute("data-kind"); renderStart(); }
  else if(b.hasAttribute("data-mod")){ var mm=b.getAttribute("data-mod"), ix=L.rc.mods.indexOf(mm); if(ix>=0){ if(L.rc.mods.length>1) L.rc.mods.splice(ix,1); } else L.rc.mods.push(mm); renderStart(); }
  else if(b.id==="f-rload"){ loadBase(); }
  else if(b.hasAttribute("data-steps")){ L.rc.steps=b.getAttribute("data-steps"); renderStart(); }
  else if(b.hasAttribute("data-lr")){ L.rc.lr=b.getAttribute("data-lr"); renderStart(); }
  else if(b.hasAttribute("data-rank")){ L.rc.rank=b.getAttribute("data-rank"); renderStart(); }
  else if(b.hasAttribute("data-ckpt")){ L.rc.ckpt=b.getAttribute("data-ckpt"); renderStart(); }
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
  else if(e.target.id==="f-note"){ L.note=e.target.value; }
  else if(e.target.id==="f-alias"){ L.alias=e.target.value.trim(); showSum(); }
  else if(e.target.id==="f-name"){ L.regname=e.target.value.trim(); L.regnameFor=L.run; }
  else if(e.target.id==="f-rname"){ L.rc.name=e.target.value.trim(); }
  else if(e.target.id==="f-rmodel"){ L.rc.model=e.target.value.trim(); }
  else if(e.target.id==="f-rmodelsel"){
    var mv=e.target.value, mm=(L.opts.model_catalog||[]).filter(function(x){return x.path===mv})[0];
    if(mv==="__other"){ L.rc.modelOther=true; L.rc.model=""; } else { L.rc.modelOther=false; L.rc.model=mv; if(mm) L.rc.kind=mm.kind; }
    renderStart();
  }
  else if(e.target.id==="f-rtrain"){ if(e.target.value==="__other"){ L.rc.trainOther=true; } else { L.rc.trainOther=false; L.rc.train=e.target.value; } renderStart(); }
  else if(e.target.id==="f-rval"){ L.rcValTouched=true; if(e.target.value==="__other"){ L.rc.valOther=true; } else { L.rc.valOther=false; L.rc.val=e.target.value; } renderStart(); }
  else if(e.target.id==="f-rtrain-path"){ L.rc.trainPath=e.target.value.trim(); }
  else if(e.target.id==="f-rval-path"){ L.rc.valPath=e.target.value.trim(); }
  else if(e.target.id==="f-rsteps"){ if(e.target.value.trim()) L.rc.steps=e.target.value.trim(); }
  else if(e.target.id==="f-rbatch"){ L.rc.batch=e.target.value.trim(); }
  else if(e.target.id==="f-rkeep"){ L.rc.keep=e.target.value.trim(); }
  else if(e.target.id==="f-rckdir"){ L.rc.ckdir=e.target.value; }
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

refresh(); loadModels(); setInterval(loadModels, 30000); setInterval(refresh, 8000); setInterval(clock, 1000);
})();
</script>
</body>
</html>
"""
