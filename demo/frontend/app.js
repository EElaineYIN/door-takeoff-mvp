// ---------- state ----------
const S = {
  job:null, pages:[], cur:-1, isPdf:false, schedPage:null, schedule:[],
  boxes:{},            // page -> [{x1,y1,x2,y2,score,type,direction,note,origin,edited,id}]
  scale:1, zoom:1, sel:null, adding:false, imgW:0, imgH:0,
  panX:0, panY:0,                       // canvas-wrap offset inside the viewport (px)
  region:null, selectingRegion:false,   // detect only inside this [x1,y1,x2,y2] (image px)
  audit:{ai:0, added:0, removed:0, edited:0},   // cumulative across session
};
const TYPES=["swing","double","sliding","pocket","other"];
const TYPE_COLOR={swing:"#2D8291",double:"#7b5ea7",sliding:"#d98a2b",pocket:"#c0504d",other:"#8a9ba0",unclassified:"#b7c4c6"};
let nextId=1;

const $=(id)=>document.getElementById(id);
const api=async(p,o)=>{const r=await fetch(p,o);if(!r.ok)throw new Error((await r.text())||r.status);return r;};
function setStatus(m,busy=false){const e=$("statusbar");e.textContent=m||"";e.classList.toggle("busy",busy);}
function setStep(name){
  const order=["upload","schedule","takeoff","review","report"];
  const idx=order.indexOf(name);
  document.querySelectorAll(".pipeline .step").forEach(s=>{
    const i=order.indexOf(s.dataset.step);
    s.classList.toggle("active",s.dataset.step===name);
    s.classList.toggle("done",i<idx);
  });
}
const disp=()=>S.scale*S.zoom;

// ---------- upload ----------
const dz=$("dropzone"),fileInput=$("fileInput");
// NOTE: the dropzone is a <label> wrapping the file input, so clicking it
// already opens the picker natively. Do NOT also call fileInput.click() here
// or the dialog fires twice (looks like you must upload the file twice).
fileInput.onchange=e=>{if(e.target.files[0])uploadFile(e.target.files[0]);e.target.value="";};
["dragover","dragenter"].forEach(ev=>dz.addEventListener(ev,e=>{e.preventDefault();dz.classList.add("drag");}));
["dragleave","drop"].forEach(ev=>dz.addEventListener(ev,e=>{e.preventDefault();dz.classList.remove("drag");}));
dz.addEventListener("drop",e=>e.dataTransfer.files[0]&&uploadFile(e.dataTransfer.files[0]));

async function uploadFile(file){
  setStatus(`Uploading ${file.name}…`,true); setStep("upload");
  const fd=new FormData(); fd.append("file",file);
  try{
    const data=await(await api("/api/upload",{method:"POST",body:fd})).json();
    S.job=data.job_id; S.pages=data.pages; S.isPdf=data.is_pdf; S.boxes={}; S.schedule=[]; S.schedPage=null;
    S.audit={ai:0,added:0,removed:0,edited:0};
    $("toolbar").hidden=false; $("exportPanel").hidden=false; $("dashBtn").hidden=false;
    renderPageList();
    $("pagesPanel").hidden=false;
    if(S.isPdf && S.pages.length>1){ await scanSchedule(); }
    else { setStatus(`Loaded image. Detect doors when ready.`); selectPage(0); }
  }catch(err){setStatus("Upload failed: "+err.message);}
}

// ---------- schedule ----------
async function scanSchedule(){
  setStatus("AI is reading the door schedule…",true); setStep("schedule");
  try{
    const data=await(await api("/api/scan_schedule",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({job_id:S.job})})).json();
    if(data.found){
      S.schedPage=data.schedule_page; S.schedule=data.doors||[];
      renderSchedule(data);
      renderPageList();
      setStatus(`AI read the schedule on ${data.schedule_page_name}: ${S.schedule.length} door type(s). Now pick a plan sheet to take off.`);
    }else{
      setStatus("No door schedule found in this PDF — pick any sheet to take off.");
    }
    setStep("takeoff");
  }catch(err){setStatus("Schedule scan failed: "+err.message);setStep("takeoff");}
}
function renderSchedule(data){
  $("schedulePanel").hidden=false;
  $("schedPill").textContent=data.schedule_page_name||"";
  $("schedNote").textContent=data.note||"";
  const t=$("schedTable"); t.innerHTML="";
  if(!S.schedule.length){t.innerHTML='<p class="muted-sm">No rows parsed.</p>';return;}
  S.schedule.forEach(d=>{
    const row=document.createElement("div"); row.className="sched-row";
    const sz=[d.width,d.height].filter(Boolean).join(" × ");
    row.innerHTML=`<span class="mark">${d.mark||"—"}</span>
      <span><span class="ty" style="background:${TYPE_COLOR[d.type]||TYPE_COLOR.other}"></span>${d.type||"—"}</span>
      <span class="sz">${sz||""}</span>`;
    t.appendChild(row);
  });
}

// ---------- page list ----------
function renderPageList(){
  const ul=$("pagelist"); ul.innerHTML="";
  S.pages.forEach(p=>{
    const li=document.createElement("li");
    const isSched=p.index===S.schedPage;
    li.className=isSched?"sched":"";
    li.innerHTML=`<span>${p.name}${isSched?" · schedule":""}</span><span class="badge" id="pb-${p.index}"></span>`;
    li.dataset.idx=p.index; li.onclick=()=>selectPage(p.index);
    ul.appendChild(li);
  });
  refreshBadges();
}
function refreshBadges(){S.pages.forEach(p=>{const b=$("pb-"+p.index);if(b)b.textContent=S.boxes[p.index]?S.boxes[p.index].length:"";});}

// ---------- select page ----------
function selectPage(idx){
  S.cur=idx; S.sel=null; setSelPanel(null);
  S.region=null; S.selectingRegion=false; $("clearRegionBtn").hidden=true;
  $("regionBtn").classList.remove("on"); $("overlay").classList.remove("regioning");
  document.querySelectorAll(".pagelist li").forEach(li=>li.classList.toggle("active",+li.dataset.idx===idx));
  $("emptyState").hidden=true; $("canvasWrap").hidden=false;
  const img=$("planImg");
  img.onload=()=>{S.imgW=img.naturalWidth;S.imgH=img.naturalHeight;fitScale();layout();updateCounts();
    $("classifyBtn").disabled=!(S.boxes[idx]&&S.boxes[idx].length);
    $("reconcileBtn").disabled=!(S.boxes[idx]&&S.boxes[idx].length&&S.schedule.length);};
  img.src=`/api/image/${S.job}/${idx}`;
}
function fitScale(){
  const vp=$("viewport");
  S.scale=Math.min(1,(vp.clientWidth-48)/S.imgW,(vp.clientHeight-48)/S.imgH);
  S.zoom=1;$("zoomVal").textContent=Math.round(S.scale*100)+"%";
  centerView();
}
function centerView(){
  const vp=$("viewport"),d=disp();
  S.panX=(vp.clientWidth-S.imgW*d)/2;
  S.panY=(vp.clientHeight-S.imgH*d)/2;
}
function applyPan(){const w=$("canvasWrap");w.style.left=S.panX+"px";w.style.top=S.panY+"px";}

// ---------- layout / boxes ----------
function layout(){
  const img=$("planImg"),wrap=$("canvasWrap"),ov=$("overlay"),d=disp();
  const w=S.imgW*d,h=S.imgH*d;
  img.style.width=w+"px";img.style.height=h+"px";wrap.style.width=w+"px";wrap.style.height=h+"px";
  applyPan();
  ov.innerHTML="";
  if(S.region){
    const r=S.region,el=document.createElement("div");el.className="region";
    el.style.left=Math.min(r.x1,r.x2)*d+"px";el.style.top=Math.min(r.y1,r.y2)*d+"px";
    el.style.width=Math.abs(r.x2-r.x1)*d+"px";el.style.height=Math.abs(r.y2-r.y1)*d+"px";
    ov.appendChild(el);
  }
  (S.boxes[S.cur]||[]).forEach(b=>ov.appendChild(makeBox(b)));
  setPanCursor();
}
function makeBox(b){
  const d=disp(),el=document.createElement("div");
  el.className="box"+(S.sel===b?" sel":"")+(b.origin==="human"?" human":"");
  el.dataset.type=b.type||"swing";
  el.style.left=b.x1*d+"px";el.style.top=b.y1*d+"px";
  el.style.width=(b.x2-b.x1)*d+"px";el.style.height=(b.y2-b.y1)*d+"px";
  const tag=document.createElement("span");tag.className="tag";
  tag.textContent=(b.type||"door")+(b.direction&&b.direction!=="n/a"?"·"+b.direction:"");
  el.appendChild(tag);
  const h=document.createElement("div");h.className="handle";el.appendChild(h);
  el.addEventListener("mousedown",e=>startDrag(e,b,el,false));
  h.addEventListener("mousedown",e=>startDrag(e,b,el,true));
  el.addEventListener("mouseenter",e=>showTip(e,b));
  el.addEventListener("mousemove",e=>moveTip(e));
  el.addEventListener("mouseleave",hideTip);
  return el;
}
function placeBox(el,b){const d=disp();el.style.left=b.x1*d+"px";el.style.top=b.y1*d+"px";el.style.width=(b.x2-b.x1)*d+"px";el.style.height=(b.y2-b.y1)*d+"px";}

// ---------- tooltip (feature D) ----------
function showTip(e,b){
  const tt=$("tooltip");
  const score=b.score!=null?` · conf ${(b.score*100|0)}%`:"";
  tt.innerHTML=`<span class="tt-type">${b.type||"door"}</span>${b.direction&&b.direction!=="n/a"?` · opens ${b.direction}`:""}<span class="tt-score">${score}</span>`+
    (b.note?`<br>🧠 ${b.note}`:"")+(b.origin==="human"?"<br>✎ added by you":"");
  tt.hidden=false; moveTip(e);
}
function moveTip(e){const tt=$("tooltip");tt.style.left=(e.clientX+14)+"px";tt.style.top=(e.clientY+14)+"px";}
function hideTip(){$("tooltip").hidden=true;}

// ---------- drag / resize ----------
function startDrag(e,b,el,resize){
  if(S.adding)return; e.stopPropagation();e.preventDefault(); selectBox(b); hideTip();
  const d=disp(),sx=e.clientX,sy=e.clientY,o={...b};
  function move(ev){
    const dx=(ev.clientX-sx)/d,dy=(ev.clientY-sy)/d;
    if(resize){b.x2=Math.max(o.x1+4,o.x2+dx);b.y2=Math.max(o.y1+4,o.y2+dy);}
    else{const w=o.x2-o.x1,h=o.y2-o.y1;b.x1=Math.max(0,Math.min(S.imgW-w,o.x1+dx));b.y1=Math.max(0,Math.min(S.imgH-h,o.y1+dy));b.x2=b.x1+w;b.y2=b.y1+h;}
    placeBox(el,b);
  }
  function up(){document.removeEventListener("mousemove",move);document.removeEventListener("mouseup",up);}
  document.addEventListener("mousemove",move);document.addEventListener("mouseup",up);
}

// ---------- add mode ----------
$("addBtn").onclick=()=>{S.adding=!S.adding;if(S.adding){S.selectingRegion=false;$("regionBtn").classList.remove("on");$("overlay").classList.remove("regioning");}$("addBtn").classList.toggle("on",S.adding);$("overlay").classList.toggle("adding",S.adding);$("modeHint").textContent=S.adding?"Draw a box on the plan":"";setPanCursor();};
$("overlay").addEventListener("mousedown",e=>{
  if(!S.adding||e.target!==$("overlay"))return;
  const rect=$("overlay").getBoundingClientRect(),d=disp();
  const sx=(e.clientX-rect.left)/d,sy=(e.clientY-rect.top)/d;
  const b={x1:sx,y1:sy,x2:sx,y2:sy,type:"swing",direction:"n/a",note:"added by reviewer",score:null,origin:"human",id:nextId++};
  function move(ev){b.x2=(ev.clientX-rect.left)/d;b.y2=(ev.clientY-rect.top)/d;layout();}
  function up(){
    document.removeEventListener("mousemove",move);document.removeEventListener("mouseup",up);
    if(Math.abs(b.x2-b.x1)<4||Math.abs(b.y2-b.y1)<4){layout();return;}
    if(b.x2<b.x1)[b.x1,b.x2]=[b.x2,b.x1]; if(b.y2<b.y1)[b.y1,b.y2]=[b.y2,b.y1];
    (S.boxes[S.cur]=S.boxes[S.cur]||[]).push(b); S.audit.added++;
    // stay in add mode so several missed doors can be drawn in a row
    layout();selectBox(b);updateCounts();refreshBadges();
    $("modeHint").textContent="Added — draw another, or press Esc / click ＋ Add to finish";
  }
  document.addEventListener("mousemove",move);document.addEventListener("mouseup",up);
});

// ---------- region select (limit detection to one drawing) ----------
$("regionBtn").onclick=()=>{
  S.selectingRegion=!S.selectingRegion;
  if(S.selectingRegion){S.adding=false;$("addBtn").classList.remove("on");$("overlay").classList.remove("adding");}
  $("regionBtn").classList.toggle("on",S.selectingRegion);
  $("overlay").classList.toggle("regioning",S.selectingRegion);
  $("modeHint").textContent=S.selectingRegion?"Drag a box around ONE drawing":"";
  setPanCursor();
};
$("clearRegionBtn").onclick=()=>{S.region=null;$("clearRegionBtn").hidden=true;layout();setStatus("Area cleared — Detect will scan the whole sheet.");};
$("overlay").addEventListener("mousedown",e=>{
  if(!S.selectingRegion||e.target!==$("overlay"))return;
  e.preventDefault();
  const rect=$("overlay").getBoundingClientRect(),d=disp();
  const sx=(e.clientX-rect.left)/d,sy=(e.clientY-rect.top)/d;
  const r={x1:sx,y1:sy,x2:sx,y2:sy}; S.region=r;
  function move(ev){r.x2=(ev.clientX-rect.left)/d;r.y2=(ev.clientY-rect.top)/d;layout();}
  function up(){
    document.removeEventListener("mousemove",move);document.removeEventListener("mouseup",up);
    S.selectingRegion=false;$("regionBtn").classList.remove("on");$("overlay").classList.remove("regioning");$("modeHint").textContent="";
    if(Math.abs(r.x2-r.x1)<8||Math.abs(r.y2-r.y1)<8){S.region=null;layout();return;}
    $("clearRegionBtn").hidden=false;layout();
    setStatus("Area set. Click ‘① Detect doors’ to scan just this drawing.");
  }
  document.addEventListener("mousemove",move);document.addEventListener("mouseup",up);
});

// ---------- pan (drag the sheet around when zoomed in) ----------
(function(){
  const vp=$("viewport");
  vp.addEventListener("mousedown",e=>{
    if(S.adding||S.selectingRegion)return;        // those modes draw boxes instead
    if(e.button!==0||e.target.closest(".box"))return;  // let box drag work
    if(S.cur<0)return;
    e.preventDefault();
    const sx=e.clientX,sy=e.clientY,px=S.panX,py=S.panY;
    vp.classList.add("panning");
    function move(ev){S.panX=px+(ev.clientX-sx);S.panY=py+(ev.clientY-sy);applyPan();}
    function up(){vp.classList.remove("panning");document.removeEventListener("mousemove",move);document.removeEventListener("mouseup",up);}
    document.addEventListener("mousemove",move);document.addEventListener("mouseup",up);
  });
  // wheel to zoom toward the cursor
  vp.addEventListener("wheel",e=>{
    if(S.cur<0)return; e.preventDefault();
    zoomAt(e.deltaY<0?1.15:1/1.15,e.clientX,e.clientY);
  },{passive:false});
})();
function setPanCursor(){const vp=$("viewport");vp.classList.toggle("pannable",S.cur>=0&&!S.adding&&!S.selectingRegion);}

// ---------- selection ----------
$("overlay").addEventListener("mousedown",e=>{if(e.target===$("overlay")&&!S.adding&&!S.selectingRegion)selectBox(null);});
function selectBox(b){S.sel=b;document.querySelectorAll(".box").forEach((el,i)=>el.classList.toggle("sel",(S.boxes[S.cur]||[])[i]===b));setSelPanel(b);}
function setSelPanel(b){const p=$("selPanel");if(!b){p.hidden=true;return;}p.hidden=false;$("selType").value=b.type||"swing";$("selDir").value=b.direction||"n/a";$("selNote").textContent=b.note?`🧠 Claude: ${b.note}`:"";}
$("selType").onchange=e=>{if(S.sel){if(S.sel.origin==="ai"&&!S.sel.edited){S.sel.edited=true;S.audit.edited++;}S.sel.type=e.target.value;layout();reselect();updateCounts();}};
$("selDir").onchange=e=>{if(S.sel){S.sel.direction=e.target.value;layout();reselect();}};
function reselect(){const b=S.sel;document.querySelectorAll(".box").forEach((el,i)=>el.classList.toggle("sel",(S.boxes[S.cur]||[])[i]===b));}
$("deleteBtn").onclick=deleteSel;
function deleteSel(){if(!S.sel)return;if(S.sel.origin==="ai")S.audit.removed++;S.boxes[S.cur]=(S.boxes[S.cur]||[]).filter(x=>x!==S.sel);S.sel=null;setSelPanel(null);layout();updateCounts();refreshBadges();}
document.addEventListener("keydown",e=>{
  if((e.key==="Delete"||e.key==="Backspace")&&S.sel&&document.activeElement.tagName!=="SELECT"){e.preventDefault();deleteSel();return;}
  if(e.key==="Escape"){
    if(S.adding){S.adding=false;$("addBtn").classList.remove("on");$("overlay").classList.remove("adding");}
    if(S.selectingRegion){S.selectingRegion=false;$("regionBtn").classList.remove("on");$("overlay").classList.remove("regioning");}
    $("modeHint").textContent="";setPanCursor();
  }
});

// ---------- scan animation ----------
function scan(on){$("scanline").hidden=!on;}

// ---------- detect (YOLO) ----------
$("detectBtn").onclick=async()=>{
  if(S.cur<0)return; setStatus("YOLO is scanning the sheet for doors…",true);setStep("takeoff");scan(true);
  try{
    const region=S.region?[S.region.x1,S.region.y1,S.region.x2,S.region.y2]:null;
    const data=await(await api("/api/detect",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({job_id:S.job,page:S.cur,conf:parseFloat($("confSlider").value),region})})).json();
    S.boxes[S.cur]=data.boxes.map(b=>({...b,type:"unclassified",direction:"n/a",note:"",origin:"ai",edited:false,id:nextId++}));
    S.audit.ai+=data.count;
    layout();updateCounts();refreshBadges();
    $("classifyBtn").disabled=data.count===0;
    $("reconcileBtn").disabled=!(data.count&&S.schedule.length);
    setStatus(`YOLO located ${data.count} door(s) — types not read yet. Click ② Classify to identify each type.`);setStep("review");
  }catch(err){setStatus("Detect failed: "+err.message);}finally{scan(false);}
};

// ---------- classify (Claude) ----------
$("classifyBtn").onclick=async()=>{
  const boxes=S.boxes[S.cur]||[]; if(!boxes.length)return;
  setStatus(`Claude is reading ${boxes.length} door symbol(s)…`,true);setStep("review");scan(true);
  try{
    const{results}=await(await api("/api/classify",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({job_id:S.job,page:S.cur,boxes:boxes.map(({x1,y1,x2,y2})=>({x1,y1,x2,y2}))})})).json();
    results.forEach((r,i)=>{if(!boxes[i])return;boxes[i].type=r.type||"other";boxes[i].direction=r.direction||"n/a";boxes[i].note=r.note||"";boxes[i].is_door=r.is_door;});
    layout();updateCounts();
    setStatus("Claude classified the door types. Adjust any in the inspector, then reconcile or export.");
  }catch(err){setStatus("Classify failed: "+err.message);}finally{scan(false);}
};

// ---------- counts ----------
function updateCounts(){
  const boxes=S.boxes[S.cur]||[];$("countNum").textContent=boxes.length;
  const c={swing:0,double:0,sliding:0,pocket:0,other:0,unclassified:0};
  boxes.forEach(b=>c[b.type]=(c[b.type]||0)+1);
  TYPES.forEach(t=>$("b-"+t).textContent=c[t]||0);
  $("b-unclassified").textContent=c.unclassified||0;
  // only show the Unclassified row while some doors still await classification
  $("row-unclassified").style.display=c.unclassified?"flex":"none";
  renderAudit();
}

// ---------- tabs ----------
document.querySelectorAll(".tab").forEach(t=>t.onclick=()=>{
  document.querySelectorAll(".tab").forEach(x=>x.classList.toggle("active",x===t));
  ["types","reconcile","audit"].forEach(name=>$("pane-"+name).hidden=name!==t.dataset.tab);
  if(t.dataset.tab==="audit")renderAudit();
});

// ---------- reconcile (feature A) ----------
$("reconcileBtn").onclick=async()=>{
  const boxes=S.boxes[S.cur]||[]; if(!boxes.length||!S.schedule.length)return;
  setStatus("Reconciling detected doors against the schedule…",true);
  try{
    const data=await(await api("/api/reconcile",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({job_id:S.job,detected:boxes.map(b=>({type:b.type}))})})).json();
    renderRecon(data); setStatus(`Reconciled: ${data.detected_total} detected vs ${data.scheduled_total} scheduled.`);
  }catch(err){setStatus("Reconcile failed: "+err.message);}
};
function renderRecon(data){
  const el=$("reconResult");
  let html=`<div class="recon-head"><span>type</span><span>sched</span><span>found</span><span>status</span></div>`;
  data.rows.forEach(r=>{html+=`<div class="recon-row"><span><span class="ty dot ${r.type}"></span>${r.type}</span><b>${r.scheduled}</b><b>${r.detected}</b><span class="st ${r.status}">${r.status}</span></div>`;});
  el.innerHTML=html;
}

// ---------- audit (feature B) ----------
function renderAudit(){
  const a=S.audit, final=(S.boxes[S.cur]||[]).length;
  const auto=a.ai>0?Math.round((a.ai-a.removed)/a.ai*100):0;
  const minSaved=Math.round(final*25/60);  // assume 25s manual vs review per door
  $("auditBox").innerHTML=`
    <div class="metric big"><span>AI automation rate<br><span class="sub">AI doors kept ÷ AI doors found</span></span><b>${auto}%</b></div>
    <div class="metric"><span>Detected by AI</span><b>${a.ai}</b></div>
    <div class="metric"><span>Added by reviewer</span><b>${a.added}</b></div>
    <div class="metric"><span>Removed by reviewer</span><b>${a.removed}</b></div>
    <div class="metric"><span>Type edits</span><b>${a.edited}</b></div>
    <div class="metric big"><span>Est. time saved<br><span class="sub">vs ~25s/door manual</span></span><b>~${minSaved} min</b></div>`;
}

// ---------- dashboard (feature C) ----------
$("dashBtn").onclick=openDash; $("dashClose").onclick=()=>$("dashModal").hidden=true;
$("dashModal").addEventListener("click",e=>{if(e.target===$("dashModal"))$("dashModal").hidden=true;});
function openDash(){
  setStep("report");
  const all=[]; S.pages.forEach(p=>(S.boxes[p.index]||[]).forEach(b=>all.push(b)));
  const total=all.length;
  const c={swing:0,double:0,sliding:0,pocket:0,other:0}; all.forEach(b=>c[b.type]=(c[b.type]||0)+1);
  // donut
  let acc=0,seg=[];TYPES.forEach(t=>{if(c[t]){const frac=c[t]/total*360;seg.push(`${TYPE_COLOR[t]} ${acc}deg ${acc+frac}deg`);acc+=frac;}});
  const donut=total?`conic-gradient(${seg.join(",")})`:"#e6eeef";
  const legend=TYPES.filter(t=>c[t]).map(t=>`<span><span class="dot ${t}"></span>${t}<b>${c[t]}</b></span>`).join("");
  // per-sheet bars
  const sheetCounts=S.pages.map(p=>({name:p.name,n:(S.boxes[p.index]||[]).length})).filter(x=>x.n>0||x.name);
  const maxN=Math.max(1,...sheetCounts.map(x=>x.n));
  const bars=sheetCounts.map(x=>`<div class="bar-row"><div class="lab"><span>${x.name}</span><b>${x.n}</b></div><div class="bar-track"><div class="bar-fill" style="width:${x.n/maxN*100}%"></div></div></div>`).join("");
  const a=S.audit,auto=a.ai>0?Math.round((a.ai-a.removed)/a.ai*100):0,minSaved=Math.round(total*25/60);
  $("dashGrid").innerHTML=`
    <div class="kpi-row">
      <div class="kpi"><div class="v">${total}</div><div class="l">Total doors</div></div>
      <div class="kpi alt"><div class="v">${S.schedule.length||"—"}</div><div class="l">Schedule types</div></div>
      <div class="kpi alt"><div class="v">${auto}%</div><div class="l">AI automation rate</div></div>
      <div class="kpi"><div class="v">~${minSaved}m</div><div class="l">Est. time saved</div></div>
    </div>
    <div class="dash-card"><h3>Door types</h3><div class="donut-wrap"><div class="donut" data-total="${total}" style="background:${donut}"></div><div class="legend">${legend||'<span class="muted-sm">No doors yet</span>'}</div></div></div>
    <div class="dash-card"><h3>Doors per sheet</h3><div class="bars">${bars||'<span class="muted-sm">No sheets analysed</span>'}</div></div>`;
  $("dashModal").hidden=false;
}

// ---------- zoom / conf ----------
$("zoomIn").onclick=()=>zoomAtCenter(1.4);$("zoomOut").onclick=()=>zoomAtCenter(1/1.4);
$("zoomFit").onclick=()=>{fitScale();layout();};
function clampZoom(z){
  // allow zooming up to ~250% of native pixels regardless of fit-scale.
  const maxZoom=S.scale>0?2.5/S.scale:8, minZoom=0.5;
  return Math.max(minZoom,Math.min(maxZoom,z));
}
function zoomAt(factor,clientX,clientY){
  const vp=$("viewport"),r=vp.getBoundingClientRect();
  const cx=clientX-r.left,cy=clientY-r.top;        // cursor in viewport coords
  const oldD=disp(),newZoom=clampZoom(S.zoom*factor);
  if(newZoom===S.zoom)return;
  S.zoom=newZoom; const newD=disp();
  // keep the image point under the cursor fixed
  S.panX=cx-(cx-S.panX)*(newD/oldD);
  S.panY=cy-(cy-S.panY)*(newD/oldD);
  $("zoomVal").textContent=Math.round(newD*100)+"%";layout();
}
function zoomAtCenter(factor){const vp=$("viewport"),r=vp.getBoundingClientRect();zoomAt(factor,r.left+vp.clientWidth/2,r.top+vp.clientHeight/2);}
$("confSlider").oninput=e=>$("confVal").textContent=e.target.value;

// ---------- export ----------
$("exportBtn").onclick=async()=>{
  const rows=S.pages.filter(p=>(S.boxes[p.index]||[]).length)
    .map(p=>{const boxes=S.boxes[p.index]||[];const c={swing:0,double:0,sliding:0,pocket:0,other:0};boxes.forEach(b=>c[b.type]=(c[b.type]||0)+1);return{sheet:p.name,door_count:boxes.length,...c};});
  if(!rows.length){setStatus("Nothing to export yet — detect doors on at least one sheet first.");return;}
  const blob=await(await api("/api/export",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({rows})})).blob();
  const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="takeoff.csv";a.click();
  setStatus("Exported takeoff.csv");setStep("report");
};

// ---------- export human-reviewed boxes as YOLO training labels (active learning) ----------
$("exportLabelsBtn").onclick=async()=>{
  const pages=S.pages.filter(p=>(S.boxes[p.index]||[]).length).map(p=>({
    page:p.index,
    boxes:(S.boxes[p.index]||[]).map(b=>({x1:b.x1,y1:b.y1,x2:b.x2,y2:b.y2,type:b.type})),
  }));
  if(!pages.length){setStatus("No reviewed doors yet — detect/review on a sheet first.");return;}
  const total=pages.reduce((n,p)=>n+p.boxes.length,0);
  setStatus(`Bundling ${total} reviewed door box(es) as YOLO labels…`,true);
  try{
    const blob=await(await api("/api/export_labels",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({job_id:S.job,pages})})).blob();
    const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="review_labels.zip";a.click();
    setStatus(`Exported review_labels.zip (${total} boxes) — add to the training set and re-fine-tune.`);setStep("report");
  }catch(err){setStatus("Label export failed: "+err.message);}
};

window.addEventListener("resize",()=>{if(S.cur>=0){fitScale();layout();}});
setStep("upload");
