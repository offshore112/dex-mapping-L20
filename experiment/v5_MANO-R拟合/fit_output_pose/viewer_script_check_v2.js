
const data=JSON.parse(document.getElementById('scene-data').textContent),$=id=>document.getElementById(id),canvas=$('canvas'),ctx=canvas.getContext('2d');
const allPts=[...data.mano.vertices,...data.target_points,...data.fitted_points,...data.l20_triangles.flat(2)];
const lo=allPts.reduce((a,p)=>a.map((v,i)=>Math.min(v,p[i])),[Infinity,Infinity,Infinity]),hi=allPts.reduce((a,p)=>a.map((v,i)=>Math.max(v,p[i])),[-Infinity,-Infinity,-Infinity]);
const center=lo.map((v,i)=>(v+hi[i])/2),span=Math.max(...hi.map((v,i)=>v-lo[i]));let yaw=-.72,pitch=.28,zoom=1,drag=null,pending=false;
function project(p){let a=p.map((x,k)=>x-center[k]),right=[-Math.sin(yaw),Math.cos(yaw),0],up=[-Math.sin(pitch)*Math.cos(yaw),-Math.sin(pitch)*Math.sin(yaw),Math.cos(pitch)],dep=[Math.cos(pitch)*Math.cos(yaw),Math.cos(pitch)*Math.sin(yaw),Math.sin(pitch)],dot=v=>v.reduce((s,x,k)=>s+x*a[k],0),s=Math.min(canvas.clientWidth,canvas.clientHeight)*zoom/span*.78;return [canvas.clientWidth*.52+dot(right)*s,canvas.clientHeight*.52-dot(up)*s,dot(dep)];}
function depth(tri){return tri.map(project).reduce((s,p)=>s+p[2],0)/3;}function poly(tri,fill,alpha){let q=tri.map(project);ctx.globalAlpha=alpha;ctx.beginPath();ctx.moveTo(q[0][0],q[0][1]);q.slice(1).forEach(p=>ctx.lineTo(p[0],p[1]));ctx.closePath();ctx.fillStyle=fill;ctx.fill();ctx.globalAlpha=1;}
function line(a,b,color,w=1,dash=[]){let p=project(a),q=project(b);ctx.beginPath();ctx.setLineDash(dash);ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.lineWidth=w;ctx.strokeStyle=color;ctx.stroke();ctx.setLineDash([]);}
function dot(p,color,r=4){let q=project(p);ctx.beginPath();ctx.arc(q[0],q[1],r,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();}
const chains=[[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]];
function draw(){pending=false;let dpr=devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr)}ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
 if($('show-l20').checked){let ps=data.l20_triangles.map(t=>({t,z:depth(t)}));ps.sort((a,b)=>a.z-b.z);ps.forEach(x=>poly(x.t,'#7b8b99',.12));}
 if($('show-mano').checked){let ps=data.mano.faces.map(f=>{let t=f.map(i=>data.mano.vertices[i]);return {t,z:depth(t)}});ps.sort((a,b)=>a.z-b.z);ps.forEach(x=>poly(x.t,'#2b7da8',.29));}
 if($('show-lines').checked){for(let i=0;i<21;i++)line(data.target_points[i],data.fitted_points[i],'#b84a4a',1,[3,3]);}
 if($('show-target').checked){chains.forEach(c=>{for(let i=1;i<c.length;i++)line(data.target_points[c[i-1]],data.target_points[c[i]],'#d23c3c',2)});data.target_points.forEach(p=>dot(p,'#d23c3c',4));}
 if($('show-fitted').checked){chains.forEach(c=>{for(let i=1;i<c.length;i++)line(data.fitted_points[c[i-1]],data.fitted_points[c[i]],'#2b9b62',1.5,[4,3])});data.fitted_points.forEach(p=>dot(p,'#2b9b62',3.5));}
 let n=data.errors_mm.reduce((s,x)=>s+x*x,0)**.5/Math.sqrt(data.errors_mm.length),mx=Math.max(...data.errors_mm);$('readout').textContent=`拟合 RMSE = ${n.toFixed(2)} mm；最大点误差 = ${mx.toFixed(2)} mm。拖动可观察 MANO-R 表面与 L20 参考点的空间关系。`;}
for(let id of ['show-mano','show-l20','show-target','show-fitted','show-lines'])$(id).onchange=()=>requestDraw();function requestDraw(){if(!pending){pending=true;requestAnimationFrame(draw)}}
$('front').onclick=()=>{yaw=-.72;pitch=.28;zoom=1;requestDraw()};$('side').onclick=()=>{yaw=Math.PI/2;pitch=.15;zoom=1;requestDraw()};$('reset').onclick=()=>{yaw=-.72;pitch=.28;zoom=1;requestDraw()};
canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};canvas.onpointerup=()=>drag=null;canvas.onpointermove=e=>{if(!drag)return;yaw-=(e.clientX-drag[0])*.008;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-drag[1])*.006));drag=[e.clientX,e.clientY];requestDraw()};canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(.5,Math.min(3,zoom*Math.exp(-e.deltaY*.001)));requestDraw()};new ResizeObserver(requestDraw).observe(canvas);
$('summary').textContent=`21 点对应 · β + 尺度 + θ_ref · scale=${data.scale.toFixed(4)} · 无硬件标定`;$('metrics').textContent=`点误差单位：mm；β 已保存为 10 维形状参数。`;requestDraw();
window.addEventListener('error',e=>{let n=$('error');if(n)n.textContent='Canvas 交互绘制错误：'+(e.message||'unknown error')+'；请使用下面的静态备用预览。'});

