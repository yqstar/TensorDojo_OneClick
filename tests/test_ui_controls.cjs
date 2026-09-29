// Run with node tests/test_ui_controls.cjs. Component behavior tests using a minimal DOM fixture.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
class Evt{constructor(type,props={}){this.type=type;Object.assign(this,props);this.defaultPrevented=false;}preventDefault(){this.defaultPrevented=true;}stopPropagation(){this.stopped=true;}}
class Events{constructor(){this.listeners={};}addEventListener(t,f){(this.listeners[t]??=[]).push(f);}removeEventListener(t,f){this.listeners[t]=(this.listeners[t]||[]).filter(x=>x!==f);}dispatchEvent(e){e.target??=this;for(const f of this.listeners[e.type]||[])f(e);return !e.defaultPrevented;}}
let doc;
class El extends Events{
 constructor(tag){super();this.tagName=tag.toUpperCase();this.attrs={};this.children=[];this.parentElement=null;this.dataset={};this.style={setProperty(k,v){this[k]=v;}};this.className='';this.classList={add:(c)=>this.className=[...new Set([...this.className.split(' ').filter(Boolean),c])].join(' '),remove:(c)=>this.className=this.className.split(' ').filter(x=>x!==c).join(' '),toggle:(c,on)=>on?this.classList.add(c):this.classList.remove(c),contains:c=>this.className.split(' ').includes(c)};this._text='';this._disabled=false;this._selected=0;this.labels=[];this.scrollTop=0;this.rect={left:40,top:100,width:160,height:42,right:200,bottom:142};}
 set textContent(v){this._text=String(v);this.children=[];}get textContent(){return this._text+this.children.map(x=>x.textContent).join('');}
 set id(v){this.attrs.id=v;}get id(){return this.attrs.id||'';}set hidden(v){if(v)this.attrs.hidden='';else delete this.attrs.hidden;}get hidden(){return 'hidden'in this.attrs;}
 set tabIndex(v){this.attrs.tabindex=String(v);}get tabIndex(){return Number(this.attrs.tabindex??0);}set disabled(v){this._disabled=v;}get disabled(){return this._disabled;}get options(){return this.tagName==='SELECT'?this.children:undefined;}
 get selectedIndex(){return this._selected;}set selectedIndex(v){this._selected=v;}get value(){return this.tagName==='SELECT'?this.options[this.selectedIndex]?.value:this._value||'';}set value(v){if(this.tagName==='SELECT')this.selectedIndex=this.options.findIndex(o=>o.value===v);else this._value=v;}
 get label(){return this.attrs.label||this.textContent;}get isConnected(){return this===doc.documentElement||!!this.parentElement?.isConnected;}
 setAttribute(k,v){this.attrs[k]=String(v);}getAttribute(k){return this.attrs[k]??null;}removeAttribute(k){delete this.attrs[k];}hasAttribute(k){return k in this.attrs;}
 append(...nodes){for(const n of nodes){n.parentElement=this;this.children.push(n);}}replaceChildren(...nodes){for(const n of this.children)n.parentElement=null;this.children=[];this.append(...nodes);}
 insertAdjacentElement(_,el){el.parentElement=this.parentElement;this.parentElement.children.splice(this.parentElement.children.indexOf(this)+1,0,el);}remove(){if(this.parentElement)this.parentElement.children=this.parentElement.children.filter(x=>x!==this);this.parentElement=null;}
 contains(n){return this===n||this.children.some(c=>c.contains(n));}closest(q){if(q.split(',').some(x=>x.trim()[0]==='.'?this.classList.contains(x.trim().slice(1)):this.tagName===x.trim().toUpperCase()))return this;return this.parentElement?.closest(q)||null;}
 matches(q){return q===':disabled'?this.disabled:false;}getBoundingClientRect(){return this.rect;}focus(){doc.activeElement=this;doc.dispatchEvent(new Evt('focusin',{target:this}));}
 get offsetTop(){return this.parentElement.children.indexOf(this)*38+5;}get offsetHeight(){return 36;}get scrollHeight(){return this.children.length*38+10;}get clientHeight(){return Math.min(this.scrollHeight,parseFloat(this.style.maxHeight)||320);}
}
class Obs{constructor(cb){this.cb=cb;}observe(){}disconnect(){}}
doc=new Events();doc.documentElement=new El('html');doc.documentElement.clientWidth=700;doc.body=new El('body');doc.documentElement.append(doc.body);doc.createElement=t=>new El(t);doc.getElementById=id=>{const walk=n=>n.id===id?n:n.children.map(walk).find(Boolean);return walk(doc.documentElement)||null;};
const win=new Events();win.innerHeight=700;win.document=doc;let frames=[];
const context={window:win,document:doc,Event:Evt,MutationObserver:Obs,requestAnimationFrame:fn=>(frames.push(fn),frames.length),cancelAnimationFrame:()=>{},setTimeout:fn=>fn(),Date,console,getComputedStyle:()=>({getPropertyValue:k=>({'--panel':'#fff','--accent':'#0071e3'}[k]||'#888'),fontFamily:'system-ui',colorScheme:'light'})};
const track=new El('select');track.id='trackFilter';track.setAttribute('aria-label','题库方向');const priority=new El('select');priority.id='priorityFilter';priority.setAttribute('aria-label','练习优先级');
for(const [sel,entries]of [[track,[['all','全部方向'],['recsys','推荐算法'],['llm','大模型'],['rl','强化学习']]],[priority,[['all','全部优先级'],['P0','P0 · 首批必练'],['P1','P1 · 核心进阶'],['P2','P2 · 专项进阶']]]]){for(const [v,t]of entries){const o=new El('option');o.value=v;o.textContent=t;sel.append(o);}doc.body.append(sel);}
const source=fs.readFileSync(require('node:path').join(__dirname,'../llm_code_lab/ui.template.html'),'utf8');
const component=source.slice(source.indexOf('/* Shared select-only combobox.'),source.indexOf('const BANK='));
assert(component.includes('window.createSelectPickers'));vm.runInNewContext(component,context);
const p=win.createSelectPickers(['trackFilter','priorityFilter']);
const a=doc.getElementById('trackFilter-picker'),b=doc.getElementById('priorityFilter-picker'),pa=doc.getElementById('trackFilter-picker-listbox'),pb=doc.getElementById('priorityFilter-picker-listbox');
const key=(el,k,props={})=>{const e=new Evt('keydown',{key:k,...props});el.dispatchEvent(e);return e;};const click=el=>el.dispatchEvent(new Evt('click'));
let input=0,change=0;track.addEventListener('input',()=>input++);track.addEventListener('change',()=>change++);
assert(track.hidden);assert.equal(track.tabIndex,-1);assert.equal(track.getAttribute('aria-hidden'),'true');assert.equal(a.getAttribute('role'),'combobox');assert.equal(pa.getAttribute('role'),'listbox');
key(a,'ArrowDown');assert.equal(a.getAttribute('aria-expanded'),'true');key(a,'ArrowDown');assert.equal(track.value,'all');assert(a.getAttribute('aria-activedescendant').endsWith('-1'));key(a,'Enter');assert.equal(track.value,'recsys');assert.equal(input,1);assert.equal(change,1);assert.equal(doc.activeElement,a);
key(a,' ');key(a,'End');key(a,'Escape');assert.equal(track.value,'recsys');assert(pa.hidden);assert.equal(a.getAttribute('aria-activedescendant'),null);
key(a,'Home');const tab=key(a,'Tab');assert(!tab.defaultPrevented);assert.equal(track.value,'all');key(a,'End');const shiftTab=key(a,'Tab',{shiftKey:true});assert(!shiftTab.defaultPrevented);assert.equal(track.value,'rl');
key(b,'p');assert(b.getAttribute('aria-activedescendant').endsWith('-1'));key(b,'p');assert(b.getAttribute('aria-activedescendant').endsWith('-2'));key(b,'Escape');assert.equal(priority.value,'all');key(b,'p');key(b,'2');key(b,'Enter');assert.equal(priority.value,'P2');
click(a);click(b);assert(pa.hidden);assert(!pb.hidden);p.closeAll(true);assert.equal(doc.activeElement,b);assert(pb.hidden);
priority.options[2].disabled=true;p.sync();key(b,'Home');key(b,'ArrowDown');key(b,'ArrowDown');assert(b.getAttribute('aria-activedescendant').endsWith('-3'));key(b,'Escape');
click(a);const row=doc.getElementById('trackFilter-picker-option-1');pa.dispatchEvent(new Evt('click',{target:row}));assert.equal(track.value,'recsys');assert.equal(doc.activeElement,a);
click(a);key(a,'End');const outside=new El('button');doc.body.append(outside);doc.dispatchEvent(new Evt('pointerdown',{target:outside}));assert(pa.hidden);assert.equal(track.value,'recsys');
track.value='all';p.sync();assert.equal(doc.getElementById('trackFilter-picker-value').textContent,'全部方向');click(a);track.disabled=true;p.sync();assert(a.disabled);assert(pa.hidden);track.disabled=false;p.sync();
a.rect={left:650,top:650,width:180,height:42,right:830,bottom:692};click(a);assert.equal(pa.dataset.side,'top');assert(parseFloat(pa.style.left)+parseFloat(pa.style.width)<=692);assert(parseFloat(pa.style.top)>=8);assert.equal(pa.style['--panel'],'#fff');a.rect={left:40,top:100,width:180,height:42,right:220,bottom:142};win.dispatchEvent(new Evt('resize'));frames.splice(0).forEach(f=>f());assert.equal(pa.dataset.side,'bottom');
p.destroy();assert(!track.hidden);assert.equal(track.getAttribute('tabindex'),null);assert.equal(track.getAttribute('aria-hidden'),null);assert.equal(doc.getElementById('trackFilter-picker'),null);assert.equal(doc.getElementById('trackFilter-picker-listbox'),null);
console.log('PASS: selection/commit/cancel, input+change, Arrow/Home/End, typeahead, Tab/ShiftTab, pointer, outside, disabled, sync, one-open, focus, placement, cleanup. DOM event mock; browser/screen-reader/visual checks remain external.');
