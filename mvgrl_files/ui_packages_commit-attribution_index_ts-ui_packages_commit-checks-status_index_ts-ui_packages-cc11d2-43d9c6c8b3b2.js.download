"use strict";(globalThis.webpackChunk=globalThis.webpackChunk||[]).push([["ui_packages_commit-attribution_index_ts-ui_packages_commit-checks-status_index_ts-ui_packages-cc11d2"],{14744:(e,t,o)=>{o.d(t,{D:()=>u,C:()=>N});var r=o(85893),a=o(57294),i=o(78212),n=o(42483),l=o(73290),s=o(97011);function c(e){return e.path?.startsWith("/apps/")??!1}var d=o(38490);function p({renderTooltip:e,author:t,children:o}){return!1===e?(0,r.jsx)(r.Fragment,{children:o}):(0,r.jsx)(d.Z,{"aria-label":`commits by ${t.login}`,direction:"se",children:o})}try{p.displayName||(p.displayName="AuthorTooltip")}catch{}function u({author:e,repo:t,avatarSize:o,sx:d={},includeTooltip:u=!0}){if(!e)return null;let h=(0,r.jsx)(a.O,{"aria-label":`${e.login||"author"}`,src:e.avatarUrl,alt:`${e.login||"author"}`,sx:{mr:2,mt:"-1px",ml:"1px"},size:o,square:c(e)});return(0,r.jsxs)(n.Z,{sx:{display:"flex",flexDirection:"row",alignItems:"center",...d},"data-testid":"author-avatar",children:[e.path?(0,r.jsx)(l.Z,{href:e.path,"data-testid":"avatar-icon-link","data-hovercard-url":e.login?(0,i.zP)({owner:e.login}):void 0,children:h}):h,e.login?(0,r.jsx)(p,{author:e,renderTooltip:u,children:(0,r.jsx)(l.Z,{muted:!0,href:(0,i.OI)({repo:t,author:e.login}),"aria-label":`commits by ${e.login}`,sx:{fontWeight:600,whiteSpace:"nowrap",color:"fg.default","&:hover":{color:"fg.default",textDecoration:"underline"}},children:e.login})}):(0,r.jsx)(s.Z,{sx:{fontWeight:600,whiteSpace:"nowrap",color:"fg.default"},children:e.displayName})]})}try{u.displayName||(u.displayName="AuthorAvatar")}catch{}var h=o(67294),m=o(52516),f=o(79902),x=o(66280);function g({authors:e,repo:t}){let o=e.length,[a,i]=(0,h.useState)(!1),n=(0,h.useRef)(null);return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsxs)(l.Z,{as:"button","aria-label":`Show ${o} authors`,"data-testid":"authors-dialog-anchor",onClick:()=>{i(!0)},sx:{mx:1},ref:n,muted:!0,children:[o," ","people"]}),a&&(0,r.jsx)(x.V,{title:`${o} authors`,onClose:()=>{i(!1),setTimeout(()=>n.current?.focus(),25)},width:"medium",height:o>=12?"small":"auto",renderBody:()=>(0,r.jsx)(m.S,{sx:{overflowY:"auto",py:2},"data-testid":"contributor-dialog-list",children:e.map((e,o)=>(0,r.jsx)(y,{author:e,repo:t},`${e.login}_${o}`))})})]})}function y({author:e,repo:t}){return(0,r.jsxs)(m.S.LinkItem,{sx:{display:"flex",flexDirection:"row",fontSize:1,py:2,color:"fg.default","&:hover":{backgroundColor:"canvas.subtle"}},"data-testid":"contributor-dialog-row",href:(0,i.OI)({repo:t,author:e.login??""}),children:[(0,r.jsx)(a.O,{src:e.avatarUrl,alt:e.login??e.displayName,sx:{mr:2},"aria-hidden":"true",square:c(e)}),(0,r.jsx)(f.Z,{inline:!0,title:e.login??e.displayName??"",children:e.login??e.displayName})]})}try{g.displayName||(g.displayName="AuthorsDialog")}catch{}try{y.displayName||(y.displayName="AuthorRow")}catch{}var b=o(90836);function j({authors:e,avatarSize:t}){return(0,r.jsx)(b.Z,{children:e.slice(0,5).map((e,o)=>(0,r.jsx)(a.O,{"data-testid":"commit-stack-avatar",src:e.avatarUrl,alt:e.login??e.displayName,"data-hovercard-url":(0,i.zP)({owner:e.login??""}),square:c(e),size:t},`${e.login}_${o}`))})}try{j.displayName||(j.displayName="CommitAuthorStack")}catch{}function v({author:e,repo:t,sx:o={},includeTooltip:a=!0}){return e?(0,r.jsx)(n.Z,{sx:{display:"flex",flexDirection:"row",alignItems:"center",...o},"data-testid":"author-link",children:e.login?(0,r.jsx)(p,{author:e,renderTooltip:a,children:(0,r.jsx)(l.Z,{muted:!0,href:(0,i.OI)({repo:t,author:e.login}),"aria-label":`commits by ${e.login}`,sx:{fontWeight:600,whiteSpace:"nowrap",color:"fg.default","&:hover":{color:"fg.default",textDecoration:"underline"}},children:e.login})}):(0,r.jsx)(s.Z,{sx:{fontWeight:600,whiteSpace:"nowrap",color:"fg.default"},children:e.displayName})}):null}try{v.displayName||(v.displayName="AuthorLink")}catch{}function k({author:e,repo:t,avatarSize:o,includeTooltip:a}){return(0,r.jsx)(u,{author:e,repo:t,avatarSize:o,includeTooltip:a})}function C({author:e,committer:t,repo:o,avatarSize:a,includeTooltip:i}){return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(j,{authors:[e,t],avatarSize:a}),(0,r.jsx)(v,{author:e,repo:o,sx:{pl:1},includeTooltip:i})]})}function w({authors:e,repo:t,avatarSize:o,includeTooltip:a}){return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(j,{authors:e,avatarSize:o}),e.map((o,i)=>(0,r.jsxs)(h.Fragment,{children:[(0,r.jsx)(v,{author:o,repo:t,sx:{pl:1},includeTooltip:a}),i!==e.length-1&&(0,r.jsx)(n.Z,{as:"span",sx:{pl:1},children:"and"})]},`${o.login}_${i}`))]})}function S({authors:e,repo:t,avatarSize:o}){return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(j,{authors:e,avatarSize:o}),(0,r.jsx)(g,{authors:e,repo:t})]})}function N({authors:e,committer:t,committerAttribution:o,repo:a,avatarSize:i,relativeTime:l,includeVerbs:s=!0,includeAuthorTooltip:c=!0}){let d=1===e.length&&!o,p=1===e.length&&o,u=2===e.length&&!o,h=s?{pl:1}:{};return(0,r.jsxs)(n.Z,{sx:{display:"flex",flexDirection:"row",flexWrap:"wrap",alignItems:"center"},children:[d&&(0,r.jsx)(k,{author:e[0],repo:a,avatarSize:i,includeTooltip:c}),p&&(0,r.jsx)(C,{author:e[0],committer:t,repo:a,avatarSize:i,includeTooltip:c}),u&&(0,r.jsx)(w,{authors:e,repo:a,avatarSize:i,includeTooltip:c}),!d&&!p&&!u&&(0,r.jsx)(S,{authors:e,repo:a,avatarSize:i}),o?(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(n.Z,{as:"span",sx:h,children:s?"authored and":"and"}),(0,r.jsx)(v,{author:t,repo:a,sx:{pl:1},includeTooltip:c}),(0,r.jsx)(n.Z,{as:"span",sx:h,children:s&&"committed"})]}):(0,r.jsx)(n.Z,{as:"span",sx:h,children:s&&"committed"}),l]})}try{k.displayName||(k.displayName="SingleAuthor")}catch{}try{C.displayName||(C.displayName="AuthorAndCommitter")}catch{}try{w.displayName||(w.displayName="TwoAuthors")}catch{}try{S.displayName||(S.displayName="MultipleAuthors")}catch{}try{N.displayName||(N.displayName="CommitAttribution")}catch{}},4751:(e,t,o)=>{o.d(t,{AF:()=>j,vC:()=>C,fQ:()=>N});var r=o(85893),a=o(85529),i=o(98833),n=o(97011),l=o(42483),s=o(78912),c=o(50919),d=o(67294),p=o(52516),u=o(74121),h=o(66280),m=o(57294),f=o(38490),x=o(73290);function g({checkRun:e}){let{icon:t,iconColor:o}=function(e){switch(e){case"check":return{icon:a.CheckIcon,iconColor:"success.fg"};case"dot-fill":return{icon:a.DotFillIcon,iconColor:"attention.fg"};case"stop":return{icon:a.StopIcon,iconColor:"muted.fg"};case"issue-reopened":return{icon:a.IssueReopenedIcon,iconColor:"muted.fg"};case"clock":return{icon:a.ClockIcon,iconColor:"attention.fg"};case"square-fill":return{icon:a.SquareFillIcon,iconColor:"fg.default"};case"skip":return{icon:a.SkipIcon,iconColor:"muted.fg"};case"alert":return{icon:a.AlertIcon,iconColor:"danger.fg"};default:return{icon:a.XIcon,iconColor:"danger.fg"}}}(e.icon),s="in_progress"===e.state;return(0,r.jsxs)(l.Z,{"data-testid":"check-run-item",sx:{display:"flex",borderBottomWidth:"1px",borderBottomStyle:"solid",borderBottomColor:"border.default",backgroundColor:"canvas.subtle",height:"38px",py:2,pr:3,pl:"12px",alignItems:"baseline"},children:[s?(0,r.jsx)(l.Z,{sx:{height:"16px",width:"16px",minWidth:"16px",alignSelf:"center",mx:"7px"},children:(0,r.jsxs)("svg",{fill:"none",viewBox:"0 0 16 16",className:"anim-rotate","aria-hidden":"true",role:"img",children:[(0,r.jsx)("path",{opacity:".5",d:"M8 15A7 7 0 108 1a7 7 0 000 14v0z",stroke:"#dbab0a",strokeWidth:"2"}),(0,r.jsx)("path",{d:"M15 8a7 7 0 01-7 7",stroke:"#dbab0a",strokeWidth:"2"}),(0,r.jsx)("path",{d:"M8 12a4 4 0 100-8 4 4 0 000 8z",fill:"#dbab0a"})]})}):(0,r.jsx)(i.Z,{icon:t,sx:{color:o,margin:"0px 7px",alignSelf:"center"}}),(0,r.jsx)(f.Z,{"aria-label":e.avatarDescription,direction:"e",children:(0,r.jsx)(x.Z,{href:e.avatarUrl,sx:{mr:2},children:(0,r.jsx)(m.O,{square:!0,src:e.avatarLogo,sx:{backgroundColor:e.avatarBackgroundColor}})})}),(0,r.jsxs)(n.Z,{sx:{overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",fontSize:"13px",color:"fg.muted"},children:[(0,r.jsxs)(n.Z,{sx:{fontWeight:"bold",color:"fg.default",mr:"2px"},children:[e.name," "]}),e.pending?(0,r.jsx)(n.Z,{sx:{fontStyle:"italic"},children:e.additionalContext}):e.additionalContext,e.description&&(0,r.jsxs)(n.Z,{children:[" ","- ",e.pending?(0,r.jsx)(n.Z,{sx:{fontStyle:"italic"},children:e.description}):e.description]})]}),(0,r.jsx)(x.Z,{href:e.targetUrl,sx:{pl:"12px",fontSize:"13px",marginLeft:"auto"},children:"Details"})]})}try{g.displayName||(g.displayName="CheckRunItem")}catch{}function y({checkRuns:e}){return(0,r.jsx)(l.Z,{sx:{display:"flex",flexDirection:"column",maxHeight:"230px",overflow:"auto"},children:e.map((e,t)=>(0,r.jsx)(g,{checkRun:e},t))})}try{y.displayName||(y.displayName="ChecksStatusBadgeFooter")}catch{}function b({checksHeaderState:e}){switch(e){case"SUCCEEDED":return(0,r.jsx)(n.Z,{sx:{fontWeight:"bold",fontSize:2},children:"All checks have passed"});case"FAILED":return(0,r.jsx)(n.Z,{sx:{color:"checks.donutError",fontWeight:"bold",fontSize:2},children:"All checks have failed"});case"PENDING":return(0,r.jsx)(n.Z,{sx:{color:"checks.donutPending",fontWeight:"bold",fontSize:2},children:"Some checks haven\u2019t completed yet"});default:return(0,r.jsx)(n.Z,{sx:{color:"checks.donutError",fontWeight:"bold",fontSize:2},children:"Some checks were not successful"})}}try{b.displayName||(b.displayName="HeaderState")}catch{}function j(e){let{combinedStatus:t,isOpen:o,rounded:a=!1,onDismiss:i}=e,n=t?(0,r.jsx)(b,{checksHeaderState:t.checksHeaderState}):"Loading...";return o?(0,r.jsx)(h.V,{onClose:i,sx:{overflowY:"auto",backgroundColor:"canvas.default",boxShadow:"none",...a?{border:"1px solid",borderColor:"border.default",borderBottom:0}:{borderRadius:0}},title:n,subtitle:t?t.checksStatusSummary:void 0,width:"xlarge",renderBody:()=>(0,r.jsx)(h.V.Body,{sx:{padding:0},children:(0,r.jsx)(p.S,{sx:{padding:0},children:t?(0,r.jsx)(y,{checkRuns:t.checkRuns}):(0,r.jsx)(l.Z,{sx:{display:"flex",justifyContent:"center",p:2},children:(0,r.jsx)(u.Z,{size:"medium"})})})})}):null}try{j.displayName||(j.displayName="CheckStatusDialog")}catch{}let v={success:{circled:a.CheckCircleIcon,filled:a.CheckCircleFillIcon,default:a.CheckIcon,color:"checks.donutSuccess"},pending:{circled:a.CircleIcon,filled:a.DotFillIcon,default:a.DotFillIcon,color:"checks.donutPending"},error:{circled:a.XCircleIcon,filled:a.XCircleFillIcon,default:a.XIcon,color:"checks.donutError"}};function k({descriptionText:e,icon:t,iconColor:o}){return(0,r.jsxs)("span",{"data-testid":"checks-status-badge-icon-only",children:[(0,r.jsx)(i.Z,{icon:t,"aria-label":"See all checks",sx:{color:o}}),e&&(0,r.jsxs)(n.Z,{children:[" ",e]})]})}function C(e){let{statusRollup:t,combinedStatus:o,variant:a="default",disablePopover:i,buttonSx:n,size:p="medium",descriptionText:u="",rounded:h=!1}=e,[m,f]=(0,d.useState)(!1),x=(0,d.useRef)(null),g=v[t],{icon:y,iconColor:b}={icon:g?.[a]||v.error[a],iconColor:g?.color||v.error.color};return i?(0,r.jsx)(k,{descriptionText:u,icon:y,iconColor:b}):(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(l.Z,{onClick:()=>{f(!0),e.onWillOpenPopup},onMouseEnter:e.onWillOpenPopup,children:u?(0,r.jsx)(s.z,{"data-testid":"checks-status-badge-button",leadingVisual:y,variant:"invisible",size:p,"aria-label":o?.checksStatusSummary??`Status checks: ${t}`,sx:{p:1,color:"fg.default",fontWeight:"normal",svg:{color:b},...n},ref:x,children:u}):(0,r.jsx)(c.h,{"data-testid":"checks-status-badge-icon",icon:y,variant:"invisible",size:p,"aria-label":o?.checksStatusSummary??t,sx:{py:0,px:0,mr:2,svg:{color:b},":hover:not([disabled])":{bg:"pageHeaderBg"},...n},ref:x})}),m&&(0,r.jsx)(j,{combinedStatus:o,isOpen:m,onDismiss:()=>{f(!1),setTimeout(()=>{x.current?.focus()},0)},rounded:h})]})}try{k.displayName||(k.displayName="IconOnlyStatus")}catch{}try{C.displayName||(C.displayName="ChecksStatusBadge")}catch{}var w=o(78212),S=o(89445);function N(e,t){let[o,r]=(0,d.useState)(),[a,i]=(0,d.useState)(),n=(0,d.useCallback)(async()=>{if(a!==e&&(i(e),r(void 0),e)){let o=(0,w.S$)(t,e),a=await (0,S.v)(o);r(await a.json())}},[e,a,t]);return[o,n]}},68912:(e,t,o)=>{o.d(t,{m:()=>u,z:()=>d});var r=o(85893),a=o(37169),i=o(85529),n=o(38490),l=o(42483),s=o(50919),c=o(67294);function d(e){if("clipboard"in navigator)return navigator.clipboard.writeText(e);let t=document.body;if(!t)return Promise.reject(Error());let o=function(e){let t=document.createElement("pre");return t.style.width="1px",t.style.height="1px",t.style.position="fixed",t.style.top="5px",t.textContent=e,t}(e);return t.appendChild(o),!function(e){if("clipboard"in navigator)return navigator.clipboard.writeText(e.textContent||"");let t=getSelection();if(null==t)return Promise.reject(Error());t.removeAllRanges();let o=document.createRange();o.selectNodeContents(e),t.addRange(o),document.execCommand("copy"),t.removeAllRanges(),Promise.resolve()}(o),t.removeChild(o),Promise.resolve()}function p({sx:e,tooltipProps:t}){return(0,r.jsx)(n.Z,{"aria-label":"Copied!",sx:e,...t,children:(0,r.jsx)(l.Z,{as:"span",sx:{display:"inline-block",color:"success.fg",p:1,mr:1},children:(0,r.jsx)(i.CheckIcon,{})})})}function u({icon:e=i.CopyIcon,size:t="medium",onCopy:o,sx:l,textToCopy:u,tooltipProps:h,confirmationComponent:m=(0,r.jsx)(p,{sx:l,tooltipProps:h}),ariaLabel:f,accessibleButton:x}){let[g,y]=c.useState(!1),b=(0,a.Z)(),j=f??`Copy ${u} to clipboard`;return g?(0,r.jsx)(r.Fragment,{children:m}):(0,r.jsx)(n.Z,{"aria-label":j,...h,children:(0,r.jsx)(s.h,{"aria-label":j,icon:e,variant:"invisible",size:t,tabIndex:!1===x?-1:0,sx:{...l},onClick:()=>{y(!0),d(u),o?.(),setTimeout(()=>b()&&y(!1),2e3)}})})}try{p.displayName||(p.displayName="CopyConfirmationCheck")}catch{}try{u.displayName||(u.displayName="CopyToClipboardButton")}catch{}},95628:(e,t,o)=>{o.d(t,{M:()=>i});let r=e=>{let t=getComputedStyle(e,null);return["overflow","overflow-y","overflow-x"].some(e=>{let o=t.getPropertyValue(e);return"auto"===o||"scroll"===o})},a=(e,t)=>e&&null!==e.parentNode?a(e.parentNode,t.concat([e])):t;function i(e){if(!(e instanceof HTMLElement||e instanceof SVGElement))return;let t=a(e.parentNode,[]);for(let e of t)if((e instanceof HTMLElement||e instanceof SVGElement)&&r(e))return e;return document.scrollingElement||document.documentElement}},78806:(e,t,o)=>{o.d(t,{Z:()=>r});let r=(e,t)=>{let o=new URL(e,window.location.origin),r=new URL(t,window.location.origin),a=r.href.includes("#");return a&&o.host===r.host&&o.pathname===r.pathname&&o.search===r.search}},2048:(e,t,o)=>{o.d(t,{g:()=>a,y:()=>i});var r=o(17891);let a=()=>r.M()?.enabled_features??{},i=e=>!!a()[e]},37169:(e,t,o)=>{o.d(t,{Z:()=>i});var r=o(78249),a=o(67294);function i(){let e=(0,a.useRef)(!1),t=(0,a.useCallback)(()=>e.current,[]);return(0,r.g)(()=>(e.current=!0,()=>{e.current=!1}),[]),t}},68203:(e,t,o)=>{o.d(t,{s:()=>c});var r=o(67294),a=o(89250),i=o(12599),n=o(78806),l=o(45055),s=o(68202);let c=()=>{let{routes:e,history:t}=r.useContext(l.I),c=(0,a.s0)();return r.useCallback((r,a)=>{let l=(0,i.i3)(r).pathname,d=!(0,i.fp)(e,l);if(d){let e=t.createHref(r);(async()=>{let{softNavigate:t}=await Promise.all([o.e("vendors-node_modules_github_turbo_dist_turbo_es2017-esm_js"),o.e("ui_packages_soft-navigate_soft-navigate_ts")]).then(o.bind(o,75198));t(e)})()}else{(0,n.Z)(location.href,r.toString())||(0,s.LD)("react"),c(r,a);let{turbo:e,...t}=window.history.state;window.history.replaceState({...t,skipTurbo:!0},"",location.href)}},[t,c,e])}},32769:(e,t,o)=>{o.d(t,{H:()=>l,d:()=>n});var r=o(85893),a=o(67294);let i=a.createContext({});function n({repository:e,children:t}){return(0,r.jsxs)(i.Provider,{value:e,children:[" ",t," "]})}function l(){return a.useContext(i)}try{i.displayName||(i.displayName="CurrentRepositoryContext")}catch{}try{n.displayName||(n.displayName="CurrentRepositoryProvider")}catch{}},57294:(e,t,o)=>{o.d(t,{O:()=>l});var r=o(85893),a=o(67294),i=o(26012),n=o(86283);let l=(0,a.forwardRef)(function({src:e,size:t=20,...o},l){let s=(0,a.useMemo)(()=>{let o=new URL(e,n.ssrSafeLocation.origin);return o.searchParams.has("size")||o.searchParams.has("s")||o.searchParams.set("size",String(2*Number(t))),o.toString()},[e,t]);return(0,r.jsx)(i.Z,{ref:l,src:s,size:t,"data-testid":"github-avatar",...o})});try{l.displayName||(l.displayName="GitHubAvatar")}catch{}},45222:(e,t,o)=>{o.d(t,{h:()=>p});var r=o(85893),a=o(42379),i=o(15173),n=o(41905),l=o(86010),s=o(67294),c=o(15388);let d=c.ZP.span`
  &::before {
    position: absolute;
    z-index: 1000001;
    display: none;
    width: 0px;
    height: 0px;
    color: ${(0,a.U2)("colors.neutral.emphasisPlus")};
    pointer-events: none;
    content: '';
    border: 6px solid transparent;
    opacity: 0;
  }
  &::after {
    position: absolute;
    z-index: 1000000;
    display: none;
    padding: 0.5em 0.75em;
    font: normal normal 11px/1.5 ${(0,a.U2)("fonts.normal")};
    -webkit-font-smoothing: subpixel-antialiased;
    color: ${(0,a.U2)("colors.fg.onEmphasis")};
    text-align: center;
    text-decoration: none;
    text-shadow: none;
    text-transform: none;
    letter-spacing: normal;
    word-wrap: break-word;
    white-space: pre;
    pointer-events: none;
    content: attr(aria-label);
    background: ${(0,a.U2)("colors.neutral.emphasisPlus")};
    border-radius: ${(0,a.U2)("radii.1")};
    opacity: 0;
  }
  /* delay animation for tooltip */
  @keyframes tooltip-appear {
    from {
      opacity: 0;
    }
    to {
      opacity: 1;
    }
  }
  &.tooltipped-open,
  &:hover,
  &:active,
  &:focus {
    &::before,
    &::after {
      display: inline-block;
      text-decoration: none;
      animation-name: tooltip-appear;
      animation-duration: 0.1s;
      animation-fill-mode: forwards;
      animation-timing-function: ease-in;
      animation-delay: 0.4s;
    }
  }

  &.tooltipped-no-delay.tooltipped-open,
  &.tooltipped-no-delay:hover,
  &.tooltipped-no-delay:active,
  &.tooltipped-no-delay:focus {
    &::before,
    &::after {
      animation-delay: 0s;
    }
  }

  /* Tooltipped south */
  &.tooltipped-s,
  &.tooltipped-se,
  &.tooltipped-sw {
    &::after {
      top: 100%;
      right: 50%;
      margin-top: 6px;
    }
    &::before {
      top: auto;
      right: 50%;
      bottom: -7px;
      margin-right: -6px;
      border-bottom-color: ${(0,a.U2)("colors.neutral.emphasisPlus")};
    }
  }
  &.tooltipped-se {
    &::after {
      right: auto;
      left: 50%;
      margin-left: -${(0,a.U2)("space.3")};
    }
  }
  &.tooltipped-sw::after {
    margin-right: -${(0,a.U2)("space.3")};
  }
  /* Tooltips above the object */
  &.tooltipped-n,
  &.tooltipped-ne,
  &.tooltipped-nw {
    &::after {
      right: 50%;
      bottom: 100%;
      margin-bottom: 6px;
    }
    &::before {
      top: -7px;
      right: 50%;
      bottom: auto;
      margin-right: -6px;
      border-top-color: ${(0,a.U2)("colors.neutral.emphasisPlus")};
    }
  }
  &.tooltipped-ne {
    &::after {
      right: auto;
      left: 50%;
      margin-left: -${(0,a.U2)("space.3")};
    }
  }
  &.tooltipped-nw::after {
    margin-right: -${(0,a.U2)("space.3")};
  }
  /* Move the tooltip body to the center of the object. */
  &.tooltipped-s::after,
  &.tooltipped-n::after {
    transform: translateX(50%);
  }
  /* Tooltipped to the left */
  &.tooltipped-w {
    &::after {
      right: 100%;
      bottom: 50%;
      margin-right: 6px;
      transform: translateY(50%);
    }
    &::before {
      top: 50%;
      bottom: 50%;
      left: -7px;
      margin-top: -6px;
      border-left-color: ${(0,a.U2)("colors.neutral.emphasisPlus")};
    }
  }
  /* tooltipped to the right */
  &.tooltipped-e {
    &::after {
      bottom: 50%;
      left: 100%;
      margin-left: 6px;
      transform: translateY(50%);
    }
    &::before {
      top: 50%;
      right: -7px;
      bottom: 50%;
      margin-top: -6px;
      border-right-color: ${(0,a.U2)("colors.neutral.emphasisPlus")};
    }
  }
  &.tooltipped-align-right-2::after {
    right: 0;
    margin-right: 0;
  }
  &.tooltipped-align-right-2::before {
    right: 15px;
  }
  &.tooltipped-align-left-2::after {
    left: 0;
    margin-left: 0;
  }
  &.tooltipped-align-left-2::before {
    left: 10px;
  }
  ${i.Z};
`,p=(0,s.forwardRef)(function({direction:e="n",className:t,text:o,noDelay:a,align:i,wrap:s,open:c=!1,portalProps:p={},...u},h){let m=(0,l.W)(t,`tooltipped-${e}`,i&&`tooltipped-align-${i}-2`,a&&"tooltipped-no-delay",s&&"tooltipped-multiline",c&&"tooltipped-open");return(0,r.jsx)(n.h,{...p,children:(0,r.jsx)(d,{ref:h,role:"tooltip","aria-label":o,...u,sx:{position:"fixed",zIndex:1,...u.sx},className:m})})});try{p.displayName||(p.displayName="ControlledTooltip")}catch{}},52793:(e,t,o)=>{o.d(t,{u:()=>s});var r=o(85893),a=o(48030),i=o(67294),n=o(45222),l=o(95628);let s=(0,i.forwardRef)(function({contentRef:e,open:t,anchoredPositionAlignment:o,anchorSide:s,anchorOffset:c,alignmentOffset:d,allowOutOfBounds:p,...u},h){let m=(0,i.useRef)(null);(0,i.useImperativeHandle)(h,()=>m.current);let f=(0,i.useRef)({left:0,top:0}),x=(0,i.useSyncExternalStore)((0,i.useCallback)(o=>{if(!m.current||!e.current||!t)return()=>void 0;let r=(0,l.M)(e.current);return r?.addEventListener("scroll",o),()=>{r?.removeEventListener("scroll",o)}},[e,t]),(0,i.useCallback)(()=>{if(!m.current||!e.current)return f.current;let t=(0,a.N)(m.current,e.current,{align:o??"center",side:s??"outside-top",alignmentOffset:d??0,anchorOffset:c??0,allowOutOfBounds:p});return(t.left!==f.current.left||t.top!==f.current.top)&&(f.current=t),f.current},[e,d,c,o,s,p]));return(0,r.jsx)(n.h,{...u,ref:m,open:t,style:{position:"absolute",...x,...u.style}})});try{s.displayName||(s.displayName="PortalTooltip")}catch{}},60348:(e,t,o)=>{o.d(t,{r:()=>c});var r=o(85893),a=o(67294),i=o(12599),n=o(79655),l=o(45055),s=o(86283);let c=a.forwardRef(({to:e,reloadDocument:t,...o},c)=>{let{routes:d}=a.useContext(l.I),p=(0,i.i3)(e,s.ssrSafeLocation.pathname).pathname;return t=t??!(0,i.fp)(d,p),(0,r.jsx)(n.rU,{to:e,...o,reloadDocument:t,ref:c})});c.displayName="Link"}}]);
//# sourceMappingURL=ui_packages_commit-attribution_index_ts-ui_packages_commit-checks-status_index_ts-ui_packages-cc11d2-d6b83ef8f3b4.js.map