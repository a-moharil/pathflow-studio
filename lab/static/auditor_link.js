// Preserve the experiment identity; the auditor explicitly selects a recorded result.
(()=>{const a=document.getElementById('auditorNav'),m=location.pathname.match(/^\/(?:experiment|observatory|healthcare|joint)\/([a-f0-9]+)(?:\/|$)/);if(a&&m)a.href='/auditor?run='+encodeURIComponent(m[1]);})();
