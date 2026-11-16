// Geometry card fit note (0.17.1): settled, near plateau, not settled, and earlier-procedure pictures.
const fs=require('fs'),path=require('path'),assert=require('assert'),vm=require('vm');
const src=fs.readFileSync(path.join(__dirname,'../lab/static/geometry.js'),'utf8');
const ctx={$:id=>({value:'3'}),number:v=>String(v),console};vm.createContext(ctx);
vm.runInContext(src+';this.GV=GeometryView;',ctx);
const note=(entry)=>ctx.GV.prototype.fitNote.call({k:0},{euclidean:{dimensions:{'3':[entry]}}});
const rule={window:100,relative_tolerance:.002,minimum_iterations:300};
let t=note({optimization:{stop_reason:'plateau',iterations:812,plateau_rule:rule},dimension_consistency:{compared_with:'2-D',median_shift:.034}});
assert(t.includes('fit settled after 812 iterations')&&t.includes('from 2-D: median node shift 0.03'),t);
t=note({optimization:{stop_reason:'iteration_budget',max_iterations:3000,trailing_improvement:.004,plateau_rule:rule}});
assert(t.includes('3000-iteration limit while still improving slowly (0.40% per 100 iterations); shape near final'),t);
t=note({optimization:{stop_reason:'iteration_budget',max_iterations:120,trailing_improvement:.09,plateau_rule:rule}});
assert(t.includes('did not settle within 120 iterations'),t);
t=note({optimization:{stop_reason:'iteration_budget',iterations:120}});
assert(t.includes('earlier fitting procedure (before 0.17.1)'),t);
t=note({optimization:{stop_reason:'plateau',iterations:400,start:'random_restart_seed_710',plateau_rule:rule}});
assert(t.includes('random restart kept'),t);
console.log('fit note: ok');
