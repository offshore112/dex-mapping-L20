// Numerical/syntax check of the actual delivered HTML JavaScript. No browser.
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const dir = path.resolve(process.argv[2] || 'experiment/v3_L20运动学核查');
const html = fs.readFileSync(path.join(dir, 'l20_fk_viewer.html'), 'utf8');
const sceneText = html.match(/<script id="scene-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const fullScript = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
new vm.Script(fullScript); // Full-page script syntax, including event handlers.
const code = fullScript.split('// BEGIN_KINEMATICS:')[1].split('// END_KINEMATICS')[0];
// The marker line contains prose; remove it before evaluating the shared core.
const core = code.slice(code.indexOf('\n') + 1);
const sandbox = {document:{getElementById:id=>({textContent:sceneText})}};
vm.createContext(sandbox);
vm.runInContext(core+'\nglobalThis.testedFK={fk,data};',sandbox);
const reference = JSON.parse(fs.readFileSync(path.join(dir,'fk_reference_cases.json'),'utf8'));
let maxError = 0, transforms = 0;
for (const sample of reference) {
  const result = sandbox.testedFK.fk(sample.q_independent);
  for (const [name, expected] of Object.entries(sample.transforms)) {
    const actual = result.T[name];
    for (let r=0;r<4;r++) for(let c=0;c<4;c++) {
      const error = Math.abs(actual[r][c]-expected[r][c]);
      if (!Number.isFinite(error)) throw new Error('Non-finite viewer transform');
      maxError = Math.max(maxError,error);
    }
    transforms++;
  }
}
if (maxError>1e-10) throw new Error(`Viewer differs from Python: ${maxError}`);
const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);
if (new Set(ids).size!==ids.length) throw new Error('Duplicate HTML id');
for (const m of fullScript.matchAll(/\$\('([^']+)'\)/g)) {
  if (!ids.includes(m[1])) throw new Error('Missing UI element: '+m[1]);
}
const result = {status:'passed_numerical_and_syntax_checks',javascript_syntax_valid:true,
  fk_case_count:reference.length,link_transform_comparisons:transforms,
  max_matrix_abs_error_against_python:maxError,
  full_browser_click_test:'not_run_browser_tool_rejects_local_file_protocol',
  note:'DOM layout and canvas event interactions require a user browser check.'};
const output = path.join(dir,'viewer_check.json');
if(fs.existsSync(output))throw new Error('Preserve previous viewer check report');
fs.writeFileSync(output,JSON.stringify(result,null,2));
console.log(JSON.stringify(result));
