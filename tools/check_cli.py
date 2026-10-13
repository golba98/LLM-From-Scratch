from pathlib import Path
import json,subprocess,os,sys
root=Path(__file__).resolve().parents[1]
workspace=root.parent
(root/'validation').mkdir(parents=True,exist_ok=True)
catalog=json.loads((root/'configs/artifact-catalog.json').read_text())
original=root
sys.path.insert(0,str(root/'src'))
from codexa_workspace.sibling_commands import SIBLING_COMMANDS
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['CUDA_VISIBLE_DEVICES']=''
env['PYTHONPATH']=os.pathsep.join(str(p/'src') for p in workspace.glob('LLM-*') if p.name!='LLM-From-Scratch')
results=[]
for name,module in SIBLING_COMMANDS.items():
 python=original/('.venv-specialist' if module.startswith('llm_specialist') else '.venv')/'bin/python'
 result=subprocess.run([str(python),'-m',module,'--help'],cwd='/tmp',env=env,capture_output=True,text=True,timeout=30)
 assert result.returncode==0,(module,result.stderr)
 results.append({'module':module,'help_passed':True})
(root/'validation/cli-help.json').write_text(json.dumps(results,indent=2)+'\n')
print(f'{len(results)} packaged CLI help commands passed outside repository roots.')
