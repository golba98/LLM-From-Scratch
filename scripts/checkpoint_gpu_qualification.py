"""Plan or separately authorize a bounded full-model qualification; default is read-only."""

import argparse
import json
from pathlib import Path
import sys
import time
import subprocess
import signal
import os

STARTED=time.monotonic()
STARTED_UTC=time.time()
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
from codexa_workspace.checkpoint_gpu_qualification import (qualification_plan,execute_qualification,
    qualification_execution_preflight)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=8)
    parser.add_argument('--sequence-length',type=int,default=2048)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--approval-file',type=Path)
    parser.add_argument('--approved-worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--original-started-monotonic',type=float,help=argparse.SUPPRESS)
    args=parser.parse_args()
    plan=qualification_plan(args.checkpoint,args.output,updates=args.updates,
                            sequence_length=args.sequence_length)
    if args.execute:
        if args.approval_file is None:
            parser.error('--execute requires separate --approval-file')
        if args.approved_worker:
            result=execute_qualification(plan,args.approval_file,
                started=args.original_started_monotonic if args.original_started_monotonic is not None else STARTED)
        else:
            qualification_execution_preflight(plan,args.approval_file)
            if time.monotonic()-STARTED >= 780:
                raise TimeoutError('Qualification startup exhausted its work allowance.')
            from llm_training.session import SessionBudget, atomic_document
            from codexa_workspace.chat_operations import watch_owned_process
            control=Path(plan['output']+'.supervisor')
            control.mkdir(exist_ok=False)
            budget=SessionBudget.start(900,120)
            budget.deadline_monotonic=STARTED+900
            budget.started_utc=STARTED_UTC
            budget.deadline_utc=STARTED_UTC+900
            command=[sys.executable,str(Path(__file__).absolute()),*sys.argv[1:],'--approved-worker',
                     '--original-started-monotonic',str(STARTED)]
            with (control/'console.log').open('x') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                atomic_document(control/'process.json',dict(worker_pid=child.pid,budget=budget.document()))
                try:
                    code=watch_owned_process(child,budget,control)
                except BaseException:
                    if child.poll() is None:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:
                            child.wait(timeout=max(.1,min(30,budget.remaining)))
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid,signal.SIGKILL); child.wait()
                    raise
            result=dict(worker_returncode=code,supervisor=str(control),production_ready=False)
            print(json.dumps(result,indent=2))
            return code
    else:
        result=plan
    print(json.dumps(result,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
