"""CPU preparation or explicitly approved inference-only baseline comparison."""

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

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from codexa_workspace.conversational_readiness import (workspace_root,build_preparation,
    resource_preflight,baseline_plan,run_baselines,proposal_preflight,admit_mixture,select_initializer,
    materialize_launch,authorize_launch,validate_baseline_approval,run_baseline_candidate)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('inspect','build','baseline-plan','baseline-run','baseline-candidate','proposal-preflight',
                                        'admit-mixture','select-initializer','materialize','authorize-launch',
                                        'foundation-build','foundation-run','foundation-candidate','foundation-report'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--approval-file',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--review',type=Path)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--selection',type=Path)
    parser.add_argument('--suite-review',type=Path)
    parser.add_argument('--qualification',type=Path)
    parser.add_argument('--admitted-mixture',type=Path)
    parser.add_argument('--resume-incomplete',action='store_true')
    parser.add_argument('--candidate',choices=('base','repair','balanced','original_2h','END3'))
    parser.add_argument('--approved-inference-worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--original-started-monotonic',type=float,help=argparse.SUPPRESS)
    args=parser.parse_args(); root=workspace_root(); directory=args.output.absolute()
    if args.action=='inspect':
        result=resource_preflight(root)
    elif args.action=='build':
        result=build_preparation(root,directory)
    elif args.action=='baseline-plan':
        result=baseline_plan(root,directory)
    elif args.action=='proposal-preflight':
        if args.config is None:
            parser.error('proposal-preflight requires --config')
        result=proposal_preflight(root,args.config)
    elif args.action=='admit-mixture':
        if args.review is None or args.manifest is None:
            parser.error('admit-mixture requires --review and --manifest')
        result=admit_mixture(args.manifest,args.review,directory)
    elif args.action=='select-initializer':
        if args.review is None or args.manifest is None:
            parser.error('select-initializer requires --review and --manifest comparison')
        result=select_initializer(args.manifest,args.review)
    elif args.action=='materialize':
        if None in (args.config,args.selection,args.suite_review,args.qualification):
            parser.error('materialize requires --config, --selection, --suite-review and --qualification')
        result=materialize_launch(root,args.config,args.selection,args.suite_review,args.qualification,
                                  admitted_mixture=args.admitted_mixture)
    elif args.action=='authorize-launch':
        if args.approval_file is None:
            parser.error('authorize-launch requires separate --approval-file')
        result=authorize_launch(directory,args.approval_file)
    elif args.action.startswith('foundation-'):
        from codexa_workspace import foundation_diagnosis
        if args.action=='foundation-build':
            result=foundation_diagnosis.build(root,directory)
        elif args.action=='foundation-report':
            if args.review is None:
                parser.error('foundation-report requires --review exported human decisions')
            result=foundation_diagnosis.reviewed_comparison(directory,args.review)
        else:
            if args.approval_file is None:
                parser.error('foundation execution requires precise --approval-file')
            if args.action=='foundation-candidate':
                if args.candidate is None or args.original_started_monotonic is None:
                    parser.error('foundation-candidate requires candidate and original start')
                result=foundation_diagnosis.candidate(root,directory,args.approval_file,
                    args.candidate,args.original_started_monotonic+3600)
            else:
                result=foundation_diagnosis.run(root,directory,args.approval_file,STARTED)
    else:
        if args.approval_file is None:
            parser.error('baseline-run requires separate --approval-file')
        if args.action=='baseline-candidate':
            if args.candidate is None or args.original_started_monotonic is None:
                parser.error('baseline-candidate requires candidate and original session start')
            result=run_baseline_candidate(root,directory,args.approval_file,args.candidate,
                                         started=args.original_started_monotonic)
        else:
            from llm_training.session import SessionBudget,atomic_document
            if args.resume_incomplete:
                authorization=json.loads(args.approval_file.read_text())
                if 'original_budget' not in authorization:
                    raise ValueError('Incomplete comparison requires its original recorded budget.')
                budget=SessionBudget.resume(authorization['original_budget'])
            else:
                budget=SessionBudget.start(3600,120);budget.deadline_monotonic=STARTED+3600
                budget.started_utc=STARTED_UTC;budget.deadline_utc=STARTED_UTC+3600
            control=directory/('baseline-continuation.supervisor' if args.resume_incomplete else 'baseline-isolated.supervisor')
            control.mkdir(exist_ok=False)
            atomic_document(control/'process.json',dict(worker_pid=os.getpid(),budget=budget.document()))
            result=run_baselines(root,directory,args.approval_file,started=budget.deadline_monotonic-3600,
                                 resume_incomplete=args.resume_incomplete)
            atomic_document(control/'completed.json',dict(budget=budget.document(),completed_candidates=list(result)))
    print(json.dumps(result,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
