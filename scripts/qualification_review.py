"""Build local human review material or import real reviewer decisions."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from codexa_workspace.qualification_review import build_dataset_review, build_blinded_review, import_decisions
from codexa_workspace.assets import resolve_input, workspace_root
from llm_tokenizer.tokenizer import load_tokenizer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('dataset','blinded','import'))
    parser.add_argument('--prepared',type=Path)
    parser.add_argument('--ledger',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--comparison',type=Path)
    parser.add_argument('--package',type=Path)
    parser.add_argument('--decisions',type=Path)
    parser.add_argument('--mapping',type=Path)
    args=parser.parse_args()
    if args.action=='dataset':
        if args.prepared is None or args.ledger is None:
            parser.error('dataset requires --prepared and --ledger')
        tokenizer=load_tokenizer(resolve_input(workspace_root(),'checkpoints/tokenizer-base-v1/tokenizer.json'))
        result=build_dataset_review(args.prepared,args.ledger,tokenizer,args.output)
        print(json.dumps({k:v for k,v in result.items() if k!='results'},indent=2))
    elif args.action=='blinded':
        if args.prepared is None or args.comparison is None:
            parser.error('blinded requires --prepared and --comparison')
        result=build_blinded_review(args.comparison,args.prepared,args.output)
        print(json.dumps(dict(items=len(result['items']),directory=str(args.output)),indent=2))
    else:
        if args.package is None or args.decisions is None:
            parser.error('import requires --package and --decisions')
        print(json.dumps(import_decisions(args.package,args.decisions,args.output,
            comparison=args.comparison,mapping=args.mapping),indent=2))


if __name__=='__main__':
    main()
