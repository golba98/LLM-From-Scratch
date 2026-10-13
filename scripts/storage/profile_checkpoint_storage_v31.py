"""Bounded publication profiling in exclusively factory-created disposable stores."""
import argparse
import cProfile
import json
import importlib.util
from pathlib import Path
import statistics
import sys
import time
import pstats
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from codexa_workspace.checkpoint_benchmark import payload, allocated
from llm_training.checkpoint_reclamation import create_disposable_fixture_store
from llm_training.recovery import RecoveryStore


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--publications', type=int, default=500)
    parser.add_argument('--legacy', action='store_true')
    parser.add_argument('--no-profile', action='store_true', help='Measure boundary timings without profiler overhead.')
    parser.add_argument('--baseline-v3', type=Path, help='Task-captured source directory for matched original V3 fixtures.')
    parser.add_argument('--tensor-side', type=int, choices=(128,1024), default=128)
    args=parser.parse_args()
    if not 6 <= args.publications <= 500:
        parser.error('6–500 publications only')
    if allocated(args.output.parent) + 128*1024**2 > 512*1024**2:
        parser.error('Validation parent exceeds the 512 MiB allowance with next-fixture reserve.')
    args.output.mkdir(parents=True, exist_ok=False)
    factory = create_disposable_fixture_store
    if args.baseline_v3:
        modules=[]
        for package,name in [('llm_architecture','components'),('llm_training','reclamation')]:
            spec=importlib.util.spec_from_file_location(package+'._v31_baseline_'+name,args.baseline_v3/('baseline_'+name+'.py'))
            module=importlib.util.module_from_spec(spec)
            sys.modules[spec.name]=module
            spec.loader.exec_module(module)
            modules.append(module)
        component,reclamation=modules
        reclamation.store_usage=component.store_usage
        sys.modules['llm_training.checkpoint_reclamation']=reclamation
        import llm_training.optimized_recovery as recovery
        import llm_training.checkpoint_retention as retention
        recovery.ComponentStore=component.ComponentStore
        retention.store_usage=component.store_usage
        factory=reclamation.create_disposable_fixture_store
    torch.set_num_threads(2)
    data=payload()
    for component in (data['model_state_dict'],data['optimizer_state_dict']['state'][0]):
        for key,value in list(component.items()):
            if isinstance(value,torch.Tensor) and value.ndim:
                component[key]=value[:args.tensor_side,:args.tensor_side].clone()
    bound=1024**2 if args.tensor_side==128 else 16*1024**2
    quota=64*1024**2 if args.tensor_side==128 else 192*1024**2
    store=(RecoveryStore(args.output/'legacy',reserve_bytes=0) if args.legacy else
        factory(args.output,max_checkpoint_bytes=bound,max_bytes=quota)[0])
    rows=[]
    before_io=Path('/proc/self/io').read_text()
    profile=cProfile.Profile()
    if not args.no_profile:
        profile.enable()
    for step in range(1,args.publications+1):
        data['training_state'].update(optimizer_step=step,micro_step=step,tokens_seen=step*128)
        data['resume_context']['budget']['elapsed_seconds']=step
        data['created_at_utc']=str(step)
        data['model_state_dict']['weight'].add_(.0001)
        data['optimizer_state_dict']['state'][0]['exp_avg'].add_(.0001)
        start,cpu=time.perf_counter(),time.process_time()
        flags=dict(best=step==1,final=step==args.publications)
        if not args.legacy:
            flags.update(best_validation=step==2,boundary=step==1)
        store.save(data,**flags)
        rows.append(dict(step=step,seconds=time.perf_counter()-start,cpu=time.process_time()-cpu))
    profile.disable()
    if not args.no_profile:
        profile.dump_stats(str(args.output/'profile.pstats'))
        with (args.output/'profile.txt').open('w') as f:
            pstats.Stats(profile,stream=f).sort_stats('cumulative').print_stats(45)
    checkpoints=[6,30,62,120,250,500]
    windows=[]
    for end in checkpoints:
        if end>args.publications:continue
        values=[r['seconds'] for r in rows[max(0,end-30):end]]
        windows.append(dict(publications=end,median=statistics.median(values),p95=statistics.quantiles(values,n=20,method='inclusive')[18],worst=max(values)))
    loaded=store.load() if not args.legacy else __import__('llm_architecture.checkpointing',fromlist=['read_native_checkpoint']).read_native_checkpoint(store.resolve())
    assert torch.equal(data['model_state_dict']['weight'],loaded['model_state_dict']['weight'])
    assert torch.equal(data['optimizer_state_dict']['state'][0]['exp_avg'],loaded['optimizer_state_dict']['state'][0]['exp_avg'])
    assert loaded['training_state']==data['training_state']
    if not args.legacy:
        assert store.load('previous')['training_state']['optimizer_step']==args.publications-1
        assert store.load('best')['training_state']['optimizer_step']==1
        assert store.load('best_validation')['training_state']['optimizer_step']==2
        plan=store.plan_retention();assert not plan['blocked'];assert plan['physical']['accounted_bytes']<=store.policy.max_bytes
    size=allocated(args.output)
    assert size<128*1024**2
    (args.output/'results.json').write_text(json.dumps(dict(rows=rows,windows=windows,allocated_bytes=size,
        profiled=not args.no_profile,io_before=before_io,io_after=Path('/proc/self/io').read_text()),indent=2))
    print(json.dumps(windows))

if __name__=='__main__':main()
