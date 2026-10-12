#!/usr/bin/env python3
"""Bounded offline investigation. Never imports a training entry point or optimizer.

Run with the existing integration virtualenv. Each phase is bounded to 30 minutes.
The directory must be a new diagnostic directory, never a historical run directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RUN = ROOT / "outputs/training/codexa-934m-foundation-5h-v1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def validation_source_list(mapping, windows):
    """The frozen source file maps window IDs to sources, not array positions."""
    return [mapping[str(int(window))] for window in windows]


def command(args):
    p = subprocess.run(args, text=True, capture_output=True, timeout=30)
    return dict(command=args, exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr)


def mechanical(ids, text):
    """Independent word/token repetition metrics; never semantic quality scores."""
    words = re.findall(r"\b[\w']+\b", text.casefold())
    result = {}
    for name, sequence in (("token", ids), ("word", words)):
        for n in (1, 2, 3, 4):
            grams = [tuple(sequence[i:i+n]) for i in range(max(0, len(sequence)-n+1))]
            distinct = len(set(grams)) / len(grams) if grams else 0.0
            result[f"{name}_distinct_{n}"] = distinct
            result[f"{name}_repeat_{n}"] = 1-distinct if grams else 0.0
    lines = Counter(line.strip() for line in text.splitlines() if line.strip())
    result["maximum_repeated_line_count"] = max(lines.values(), default=0)
    result["historical_severe"] = result["word_repeat_4"] > .5
    result["independent_severe"] = (result["historical_severe"] or
        result["token_repeat_4"] > .65 or result["maximum_repeated_line_count"] >= 3)
    return result


def comparison(a, b):
    import torch
    a, b = a.float(), b.float()
    delta = (a-b).abs()
    p, q = a.log_softmax(-1), b.log_softmax(-1)
    return dict(max_absolute=delta.max().item(),
        max_relative_stabilized=(delta / a.abs().clamp_min(1e-3)).max().item(),
        normalized_rms=(delta.square().mean().sqrt() / a.square().mean().sqrt().clamp_min(1e-6)).item(),
        kl=max(0.0, (p.exp()*(p-q)).sum(-1).max().item()),
        argmax_agreement=a.argmax(-1).eq(b.argmax(-1)).float().mean().item(),
        fp32_close=bool(torch.allclose(a, b, atol=1e-4, rtol=1e-4)),
        bf16_screen_close=bool(torch.allclose(a, b, atol=.125, rtol=.02)))


def conversational_cases(tokenizer):
    from llm_tokenizer.sft import ChatMessage, format_chat_prompt
    pairs = [
        ("greeting", "Hello!", None), ("greeting", "How are you today?", None),
        ("factual", "What is the capital of France?", "Paris"),
        ("factual", "How many days are in one week?", "7"),
        ("instruction", "Reply with only the word READY.", "READY"),
        ("instruction", "Write the numbers 1, 2, 3 on separate lines. No other text.", "1\n2\n3"),
        ("explanation", "Explain why ice floats on water in two sentences.", None),
        ("explanation", "Briefly explain what a computer variable is.", None),
        ("multi_sentence", "Give three sentences about caring for a houseplant.", None),
        ("multi_sentence", "Explain how to make a cup of tea in three sentences.", None),
        ("reasoning", "I have 3 apples and receive 2 more. How many apples do I have? Reply with only the number.", "5"),
        ("reasoning", "Lina arrived before Omar. Who arrived first? Reply with only the name.", "Lina"),
        ("code", "What does a Python for-loop do?", None),
        ("code", "Why must a program avoid dividing by zero?", None),
        ("termination", "Say goodbye in one short sentence.", None),
        ("termination", "Answer yes or no: Is water a liquid at room temperature? No explanation.", "yes"),
    ]
    rows = []
    for i, (category, question, expected) in enumerate(pairs):
        prompt = format_chat_prompt([ChatMessage("user", question)], tokenizer)
        rows.append(dict(id=f"conversation-{i:02}", category=category, form="chat", prompt=prompt,
            question=question, input_ids=tokenizer.encode(prompt, add_special_tokens=False).ids,
            expected=expected, human_scores=None, provenance="new development diagnostic; not held-out generalization proof"))
    return rows


def prepare(directory):
    import numpy as np
    from llm_tokenizer.tokenizer import load_tokenizer
    job = read(RUN / "config/job.json")
    spec = job["spec"]
    manifest = dict(format="codexa-post-training-diagnostic-v1", run=str(RUN),
        directory=str(directory), phase_timeout_seconds=1800, gpu_total_budget_seconds=5400,
        command_line=sys.argv, files={}, code={}, git={}, system={}, limitations=[])
    paths = [RUN / "config/job.json", RUN / "metrics/report.json", RUN / "supervisor/completion.json",
        RUN / "checkpoints/slot-0/checkpoint.pt", Path(spec["initialization"]["path"]),
        Path(spec["tokenizer"]["path"]), Path(spec["data"]["train_window_order"]),
        Path(spec["data"]["validation_window_ids"]), Path(spec["data"]["validation_sources"])]
    for path in paths:
        manifest["files"][str(path)] = dict(sha256=digest(path), bytes=path.stat().st_size)
    for name, expected in job["code_identity"].items():
        path = ROOT / name if name.startswith("src/") else Path(importlib.util.find_spec(name).origin)
        actual = digest(path)
        manifest["code"][name] = dict(path=str(path), sha256=actual, frozen_expected=expected, matches=actual==expected)
    for name in ("llm_inference.generate", "llm_tokenizer.tokenizer", "llm_tokenizer.sft"):
        path = Path(importlib.util.find_spec(name).origin)
        repo = {"llm_inference": "LLM-Inference", "llm_tokenizer": "LLM-Tokenizer"}[name.split(".")[0]]
        source = ROOT.parent / repo / "src" / Path(*name.split(".")).with_suffix(".py")
        manifest["code"][name] = dict(path=str(path), sha256=digest(path),
            source_path=str(source), source_sha256=digest(source), installed_matches_source=digest(path)==digest(source),
            historical_frozen_hash_available=False)
    for repo in sorted(ROOT.parent.glob("LLM-*")):
        if (repo/".git").exists():
            manifest["git"][repo.name] = command(["git", "-C", str(repo), "status", "--short"])
    manifest["system"]["ram"] = command(["free", "-b"])
    manifest["system"]["gpu"] = command(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free", "--format=csv"])
    manifest["system"]["gpu_processes"] = command(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"])
    expected_final = read(RUN/"supervisor/completion.json")["final_checkpoint"]["sha256"]
    assert manifest["files"][str(RUN/"checkpoints/slot-0/checkpoint.pt")]["sha256"] == expected_final
    assert manifest["files"][spec["initialization"]["path"]]["sha256"] == spec["initialization"]["sha256"]
    assert manifest["files"][spec["tokenizer"]["path"]]["sha256"] == spec["tokenizer"]["sha256"]
    assert all(v["matches"] for v in manifest["code"].values() if "matches" in v)
    tokenizer = load_tokenizer(spec["tokenizer"]["path"])
    write(directory/"tokenizer.json", dict(vocab_size=tokenizer.get_vocab_size(),
        special_tokens={s:tokenizer.token_to_id(s) for s in ("<pad>","<bos>","<eos>","<unk>","<|system|>","<|user|>","<|assistant|>","<|end|>")},
        automatic_bos_eos=tokenizer.encode("Hello", add_special_tokens=True).ids != tokenizer.encode("Hello", add_special_tokens=False).ids))
    from codexa_workspace.foundation_diagnosis import make_cases
    constructed = {x["id"]:x for x in make_cases(tokenizer)}
    historical = {}
    for label, suffix in (("baseline", "baseline-step-00000"), ("final", "final-step-01792")):
        path = RUN / "evaluations" / ("generation-"+suffix) / "outputs.jsonl"
        manifest["files"][str(path)] = dict(sha256=digest(path),bytes=path.stat().st_size)
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        assert len(rows)==48
        for row in rows:
            assert constructed[row["id"]]["input_ids"] == row["input_ids"]
            append(directory/f"historical-{label}.jsonl", row)
        historical[label] = dict(cases=len(rows), exact=sum(x["metrics"]["exact_surface_correct"] for x in rows),
            severe=sum(x["metrics"]["repeated_ngram_severe"] for x in rows),
            length=sum(x["termination"]=="length" for x in rows),
            expected_preferred=sum(x["likelihood"]["expected"]["mean_token_nll"]<x["likelihood"]["distractor"]["mean_token_nll"] for x in rows))
    write(directory/"historical-summary.json", historical)
    write(directory/"conversational-cases.json", conversational_cases(tokenizer))
    order = np.load(spec["data"]["train_window_order"])
    validation = np.load(spec["data"]["validation_window_ids"])
    samples = np.sort(np.random.default_rng(42).choice(order,256,replace=False))
    corpus = np.memmap(spec["data"]["corpus_path"], dtype=spec["data"]["dtype"], mode="r")
    sampled = []
    for window in samples:
        start=int(window)*2048
        ids=corpus[start:start+2049].astype(int).tolist()
        text=tokenizer.decode(ids[:-1],skip_special_tokens=False)
        row=dict(window_id=int(window),input_ids=ids[:-1],target_ids=ids[1:],target_mask="all 2048 targets supervised",
            text=text, eos_targets=ids[1:].count(2), roles=sum(ids.count(t) for t in (4,5,6,7)),
            unknown=ids.count(3), padding=ids.count(0), max_id=max(ids),
            roundtrip=tokenizer.encode(text,add_special_tokens=False).ids==ids[:-1],
            metrics=mechanical(ids[:-1],text),documents=[])
        sampled.append(row)
    asset=Path(spec["data"]["corpus_path"]).parents[3]
    inventory={}
    for relative in ("tokenized/base-v1/mixed/token_data_manifest.json", "processed/general-chat-sft-v2/dataset_manifest.json",
        "processed/chat-sft-repair-v1/dataset_manifest.json", "raw/ultrachat/download_manifest.json", "raw/oasst1/download_manifest.json"):
        path=asset/relative
        inventory[relative]=dict(path=str(path),available=path.exists(),manifest=read(path) if path.exists() else None)
    # Stream the 3.1 GiB line-delimited JSON array; never load the complete index.
    index=Path(spec["data"]["corpus_path"]).with_name("train_index.json")
    sorted_order=sorted(int(x) for x in order)
    pointer=0
    source_targets=Counter()
    boundary_count=bad_eos=0
    records=0
    index_hash=hashlib.sha256()
    sample_pointer=0
    started=time.monotonic()
    with index.open("rb") as stream:
        for line in stream:
            index_hash.update(line)
            stripped=line.strip().rstrip(b",")
            if not stripped.startswith(b"{"):continue
            doc=json.loads(stripped);records+=1
            left,right=doc["token_start"],doc["token_end"]
            while pointer<len(sorted_order) and sorted_order[pointer]*2048+2049<=left: pointer+=1
            scan=pointer
            while scan<len(sorted_order) and sorted_order[scan]*2048+1<right:
                lo=max(left,sorted_order[scan]*2048+1);hi=min(right,sorted_order[scan]*2048+2049)
                if hi>lo:source_targets[doc["source"]]+=hi-lo
                scan+=1
            while sample_pointer<len(samples) and int(samples[sample_pointer])*2048+2049<=left:sample_pointer+=1
            scan=sample_pointer
            while scan<len(samples) and int(samples[scan])*2048<right:
                start=int(samples[scan])*2048
                if min(right,start+2049)>max(left,start):
                    sampled[scan]["documents"].append(doc)
                    boundary_count+=1
                    if int(corpus[right-1])!=2:bad_eos+=1
                scan+=1
            if records%1000000==0:print(f"index records {records:,}; {time.monotonic()-started:.0f}s",flush=True)
    for row in sampled:append(directory/"training-samples.jsonl",row)
    normalize=lambda s:' '.join(re.findall(r'\w+',s.casefold()))
    normalized=[normalize(x["text"]) for x in sampled]
    exact_duplicate_pairs=sum(n*(n-1)//2 for n in Counter(normalized).values())
    shingles=[set(zip(*(s.split()[i:] for i in range(5)))) for s in normalized]
    near=[]
    for i in range(len(shingles)):
        for k in range(i):
            union=shingles[i]|shingles[k]
            similarity=len(shingles[i]&shingles[k])/len(union) if union else 0
            if similarity>=.8:near.append(dict(left=sampled[k]["window_id"],right=sampled[i]["window_id"],jaccard=similarity))
    counts=Counter(t for x in sampled for t in x["target_ids"] if t<8)
    summary=dict(train_windows=len(order),unique_train_windows=len(set(order.tolist())),validation_windows=len(validation),
        train_validation_window_overlap=len(np.intersect1d(order,validation)),
        source_target_counts=dict(source_targets),total_targets=sum(source_targets.values()),
        streamed_index_records=records,streamed_index_sha256=index_hash.hexdigest(),
        sample_count=len(sampled),sampling="seed 42 uniform without replacement over actual selected windows; source composition measured afterward",
        sample_source_document_counts=dict(Counter(d["source"] for x in sampled for d in x["documents"])),
        sample_target_special_counts=dict(counts),sample_document_boundaries_checked=boundary_count,bad_eos=bad_eos,
        sample_roundtrip_failures=sum(not x["roundtrip"] for x in sampled),
        sample_severe=sum(x["metrics"]["independent_severe"] for x in sampled),
        normalized_exact_duplicate_pairs=exact_duplicate_pairs,near_duplicate_pairs=near,inventory=inventory,
        corpus_full_payload_hash_refreshed=False,full_corpus_checksum=inventory["tokenized/base-v1/mixed/token_data_manifest.json"]["manifest"]["output_checksums"]["train"])
    assert summary["total_targets"]==len(order)*2048
    assert summary["streamed_index_sha256"]==inventory["tokenized/base-v1/mixed/token_data_manifest.json"]["manifest"]["output_checksums"]["index"]
    write(directory/"dataset.json",summary)
    write(directory/"manifest.json",manifest)
    supplemental_preparation(directory,spec,tokenizer,corpus,order,sampled)
    print("Preparation complete",flush=True)


def supplemental_preparation(directory,spec,tokenizer,corpus,order,samples):
    """Bounded supplementary evidence, included in fresh-directory reproduction."""
    import numpy as np
    from llm_data.token_data import MemmapTokenDataset
    from llm_tokenizer.sft import ChatMessage,serialize_conversation
    dataset=MemmapTokenDataset(spec['data']['corpus_path'],dtype='uint16',context_length=2048)
    checks=[]
    for sample in samples[:8]:
        inputs,labels=dataset[sample['window_id']]
        checks.append(dict(window=sample['window_id'],inputs_match=inputs.tolist()==sample['input_ids'],
            labels_match=labels.tolist()==sample['target_ids'],masked_targets=int((labels==-100).sum()),eos_targets=int((labels==2).sum())))
    write(directory/'actual-dataset-shift-tests.json',checks)
    seen=set();counts=Counter()
    for sample in samples:
        for document in sample['documents']:
            source=document['source'];key=document['document_ordinal']
            if key in seen or counts[source]>=16:continue
            seen.add(key);counts[source]+=1
            start,end=document['token_start'],document['token_end']
            ids=corpus[start:min(end,start+8192)].astype(int).tolist()
            append(directory/'document-samples.jsonl',dict(index_record=document,raw_decoded_document=tokenizer.decode(ids,skip_special_tokens=False),
                excerpt_truncated=end-start>8192,tokens=len(ids)))
    asset=Path(spec['data']['corpus_path']).parents[3];audit={}
    for name,context in (('general-chat-sft-v2',2048),('chat-sft-repair-v1',512)):
        path=asset/'processed'/name/'train.jsonl';results=[]
        if path.exists():
            with path.open() as stream:
                for i in range(64):
                    line=stream.readline()
                    if not line:break
                    row=json.loads(line)
                    try:
                        serialized=serialize_conversation([ChatMessage(m['role'],m['content']) for m in row['messages']],tokenizer,maximum_tokens=context)
                        labels=list(serialized.labels)
                        results.append(dict(ok=True,tokens=len(serialized.input_ids),supervised=serialized.supervised_token_count,
                            supervised_END=labels.count(7),masked=labels.count(-100)))
                        if i<2:append(directory/'conversation-data-examples.jsonl',dict(dataset=name,messages=row['messages'],input_ids=list(serialized.input_ids),labels=labels))
                    except (KeyError,TypeError,ValueError) as error:results.append(dict(ok=False,error=repr(error)))
        audit[name]=dict(path=str(path),checked=len(results),successful=sum(x['ok'] for x in results),results=results)
    write(directory/'conversation-data-audit.json',audit)
    counts=Counter();selected_hash=hashlib.sha256();max_id=0;started=time.monotonic()
    for window in order:
        labels=corpus[int(window)*2048+1:int(window)*2048+2049]
        selected_hash.update(labels.tobytes());max_id=max(max_id,int(labels.max()))
        counts.update(labels[labels<8].astype(int).tolist())
    write(directory/'full-selected-target-audit.json',dict(targets=len(order)*2048,special_token_counts=dict(counts),maximum_id=max_id,
        selected_target_payload_sha256=selected_hash.hexdigest(),ordering='frozen train_window_order; little/native-endian uint16 target slices',
        scan_seconds=time.monotonic()-started,complete_selected_training_payload_scan=True))
    prepared=ROOT/'outputs/readiness/codexa-934m-foundational-pretraining-v1/prepared'
    selection=read(prepared/'selection-manifest.json');documents=read(prepared/'validation-documents.json')
    starts=order.astype(np.int64)*2048;stops=starts+2049;violations=[]
    for left,right in selection['excluded_intervals']:
        overlap=order[(starts<right)&(stops>left)]
        if len(overlap):violations.append(dict(interval=[left,right],window_ids=overlap.astype(int).tolist()))
    document_checks=[dict(document_id=doc['document_id'],token_bytes_match=hashlib.sha256(corpus[doc['token_start']:doc['token_end']].tobytes()).hexdigest()==doc['token_bytes_sha256'],
        boundary_EOS=int(corpus[doc['token_end']-1])==2) for doc in documents]
    stat=Path(spec['data']['corpus_path']).stat()
    identity=dict(device=stat.st_dev,inode=stat.st_ino,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,ctime_ns=stat.st_ctime_ns)
    write(directory/'validation-split-audit.json',dict(excluded_intervals=len(selection['excluded_intervals']),overlap_violations=violations,
        validation_document_count=len(documents),document_checks=document_checks,corpus_stat_identity=identity,
        matches_frozen_corpus_stat_identity=identity==spec['data']['corpus_input_identity'],semantic_duplicate_isolation_proven=False))
    old=asset.parent/'logs/codexa-900m-base-v1/run_metadata.json'
    if old.exists():write(directory/'baseline-original-run-metadata.json',read(old))
    training_analysis(directory)


def training_analysis(directory):
    rows=[json.loads(s) for s in (RUN/'metrics/train_metrics.jsonl').read_text().splitlines()]
    summary=dict(updates=len(rows),unique_steps=len({x['optimizer_step'] for x in rows}),
        missing_steps=sorted(set(range(1,1793))-{x['optimizer_step'] for x in rows}),
        nonfinite_rows=sum(not all(math.isfinite(x[k]) for k in ('training_loss','gradient_norm','learning_rate')) for x in rows),
        sum_step_seconds=sum(x['step_time_seconds'] for x in rows),sum_forward_backward_seconds=sum(x['forward_backward_seconds'] for x in rows),
        tokens_processed=sum(x['tokens_processed'] for x in rows),
        first_64_mean_loss=statistics.mean(x['training_loss'] for x in rows[:64]),last_64_mean_loss=statistics.mean(x['training_loss'] for x in rows[-64:]))
    for key in ('training_loss','gradient_norm','learning_rate','peak_allocated_vram_bytes','peak_reserved_vram_bytes','reserved_vram_bytes','tokens_per_second'):
        values=[x[key] for x in rows];summary[key]=dict(minimum=min(values),maximum=max(values),median=statistics.median(values))
    summary['segments']=[]
    for path in sorted((RUN/'supervisor/segments').glob('*/trainer-report.json')):
        record=read(path)
        summary['segments'].append(dict(path=str(path),**{key:record.get(key) for key in ('state','stop_reason','training_state','exact_resume','configuration_sha256','budget')}))
    for key,path in [('budget_extensions','config/budget-extensions.jsonl'),('events','supervisor/events.jsonl'),
        ('launch_gate_history','config/launch-gates-history.jsonl'),('resume_gate_history','config/resume-gates-history.jsonl')]:
        summary[key]=[json.loads(s) for s in (RUN/path).read_text().splitlines()]
    write(directory/'training-analysis.json',summary)


def resource_guard():
    import torch
    if torch.cuda.is_available():
        free,total=torch.cuda.mem_get_info()
        if free<2*1024**3:raise RuntimeError("Diagnostic GPU headroom below 2 GiB")
        if torch.cuda.memory_allocated()>8*1024**3:raise RuntimeError("Diagnostic CUDA allocation above 8 GiB")


def distribution(logits, tokenizer):
    import torch
    logits=logits.float()
    probabilities=logits.softmax(-1)
    ids=logits.topk(5).indices.tolist()
    return dict(eos_probability=probabilities[2].item(),eos_rank=int((logits>logits[2]).sum())+1,
        end_probability=probabilities[7].item(),end_rank=int((logits>logits[7]).sum())+1,
        entropy=-(probabilities*probabilities.clamp_min(1e-30).log()).sum().item(),
        top=[dict(id=t,text=tokenizer.decode([t],skip_special_tokens=False),probability=probabilities[t].item()) for t in ids])


def model_phase(directory,label):
    import numpy as np
    import torch
    from llm_architecture.model import LanguageModel,ModelConfig,count_parameters
    from llm_inference.generate import GenerationConfig,generate_sequences
    from llm_tokenizer.tokenizer import load_tokenizer
    from codexa_workspace.foundation_diagnosis import likelihood_probe
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    job=read(RUN/"config/job.json");spec=job["spec"]
    path=Path(spec["initialization"]["path"]) if label=="baseline" else RUN/"checkpoints/slot-0/checkpoint.pt"
    payload=torch.load(path,map_location="cpu",weights_only=False,mmap=True)
    finite={k:bool(torch.isfinite(v).all()) for k,v in payload["model_state_dict"].items()}
    assert all(finite.values())
    config=payload["config"]["model"]
    assert config==spec["model"]
    tokenizer=load_tokenizer(spec["tokenizer"]["path"])
    assert tokenizer.get_vocab_size()==config["vocab_size"]
    model=LanguageModel(ModelConfig(**config))
    model.load_state_dict(payload["model_state_dict"],strict=True)
    assert model.lm_head.weight is model.token_embeddings.weight
    model.requires_grad_(False);model.eval()
    assert count_parameters(model)==spec["expected_parameters"]
    metadata=dict(path=str(path),model_tensor_count=len(finite),all_finite=all(finite.values()),
        config=payload["config"],training_state=payload["training_state"],
        scheduler_state=payload.get("scheduler_state"),resume_context=payload.get("resume_context"),
        optimizer_present=bool(payload.get("optimizer_state_dict")),rng_present=bool(payload.get("random_states")),
        optimizer_parameter_steps=sorted({int(v["step"]) for v in payload.get("optimizer_state_dict",{}).get("state",{}).values() if "step" in v}),
        tokenizer_sha256=payload.get("tokenizer_sha256"),parameter_count=count_parameters(model),tied_embeddings=True)
    write(directory/f"checkpoint-{label}.json",metadata)
    del payload;gc.collect()
    device="cuda" if torch.cuda.is_available() else "cpu"
    if device=="cuda":torch.cuda.set_per_process_memory_fraction(.5)
    resource_guard()
    model.to(device)
    def finite_logits(_model,_inputs,result):
        if not torch.isfinite(result[0][:,-1]).all():
            raise FloatingPointError("Nonfinite diagnostic next-token logits")
    model.register_forward_hook(finite_logits)
    def context():return torch.autocast("cuda",dtype=torch.bfloat16) if device=="cuda" else nullcontext()
    metadata["runtime"]=dict(device=device,weights_dtype=str(next(model.parameters()).dtype),autocast="bf16" if device=="cuda" else None,
        torch_version=torch.__version__,cuda_version=torch.version.cuda,model_eval=not model.training,context_length=2048,
        attention="SDPA; causal prefill, explicit offset causal cached decode",eos=2,pad=0,stop_sequences=[[7]],attention_mask="None; all prompt tokens real")
    historical=[json.loads(s) for s in (directory/f"historical-{label}.jsonl").read_text().splitlines()]
    configs={"greedy":GenerationConfig(max_new_tokens=128,seed=42),
        "sample-t1":GenerationConfig(max_new_tokens=128,seed=42,do_sample=True,temperature=1),
        "sample-t08":GenerationConfig(max_new_tokens=128,seed=42,do_sample=True,temperature=.8),
        "nucleus":GenerationConfig(max_new_tokens=128,seed=42,do_sample=True,temperature=.8,top_p=.9),
        "topk":GenerationConfig(max_new_tokens=128,seed=42,do_sample=True,temperature=.8,top_k=40),
        "nucleus-r11":GenerationConfig(max_new_tokens=128,seed=42,do_sample=True,temperature=.8,top_p=.9,repetition_penalty=1.1)}
    tasks=[("historical",x,cfg) for cfg in (configs if label=="final" else ("greedy",)) for x in historical]
    tasks += [("conversation",x,cfg) for cfg in ("greedy","nucleus") for x in read(directory/"conversational-cases.json")]
    completed=set()
    generations_path=directory/f"generations-{label}.jsonl"
    if generations_path.exists():
        for line in generations_path.read_text().splitlines():
            row=json.loads(line)
            assert row["configuration"]==asdict(configs[row["decode"]])
            completed.add((row["suite"],row["id"],row["decode"]))
    started=time.monotonic()
    with torch.inference_mode(),context():
        for number,(suite,case,decode) in enumerate(tasks):
            if (suite,case["id"],decode) in completed:continue
            resource_guard()
            inputs=torch.tensor([case["input_ids"]],device=device)
            t=time.monotonic()
            result=generate_sequences(model,inputs,eos_token_id=2,pad_token_id=0,stop_sequences=[[7]],config=configs[decode]).sequences[0]
            ids=list(result.token_ids);visible=list(result.visible_token_ids)
            display=tokenizer.decode(visible,skip_special_tokens=True).strip()
            row=dict(checkpoint=label,suite=suite,id=case["id"],category=case["category"],form=case["form"],
                prompt=case["prompt"],input_ids=case["input_ids"],expected=case.get("expected"),decode=decode,
                configuration=asdict(configs[decode]),raw_token_ids=ids,visible_token_ids=visible,
                raw_text=tokenizer.decode(ids,skip_special_tokens=False),response=display,
                generated_tokens=len(ids),finish_reason=result.finish_reason,termination=result.termination_cause,
                terminator=result.terminating_token_id,seconds=time.monotonic()-t,
                metrics=mechanical(visible,display),human_scores={k:None for k in ("coherence","relevance","instruction_following","repetition","termination","usefulness")})
            if case.get("expected") is not None:
                row["exact_surface_match"]=display==case["expected"]
            if suite=="historical" and decode=="greedy":
                row["historical_token_match"]=ids==case["raw_token_ids"]
                row["historical_text_match"]=display==case["response"]
                row["likelihood"]=likelihood_probe(model,tokenizer,case,device=device)
            append(directory/f"generations-{label}.jsonl",row)
            if number%32==0:print(f"{label}: {number+1}/{len(tasks)} generations; {time.monotonic()-started:.0f}s",flush=True)
    # 256 authentic held-out windows, exact source-weighted validation.
    validation=np.load(spec["data"]["validation_window_ids"])
    sources=validation_source_list(read(spec["data"]["validation_sources"]),validation)
    corpus=np.memmap(spec["data"]["corpus_path"],dtype="uint16",mode="r")
    totals=Counter();counts=Counter();eos_nll=[];end_distributions=[]
    with torch.inference_mode(),context():
        for i,window in enumerate(validation):
            resource_guard()
            start=int(window)*2048
            ids=torch.tensor(corpus[start:start+2049].astype(np.int64),device=device)
            logits,_=model(ids[:-1].unsqueeze(0))
            losses=torch.nn.functional.cross_entropy(logits.float().transpose(1,2),ids[1:].unsqueeze(0),reduction="none")[0]
            totals[sources[i]]+=losses.sum().item();counts[sources[i]]+=len(losses)
            indices=torch.nonzero(ids[1:]==2).flatten()
            for idx in indices[:8]:
                eos_nll.append(losses[idx].item())
                if len(end_distributions)<16:end_distributions.append(dict(window_id=int(window),target_position=int(idx),distribution=distribution(logits[0,idx],tokenizer)))
        source_losses={s:totals[s]/counts[s] for s in totals}
        write(directory/f"validation-{label}.json",dict(sources=source_losses,tokens=sum(counts.values()),
            weighted_loss=sum(source_losses[s]*w for s,w in spec["data"]["source_loss_weights"].items()),
            eos_nll_count=len(eos_nll),mean_eos_nll=sum(eos_nll)/len(eos_nll) if eos_nll else None,
            document_end_distributions=end_distributions))
        if not (directory/f"document-eos-probes-{label}.jsonl").exists():
            documents=[json.loads(s) for s in (directory/"document-samples.jsonl").read_text().splitlines()]
            for document in documents:
                record=document["index_record"]
                start,end=record["token_start"],record["token_end"]
                tail=corpus[max(start,end-257):end-1].astype(int).tolist()
                if not tail:continue
                logits=model(torch.tensor([tail],device=device))[0][0,-1]
                append(directory/f"document-eos-probes-{label}.jsonl",dict(source=record["source"],document_ordinal=record["document_ordinal"],
                    context_tokens=len(tail),tail_text=tokenizer.decode(tail,skip_special_tokens=False),
                    actual_target=int(corpus[end-1]),distribution=distribution(logits,tokenizer)))
    if label=="final" and not (directory/"cache-equivalence.json").exists():
        cache_checks(model,tokenizer,directory,context)
    if label=="final" and not (directory/"token-traces.jsonl").exists():
        token_traces(model,tokenizer,directory,historical,context,corpus,validation)
    if label=="final" and not (directory/"free-generation-cache-comparison.jsonl").exists():
        extra_checks(model,tokenizer,directory,historical,context)
    if label=="final" and not (directory/"cache-boundary-fp32-followup.json").exists():
        from torch.nn.attention import SDPBackend,sdpa_kernel
        rows=[]
        base=tokenizer.encode("A computer stores information and follows instructions. ",add_special_tokens=False).ids
        for prefix,chunk in ((2040,4),(2047,1)):
            ids=torch.tensor([(base*((prefix+chunk)//len(base)+1))[:prefix+chunk]],device=device)
            with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
                full=model(ids)[0][:,-chunk:]
                cache=model(ids[:,:prefix],use_cache=True)[2]
                inc=model(ids[:,prefix:],past_key_values=cache,use_cache=True)[0]
                rows.append(dict(prefix=prefix,chunk=chunk,precision="fp32",backend="math",**comparison(full,inc)))
            del full,cache,inc
            resource_guard()
        write(directory/"cache-boundary-fp32-followup.json",rows)
    if label=="final" and not (directory/"precision-localization.json").exists():
        precision_localization(model,tokenizer,directory,historical,context)
    torch.cuda.synchronize() if device=="cuda" else None
    metadata["runtime"]["peak_allocated_bytes"]=torch.cuda.max_memory_allocated() if device=="cuda" else None
    metadata["runtime"]["peak_reserved_bytes"]=torch.cuda.max_memory_reserved() if device=="cuda" else None
    metadata["runtime"]["phase_seconds"]=time.monotonic()-started
    write(directory/f"checkpoint-{label}.json",metadata)
    print(f"{label} complete in {time.monotonic()-started:.1f}s",flush=True)


def cache_checks(model,tokenizer,directory,context):
    import torch
    from torch.nn.attention import SDPBackend,sdpa_kernel
    device=next(model.parameters()).device
    base=tokenizer.encode("A computer stores information and follows instructions. ",add_special_tokens=False).ids
    rows=[]
    def check(n,chunk,precision):
        ids=torch.tensor([(base*((n+chunk)//len(base)+1))[:n+chunk]],device=device)
        ctx=context() if precision=="bf16" else nullcontext()
        backend=sdpa_kernel(SDPBackend.MATH) if precision=="fp32" else nullcontext()
        with torch.inference_mode(),ctx,backend:
            full=model(ids)[0][:,-chunk:]
            pre=model(ids[:,:n],use_cache=True)[2]
            inc=model(ids[:,n:],past_key_values=pre,use_cache=True)[0]
            row=dict(prefix=n,chunk=chunk,precision=precision,**comparison(full,inc))
            rows.append(row)
        del full,pre,inc
        resource_guard()
    for n in (1,8,32,128,2040,2047):
        for chunk in (1,4):
            if n+chunk<=2048:check(n,chunk,"bf16")
    for n in (1,8,32,128):check(n,1,"fp32")
    with torch.inference_mode():
        ids=torch.tensor([base[:8]],device=device)
        pre=model(ids,use_cache=True)[2]
        continuation=torch.tensor([[base[8]]],device=device)
        full=model(torch.cat((ids,continuation),1))[0][:,-1:]
        masked=model(continuation,attention_mask=torch.ones_like(continuation,dtype=torch.bool),past_key_values=pre,use_cache=True)[0]
        fixed=model(continuation,attention_mask=torch.ones_like(continuation,dtype=torch.bool),position_ids=torch.tensor([[8]],device=device),past_key_values=pre,use_cache=True)[0]
        write(directory/"masked-cache-isolated-correction.json",dict(original=comparison(full,masked),explicit_offset=comparison(full,fixed),
            production_generator_supplies_mask=False,correction="Explicit position_ids=[[past_length]] in diagnostic call; no production edit"))
    # Compact tiny-model CPU tests cover rotary positions, padded masks, and loss.
    write(directory/"cache-equivalence.json",rows)


def token_traces(model,tokenizer,directory,historical,context,corpus,validation):
    import torch
    device=next(model.parameters()).device
    selected=[historical[i] for i in (0,1,8,9,12,13,36,37)]
    by_id={x["id"]:x for x in [json.loads(s) for s in (directory/"generations-final.jsonl").read_text().splitlines()] if x["decode"]=="greedy" and x["suite"]=="historical"}
    with torch.inference_mode(),context():
        for case in selected:
            ids=case["input_ids"];generated=by_id[case["id"]]["raw_token_ids"]
            all_ids=torch.tensor([[*ids,*generated]],device=device)
            logits=model(all_ids)[0][0]
            steps=[]
            for step,t in enumerate(generated):
                dist=distribution(logits[len(ids)-1+step],tokenizer)
                prefix=generated[:step]
                dist.update(step=step,selected_token=t,selected_token_probability=logits[len(ids)-1+step].float().softmax(-1)[t].item(),
                    selected_seen_in_generated_prefix=t in prefix,
                    prefix_word_repeat_4=mechanical(prefix,tokenizer.decode(prefix))["word_repeat_4"])
                steps.append(dist)
            expected=case["answer_token_ids"]["expected"]
            tf=model(torch.tensor([[*ids,*expected]],device=device))[0][0]
            teacher=[]
            for step,t in enumerate(expected):
                dist=distribution(tf[len(ids)-1+step],tokenizer)
                dist.update(target=t,target_probability=tf[len(ids)-1+step].float().softmax(-1)[t].item(),
                    argmax=int(tf[len(ids)-1+step].argmax()))
                teacher.append(dist)
            answer_end=distribution(tf[len(ids)+len(expected)-1],tokenizer)
            append(directory/"token-traces.jsonl",dict(id=case["id"],prompt=case["prompt"],free_steps=steps,
                teacher_expected=teacher,expected_answer_end=answer_end,
                first_expected_argmax_mismatch=next((i for i,x in enumerate(teacher) if x["target"]!=x["argmax"]),None),
                first_sustained_repetition=next((x["step"] for x in steps if x["step"]>=16 and x["prefix_word_repeat_4"]>.5),None)))
        for window in validation[:4]:
            start=int(window)*2048
            document=corpus[start:start+65].astype(int).tolist()
            logits=model(torch.tensor([document],device=device))[0][0]
            prompt=document[:16];free=[];steps=[]
            for i in range(32):
                out=model(torch.tensor([[*prompt,*free]],device=device))[0][0,-1]
                t=int(out.argmax());target=document[16+i]
                teacher=logits[15+i]
                steps.append(dict(step=i,target=target,free_argmax=t,teacher_argmax=int(teacher.argmax()),
                    target_teacher_probability=teacher.float().softmax(-1)[target].item(),
                    target_free_probability=out.float().softmax(-1)[target].item()))
                free.append(t)
            append(directory/"teacher-vs-free-corpus.jsonl",dict(window=int(window),prompt_ids=prompt,prompt=tokenizer.decode(prompt),
                true_ids=document[16:48],true_text=tokenizer.decode(document[16:48]),free_ids=free,free_text=tokenizer.decode(free),steps=steps,
                first_free_vs_teacher_argmax_divergence=next((x["step"] for x in steps if x["free_argmax"]!=x["teacher_argmax"]),None)))


def extra_checks(model,tokenizer,directory,historical,context):
    import torch
    from llm_inference.generate import GenerationConfig,generate_sequences
    device=next(model.parameters()).device
    greedy={x["id"]:x for x in [json.loads(s) for s in (directory/"generations-final.jsonl").read_text().splitlines()]
        if x["decode"]=="greedy" and x["suite"]=="historical"}
    with torch.inference_mode(),context():
        for index in (0,1,8,9,12,13,36,37):
            case=historical[index];inputs=torch.tensor([case["input_ids"]],device=device)
            full=generate_sequences(model,inputs,eos_token_id=2,pad_token_id=0,stop_sequences=[[7]],
                config=GenerationConfig(max_new_tokens=128,seed=42,use_kv_cache=False)).sequences[0]
            original=greedy[case["id"]]["raw_token_ids"];ids=list(full.token_ids)
            mismatch=next((i for i,(a,b) in enumerate(zip(original,ids)) if a!=b),None)
            row=dict(id=case["id"],cached_token_ids=original,full_token_ids=ids,
                full_text=tokenizer.decode(ids,skip_special_tokens=False),identical=original==ids,first_mismatch=mismatch)
            if mismatch is not None:
                prefix=torch.tensor([[*case["input_ids"],*original[:mismatch]]],device=device)
                direct=model(prefix)[0][:,-1]
                cache=model(prefix[:,:-1],use_cache=True)[2]
                inc=model(prefix[:,-1:],past_key_values=cache,use_cache=True)[0][:,-1]
                values=direct.topk(2).values
                row.update(bf16_same_prefix=comparison(direct,inc),direct_top_margin=float(values[0,0]-values[0,1]))
            append(directory/"free-generation-cache-comparison.jsonl",row)
        cases=historical[:2];length=max(len(x["input_ids"]) for x in cases)
        inputs=torch.tensor([[0]*(length-len(x["input_ids"]))+x["input_ids"] for x in cases],device=device)
        mask=inputs!=0
        batch=generate_sequences(model,inputs,attention_mask=mask,eos_token_id=2,pad_token_id=0,stop_sequences=[[7]],config=GenerationConfig(max_new_tokens=128)).sequences
        write(directory/"actual-padded-batch.json",[dict(id=x["id"],single_batch_identical=list(y.token_ids)==greedy[x["id"]]["raw_token_ids"])
            for x,y in zip(cases,batch)])
    # Same real model, no simultaneous second model or optimizer allocation.
    case=historical[0];inputs=torch.tensor([case["input_ids"]],device=device)
    with torch.inference_mode():
        gpu_logits=model(inputs)[0][:,-1].cpu()
        gpu_ids=generate_sequences(model,inputs,eos_token_id=2,pad_token_id=0,stop_sequences=[[7]],config=GenerationConfig(max_new_tokens=8)).sequences[0]
        model.to("cpu")
        cpu_logits=model(inputs.cpu())[0][:,-1]
        cpu_ids=generate_sequences(model,inputs.cpu(),eos_token_id=2,pad_token_id=0,stop_sequences=[[7]],config=GenerationConfig(max_new_tokens=8)).sequences[0]
        write(directory/"actual-cpu-gpu.json",dict(prompt=case["prompt"],both_fp32=comparison(gpu_logits,cpu_logits),
            gpu_token_ids=list(gpu_ids.token_ids),cpu_token_ids=list(cpu_ids.token_ids),
            identical=gpu_ids.token_ids==cpu_ids.token_ids,gpu_termination=gpu_ids.termination_cause,cpu_termination=cpu_ids.termination_cause))
        model.to(device)


def precision_localization(model,tokenizer,directory,historical,context):
    """Inspect actual accumulated caches at BF16 branch points and long-prefix error."""
    import torch
    from torch.nn.attention import SDPBackend,sdpa_kernel
    device=next(model.parameters()).device
    cases={x["id"]:x for x in historical}
    branches=[json.loads(s) for s in (directory/"free-generation-cache-comparison.jsonl").read_text().splitlines()]
    result={"branch_points":[]}
    for row in branches:
        mismatch=row["first_mismatch"]
        if mismatch is None:continue
        case=cases[row["id"]];original=row["cached_token_ids"]
        for precision in ("bf16","fp32"):
            ctx=context() if precision=="bf16" else nullcontext()
            backend=sdpa_kernel(SDPBackend.MATH) if precision=="fp32" else nullcontext()
            with torch.inference_mode(),ctx,backend:
                cache=None;next_input=torch.tensor([case["input_ids"]],device=device)
                for step in range(mismatch+1):
                    logits,_,cache=model(next_input,past_key_values=cache,use_cache=True)
                    next_input=torch.tensor([[original[step]]],device=device)
                inc=logits[:,-1]
                full=model(torch.tensor([[*case["input_ids"],*original[:mismatch]]],device=device))[0][:,-1]
                result["branch_points"].append(dict(id=row["id"],step=mismatch,precision=precision,
                    accumulated_cached_top=inc.topk(3).indices[0].tolist(),direct_top=full.topk(3).indices[0].tolist(),
                    cached_margin=float(inc.topk(2).values[0,0]-inc.topk(2).values[0,1]),
                    direct_margin=float(full.topk(2).values[0,0]-full.topk(2).values[0,1]),**comparison(full,inc)))
            del cache,logits,full,inc
    prefix,chunk=2040,4
    base=tokenizer.encode("A computer stores information and follows instructions. ",add_special_tokens=False).ids
    ids=torch.tensor([(base*((prefix+chunk)//len(base)+1))[:prefix+chunk]],device=device)
    snapshots={"full":{},"incremental":{}};positions={};phase="full"
    def capture(layer):
        def hook(_module,_inputs,output):
            hidden=output[0] if isinstance(output,tuple) else output
            snapshots[phase][layer]=hidden[0,-chunk:].float().cpu()
        return hook
    def capture_positions(_module,inputs):positions[phase]=inputs[0].flatten()[-chunk:].tolist()
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        cache=model(ids[:,:prefix],use_cache=True)[2]
        handles=[block.register_forward_hook(capture(i)) for i,block in enumerate(model.blocks)]
        handles.append(model.position_embeddings.register_forward_pre_hook(capture_positions))
        try:
            full=model(ids)[0][:,-chunk:]
            phase="incremental"
            inc=model(ids[:,prefix:],past_key_values=cache,use_cache=True)[0]
        finally:
            for handle in handles:handle.remove()
        result["long_prefix"]=dict(prefix=prefix,chunk=chunk,position_ids=positions,
            layer_max_absolute=[float((snapshots["full"][i]-snapshots["incremental"][i]).abs().max()) for i in range(len(model.blocks))],
            logit_comparison=comparison(full,inc),
            long_context_absolute_screen_pass=bool(torch.allclose(full,inc,atol=5e-4,rtol=1e-4)),
            interpretation="Report original strict screening failure too; layerwise accumulation and exact positions distinguish roundoff from offset defects.")
    write(directory/"precision-localization.json",result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("prepare","baseline","final"),required=True)
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--attempt",type=int,default=0,help="New diagnostic attempt record; completed generation rows are reused, never training state")
    args=parser.parse_args()
    directory=args.directory.resolve()
    allowed=(ROOT/"outputs/diagnostics/post-training-generation-investigation").resolve()
    if not directory.is_relative_to(allowed) or directory==allowed:parser.error("Use a dedicated child of outputs/diagnostics/post-training-generation-investigation")
    directory.mkdir(parents=True,exist_ok=True)
    phase_name=args.phase if args.attempt==0 else f"{args.phase}.attempt-{args.attempt}"
    marker=directory/f"{phase_name}.started.json"
    if marker.exists():parser.error("Phase already started; choose a fresh collision-safe directory or a different phase")
    write(marker,dict(started=time.time(),argv=sys.argv,script_sha256=digest(__file__)))
    def timeout(_signum,_frame):raise TimeoutError("Diagnostic phase exceeded 1800-second bound")
    seconds_used=0
    if args.phase!="prepare":
        for previous in directory.glob("*.started.json"):
            if not previous.name.startswith(("baseline.","final.")) or previous==marker:continue
            finished=previous.with_name(previous.name.replace(".started.json",".completion.json"))
            if finished.exists():seconds_used+=max(0,read(finished)["finished"]-read(previous)["started"])
        if seconds_used>=5400:raise TimeoutError("Total diagnostic accelerated phase budget exhausted")
    signal.signal(signal.SIGALRM,timeout);signal.alarm(max(1,min(1800,int(5400-seconds_used))))
    try:
        if args.phase=="prepare":prepare(directory)
        else:model_phase(directory,args.phase)
        write(directory/f"{phase_name}.completion.json",dict(status="passed",finished=time.time()))
    except BaseException as error:
        write(directory/f"{phase_name}.completion.json",dict(status="failed",error=repr(error),finished=time.time()))
        raise
    finally:signal.alarm(0)


if __name__=="__main__":main()
