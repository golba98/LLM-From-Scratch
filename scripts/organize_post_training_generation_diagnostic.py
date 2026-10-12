"""Organize measured diagnostics without model execution or evidence deletion."""
import hashlib
import json
from pathlib import Path
import re

REPORTS = ('DIAGNOSTIC_REPORT.md', 'GENERATION_COMPARISON.md', 'TOKENIZER_EOS_AUDIT.md', 'INFERENCE_AUDIT.md', 'DATASET_AUDIT.md', 'GPU_MEMORY_AUDIT.md', 'NEXT_EXPERIMENT.md')
GROUPS = {
    'generation': 'baseline-final-paired.json conversational-cases.json generation-metrics.csv generation-summary.csv generation-summary.json generations-baseline.jsonl generations-final.jsonl historical-baseline.jsonl historical-final.jsonl historical-summary.json human-review-blinded.jsonl human-review-candidates.jsonl human-review-key.json',
    'tokenizer': 'tokenizer.json document-eos-probes-baseline.jsonl document-eos-probes-final.jsonl token-traces.jsonl',
    'inference': 'actual-cpu-gpu.json actual-padded-batch.json cache-boundary-fp32-followup.json cache-equivalence.json free-generation-cache-comparison.jsonl masked-cache-isolated-correction.json precision-localization.json teacher-vs-free-corpus.jsonl',
    'data': 'actual-dataset-shift-tests.json conversation-data-audit.json conversation-data-examples.jsonl dataset.json document-samples.jsonl full-selected-target-audit.json training-samples.jsonl validation-split-audit.json',
    'training': 'baseline-original-run-metadata.json checkpoint-baseline.json checkpoint-final.json training-analysis.json training-metrics.csv validation-baseline.json validation-final.json',
    'verification': 'diagnostic-unit-tests-final.txt diagnostic-unit-tests.txt generation-regression-tests.txt git-status-final.json manifest.json preservation-final.json reproducibility-check.json root-causes.json testing.json verification-summary.json',
}
DESTINATIONS = {name: f'evidence/{group}/{name}' for group, names in GROUPS.items() for name in names.split()}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def destination(relative):
    name = relative.as_posix()
    if name in DESTINATIONS: return DESTINATIONS[name]
    if name in REPORTS or name == 'README.md': return f'archive/reports/{name}'
    if name in ('training-curves.png', 'training-curves.svg'): return f'figures/{name}'
    if relative.parts[0] == 'reproduction-check': return f'archive/{name}'
    if relative.parts[0] == '__pycache__' or relative.suffix == '.py': return f'archive/helpers/{name}'
    if relative.suffix == '.log' or name.endswith(('.started.json', '.completion.json')): return f'archive/logs/{name}'
    if 'initial-measurement' in name or 'followup-measurement' in name: return f'archive/measurements/{name}'
    return f'archive/supporting/{name}'


def organize(directory):
    inventory_path = directory / 'evidence/verification/organization-manifest.json'
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text())
    else:
        for name in REPORTS:
            if not (directory/name).is_file(): raise ValueError(f'Missing report: {name}')
        originals = sorted(p for p in directory.rglob('*') if p.is_file())
        inventory = dict(format='codexa-diagnostic-organization-v1', original_file_count=len(originals),
            files=[dict(original=p.relative_to(directory).as_posix(), destination=destination(p.relative_to(directory)), bytes=p.stat().st_size, sha256=digest(p)) for p in originals])
        targets = [x['destination'] for x in inventory['files']]
        if len(targets) != len(set(targets)): raise ValueError('Duplicate destinations')
        inventory_path.parent.mkdir(parents=True, exist_ok=True)
        inventory_path.write_text(json.dumps(inventory, indent=2)+'\n')
    for entry in inventory['files']:
        source, target = directory/entry['original'], directory/entry['destination']
        if target.exists():
            if digest(target) != entry['sha256']: raise ValueError(f'Changed evidence: {target}')
            if source.exists() and source != target: raise ValueError(f'Both paths exist: {source}')
            continue
        if not source.is_file() or digest(source) != entry['sha256']: raise ValueError(f'Missing or changed original: {source}')
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
    for path in sorted(directory.rglob('*'), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()): path.rmdir()
    return inventory


def sections(text):
    result = {}; title = None; lines = []; fence = False
    for line in text.splitlines():
        if line.startswith('```'): fence = not fence
        if not fence and line.startswith('## '):
            if title is not None: result[title] = '\n'.join(lines).strip()
            title = line[3:]; lines = []
        elif title is not None: lines.append(line)
    if title is not None: result[title] = '\n'.join(lines).strip()
    return result


def lower_headings(text):
    lines = []; fence = False
    for line in text.splitlines():
        if line.startswith('```'): fence = not fence
        if not fence and line.startswith('#'): line = '#'+line
        lines.append(line)
    return '\n'.join(lines)


def report_body(text):
    return lower_headings(text[text.index('\n## ')+1:])


def consolidate(directory, inventory):
    docs = {name: (directory/'archive/reports'/name).read_text() for name in REPORTS}
    diag, comparison = sections(docs[REPORTS[0]]), sections(docs[REPORTS[1]])
    navigation = [('verdict','1. Verdict and first intervention'), ('verified-run','2. Verified run and checkpoint'), ('generations','3. Actual generations and decoding comparisons'), ('root-causes','4. Root causes and confidence'), ('technical-findings','5. Technical findings'), ('checks','6. Testing and limitations'), ('next-experiment','7. Proposed next experiment — not authorized'), ('reproduction','8. Evidence and reproducibility'), ('review-appendix','Appendix: additional human-review candidates')]
    parts = ['# Codexa post-training generation investigation', 'One report for the completed investigation. Checkpoints, tokenizer, datasets, and training records were preserved. No model evaluation or training was performed during this organization. Human semantic ratings remain pending.', '## Contents\n\n'+'\n'.join(f'- [{title}](#{anchor})' for anchor,title in navigation)]
    def add(index, content):
        anchor,title = navigation[index]
        parts.append(f'<a id="{anchor}"></a>\n\n## {title}\n\n{content}')
    executive = diag['Executive verdict'].split('\n\nEvidence directory:', 1)[0]
    add(0, executive+'\n\n### Recommended first intervention\n\n'+diag['Recommended next action'])
    add(1, diag['Verified run and checkpoint']+'\n\n### Canonical project locations\n\n'+diag['Canonical ownership and locations'])
    blocks = re.split(r'(?m)^### (?=conversation-)', comparison['Conversational human-review candidates'])
    main_examples, extra_examples = [], []
    for block in blocks[1:]:
        (main_examples if block.startswith(('conversation-00','conversation-02','conversation-04','conversation-10')) else extra_examples).append('### '+block.strip())
    add(2, '### Measurements\n\n'+comparison['Matched measurements']+'\n\n### Controlled settings\n\n'+comparison['Controlled settings']+'\n\n### Representative conversations\n\n'+blocks[0].strip()+'\n\n'+'\n\n'.join(main_examples)+'\n\n<details>\n<summary>Historical baseline/final examples: four matched prompts</summary>\n\n'+comparison['Representative historical comparison']+'\n\n</details>')
    add(3, diag['Ranked causes and associated defects']+'\n\n### Possible and ruled-out explanations\n\n'+diag['Possible hypotheses and ruled-out explanations'])
    technical = []
    for name,title in [(REPORTS[2],'Tokenizer and termination'),(REPORTS[3],'Inference and KV cache'),(REPORTS[4],'Data and learning objective'),(REPORTS[5],'GPU memory and run stability')]:
        technical.append(f'<details>\n<summary>{title}</summary>\n\n'+report_body(docs[name])+'\n\n</details>')
    add(4, '### Architecture and training dynamics\n\n'+diag['Architecture, optimization and stability']+'\n\n![Training, validation, learning rate, and GPU memory across 1,792 updates](figures/training-curves.png)\n\n'+'\n\n'.join(technical))
    tests = json.loads((directory/DESTINATIONS['testing.json']).read_text())
    table = ['| Check | Recorded result |','|---|---|']
    for key,value in tests.items():
        if isinstance(value,list): value = '; '.join(value)
        table.append('| '+key.replace('_',' ')+' | '+str(value).replace('|',' / ')+' |')
    organization_tests = ' and [offline organization test results](evidence/verification/organization-tests.txt)' if (directory/'evidence/verification/organization-tests.txt').exists() else ''
    add(5, '\n'.join(table)+'\n\nNumerical screening failures and the masked-cache defect remain documented in the technical findings. No human quality scores are inferred from automated metrics.\n\nFolder organization checks are separate from the original model tests: see the [preservation and navigation checks](evidence/verification/organization-checks.json)'+organization_tests+'.\n\nAll critical originals'+diag['Reproduction and limitations'].split('All critical originals',1)[1])
    add(6, report_body(docs[REPORTS[6]]))
    index = ['### Folder guide\n\n| Folder | Contents |\n|---|---|', '| `evidence/generation/` | Full raw generations, decoding metrics, comparisons, and human-review candidates |','| `evidence/tokenizer/` | Token IDs, EOS probes, and token traces |','| `evidence/inference/` | Cache, precision, padding, device, and teacher-forcing checks |','| `evidence/data/` | Decoded samples, exposure accounting, and validation isolation |','| `evidence/training/` | Checkpoint metadata, training metrics, and run analysis |','| `evidence/verification/` | Hashes, tests, source identities, and relocation accounting |','| `figures/` | Training-curve PNG and SVG |','| `archive/` | Superseded reports, logs, previous measurements, helpers, and reproduction snapshots |']
    for group,names in GROUPS.items():
        index.append(f'\n<details>\n<summary>Evidence index: {group}</summary>\n')
        index.extend(f'- [{name}]({DESTINATIONS[name]})' for name in names.split() if (directory/DESTINATIONS[name]).exists())
        index.append('\n</details>')
    index.append('\n<details>\n<summary>Archive index: superseded material and recovery evidence</summary>\n')
    for entry in inventory['files']:
        if entry['destination'].startswith('archive/'):
            index.append(f"- [{entry['destination'][8:]}]({entry['destination']})")
    index.append('\n</details>')
    index.append(f"\n### Preservation and organization\n\nAll {inventory['original_file_count']} original files are retained byte-for-byte. The [relocation manifest](evidence/verification/organization-manifest.json) records every original path, organized path, size, and SHA-256 hash. Archived reports retain historical paths and are superseded by this report. [Organization checks](evidence/verification/organization-checks.json) verify file contents and report navigation.")
    relative = directory.relative_to(Path(__file__).resolve().parents[1])
    index.append(f'\n### Rebuild this report only\n\nRun from `LLM-From-Scratch`:\n\n```sh\n.venv/bin/python scripts/render_post_training_generation_diagnostic.py --directory "{relative}"\n```\n\nThis reads retained findings and rebuilds only `REPORT.md` and organization checks. It does not load model weights, execute evaluations, or start training.')
    index.append('\n### Reproduce measured diagnostics separately\n\n'+diag['Reproduction and limitations'].split('All critical originals',1)[0].strip())
    add(7,'\n\n'.join(index))
    add(8,'<details>\n<summary>Additional 12 prompts: baseline/final and greedy/nucleus excerpts</summary>\n\n'+'\n\n'.join(extra_examples)+'\n\n</details>')
    text = '\n\n'.join(parts)
    report_anchors = dict(zip(REPORTS, ('verdict','generations','technical-findings','technical-findings','technical-findings','technical-findings','next-experiment')))
    linked = []; fence = False
    for line in text.splitlines():
        if line.startswith('```'): fence = not fence
        if not fence:
            for name,anchor in report_anchors.items():
                line = re.sub(r'(?<![/\w])'+re.escape(name)+r'(?![\w])',f'[corresponding findings](#{anchor})',line)
            if '](' not in line:
                for name,target in DESTINATIONS.items():
                    line = re.sub(r'(?<![/\w`])'+re.escape(name)+r'(?![\w`])',f'[{name}]({target})',line)
        linked.append(line)
    (directory/'REPORT.md').write_text('\n'.join(linked).strip()+'\n')


def verify(directory, inventory):
    for entry in inventory['files']:
        if digest(directory/entry['destination']) != entry['sha256']: raise ValueError(f"Changed evidence: {entry['destination']}")
    text = (directory/'REPORT.md').read_text()
    anchors = set(re.findall(r'<a id="([^"]+)"',text)); checked = 0
    for link in re.findall(r'\]\(([^)]+)\)',text):
        if link.startswith('#'):
            if link[1:] not in anchors: raise ValueError(f'Broken anchor: {link}')
            checked += 1
        elif not link.startswith(('<','/','http')):
            if link == 'evidence/verification/organization-checks.json':
                checked += 1
                continue  # This verification result is written after successful checks.
            if not (directory/link).is_file(): raise ValueError(f'Broken link: {link}')
            checked += 1
    if text.count('```')%2: raise ValueError('Unbalanced code fences')
    original_comparison = (directory/'archive/reports/GENERATION_COMPARISON.md').read_text()
    raw_excerpts = re.findall(r'```text\n(.*?)\n```', original_comparison, re.S)
    if not all('```text\n'+excerpt+'\n```' in text for excerpt in raw_excerpts):
        raise ValueError('A raw generation excerpt was changed or omitted')
    source_links = re.findall(r'\]\(<([^>]+)>\)', text)
    for link in source_links:
        if not Path(re.sub(r':\d+$', '', link)).is_file():
            raise ValueError(f'Broken source link: {link}')
    for entry in inventory['files']:
        path = directory/entry['destination']
        if path.suffix == '.json': json.loads(path.read_text())
        elif path.suffix == '.jsonl':
            for line in path.read_text().splitlines():
                if line.strip(): json.loads(line)
        elif path.suffix == '.csv':
            import csv
            with path.open() as stream: list(csv.DictReader(stream))
    expected = ['REPORT.md','archive','evidence','figures']
    if sorted(p.name for p in directory.iterdir()) != expected: raise ValueError('Unexpected top-level clutter')
    result = dict(original_files_preserved=inventory['original_file_count'],all_original_hashes_match=True,report_links_checked=checked,structured_evidence_readable=True,code_fences_balanced=True,raw_generation_excerpts_preserved=len(raw_excerpts),source_file_links_checked=len(source_links),report_sha256=digest(directory/'REPORT.md'),top_level=expected,model_or_training_execution=False)
    (directory/'evidence/verification/organization-checks.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def render(directory):
    directory = Path(directory)
    inventory = organize(directory)
    consolidate(directory,inventory)
    return verify(directory,inventory)
