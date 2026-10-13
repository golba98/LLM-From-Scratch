"""The integration package is layered: a module may import only from its own layer or lower ones.

    0  paths, hashing, sibling_commands          fixed locations and content identities
    1  assets                                    workspace, asset and output resolution
    2  evaluation                                metrics, suites, generation helpers, capability benchmark
    3  training: gates, environment, runs, identity, telemetry, supervisor (state/runner/systemd)
    4  notifications                             run-end popup, AI wakeup, reports, agent window
    5  monitoring, supervisor cli/ssd            dashboard, status, supervisor command line
    6  training.foundation, review, chat, cli    jobs and operator entry points

Imports inside functions count too, so a lazy import cannot hide an upward dependency.
"""
import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / 'src/codexa_workspace'
RANKS = {
    'codexa_workspace.paths': 0, 'codexa_workspace.hashing': 0, 'codexa_workspace.sibling_commands': 0,
    'codexa_workspace.assets': 1,
    'codexa_workspace.evaluation': 2,
    'codexa_workspace.training': 3,
    'codexa_workspace.notifications': 4,
    'codexa_workspace.monitoring': 5, 'codexa_workspace.training.supervisor.cli': 5,
    'codexa_workspace.training.supervisor.ssd': 5,
    'codexa_workspace.training.foundation': 6, 'codexa_workspace.review': 6, 'codexa_workspace.chat': 6,
    'codexa_workspace.cli': 6,
}


def rank(module: str) -> int | None:
    matches = [prefix for prefix in RANKS if module == prefix or module.startswith(prefix + '.')]
    return RANKS[max(matches, key=len)] if matches else None


def module_name(path: Path) -> str:
    parts = path.relative_to(PACKAGE.parent).with_suffix('').parts
    return '.'.join(parts[:-1] if parts[-1] == '__init__' else parts)


def imports(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('codexa_workspace'):
            found.add(node.module)
            found.update(f'{node.module}.{alias.name}' for alias in node.names)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names if alias.name.startswith('codexa_workspace'))
    return found


def test_every_module_has_a_layer():
    unranked = [module_name(p) for p in PACKAGE.rglob('*.py') if rank(module_name(p)) is None
                and module_name(p) != 'codexa_workspace']
    assert not unranked, f'add these modules to RANKS: {unranked}'


def test_no_module_imports_a_higher_layer():
    violations = []
    for path in PACKAGE.rglob('*.py'):
        source = module_name(path)
        for target in imports(path):
            own, other = rank(source), rank(target)
            if own is not None and other is not None and other > own:
                violations.append(f'{source} (layer {own}) imports {target} (layer {other})')
    assert not violations, '\n'.join(sorted(violations))


def test_no_relative_imports():
    relative = [str(p.relative_to(PACKAGE)) for p in PACKAGE.rglob('*.py')
                if any(isinstance(n, ast.ImportFrom) and n.level for n in ast.walk(ast.parse(p.read_text())))]
    assert not relative, f'use absolute codexa_workspace imports: {relative}'
