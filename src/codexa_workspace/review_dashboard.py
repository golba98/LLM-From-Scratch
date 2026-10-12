"""Publish a public-safe, single-question human review dashboard from an existing package.

The frozen ``qualification_review`` module is hash-bound by approved diagnostic plans, so the
improved interface lives here and consumes its unchanged ``package.json`` output.
"""

import hashlib
import html
import json
import re
from pathlib import Path

TEMPLATE = Path(__file__).with_name('review_dashboard.html')
PUBLIC_KEYS = ('format', 'kind', 'package_sha256', 'candidate_aliases', 'rubric', 'provenance', 'items')
DASHBOARD_VERSION = 'codexa-review-dashboard-v2'


def package_fingerprint(package: dict) -> str:
    """Recompute the identity exactly as ``write_review_page`` and ``import_decisions`` do."""
    unsigned = {k: v for k, v in package.items() if k != 'package_sha256'}
    return hashlib.sha256(json.dumps(unsigned, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def public_view(package: dict) -> dict:
    """Whitelist reviewer-facing fields; mapping and comparison hashes never leave the source."""
    return {key: package[key] for key in PUBLIC_KEYS if key in package}


def render_dashboard(package: dict, *, title: str = 'Codexa — Model Evaluation', label: str = '') -> str:
    """Embed the public view as inert JSON; model text is only ever inserted as text nodes."""
    encoded = (json.dumps(public_view(package), ensure_ascii=False)
               .replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
               .replace('\u2028', '\\u2028').replace('\u2029', '\\u2029'))
    template = TEMPLATE.read_text()
    if template.count('__PACKAGE_JSON__') != 1:
        raise ValueError('Dashboard template must contain exactly one package placeholder.')
    for key, value in (('__TITLE__', title), ('__LABEL__', label), ('__DASHBOARD_VERSION__', DASHBOARD_VERSION)):
        template = template.replace(key, html.escape(value))
    return template.replace('__PACKAGE_JSON__', encoded)


def secret_strings(mapping: Path) -> tuple[set[str], str]:
    """Read private identities and the mapping hash without ever writing them anywhere."""
    values = json.loads(mapping.read_text())
    names = set()
    for value in values.values() if isinstance(values, dict) else values:
        text = str(value)
        names.update(part for part in (text, Path(text).name, Path(text).stem) if part)
    return names, hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def structural_text(package: dict) -> str:
    """Public page content except model-generated and prompt text, where ordinary words may occur."""
    view = public_view(package)
    view['items'] = [{k: v for k, v in item.items() if k not in ('answer', 'instruction', 'findings')}
                     | {'candidate': (item.get('instruction') or {}).get('candidate')
                        if isinstance(item.get('instruction'), dict) else None}
                     for item in package.get('items', [])]
    return TEMPLATE.read_text() + json.dumps(view, ensure_ascii=False)


def leaked_identities(package: dict, page: str, mapping: Path | None) -> list[str]:
    """Hashes must be absent everywhere; checkpoint names must be absent from every structural field."""
    hashes = {package[key] for key in ('mapping_sha256', 'comparison_sha256') if package.get(key)}
    names = set()
    if mapping is not None:
        names, mapping_hash = secret_strings(mapping)
        hashes.add(mapping_hash)
    structure = structural_text(package)
    found = [value for value in hashes if value in page]
    found += [name for name in names if re.search(r'(?<![\w-])' + re.escape(name) + r'(?![\w-])', structure)]
    return found


def publish_public_review(source: Path, output: Path, *, mapping: Path | None = None,
                          title: str = 'Codexa — Model Evaluation') -> dict:
    """Write only ``index.html`` to a fresh directory outside the private source directory."""
    source, output = source.resolve(), output.resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError('The public review directory must be outside the private source directory.')
    if output.exists():
        raise FileExistsError('Use a fresh public review directory.')
    package = json.loads((source/'package.json').read_text())
    if package_fingerprint(package) != package.get('package_sha256'):
        raise ValueError('Review package was modified.')
    page = render_dashboard(package, title=title, label=source.name)
    leaked = leaked_identities(package, page, mapping)
    if leaked:
        raise ValueError(f'{len(leaked)} private identity string(s) would appear in the public page.')
    output.mkdir(parents=True)
    (output/'index.html').write_text(page)
    return dict(directory=str(output), page=str(output/'index.html'), package_sha256=package['package_sha256'],
                items=len(package['items']), bytes=len(page.encode()), dashboard=DASHBOARD_VERSION)
