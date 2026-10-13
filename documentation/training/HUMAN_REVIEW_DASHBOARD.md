# Human review dashboard

A single-question review interface for Codexa human review packages. It is a public-safe view of an existing `package.json`. It does not regenerate responses, change package identities or touch the private alias mapping.

## Why it exists

The original page template (`PAGE` in `src/codexa_workspace/qualification_review.py`) has four problems:

- It renders every item at once and calls `JSON.stringify` on every non-string field, so prompts and the 26 automatic metrics appear as JSON blobs.
- It appends the five overall-judgment panels to the sticky header.
- It exports unchecked overall checkboxes as `false`.
- It has no autosave.

That module is hash-bound by the approved `job-v2/plan.json` `code_identity`, so it is left byte-identical. The new interface lives in:

- `src/codexa_workspace/review_dashboard.py`: the generator and blinding checks.
- `src/codexa_workspace/review_dashboard.html`: a vanilla HTML/CSS/JS template. It loads no external resources and has a restrictive CSP.
- `scripts/evaluation/review_dashboard.py`: the CLI.
- `tests/test_review_dashboard.py`: the tests.

## Fresh blinded baseline (934M)

Public page, which contains only `index.html`:
`outputs/diagnosis/codexa-934m-foundation-20261011-v1/review/fresh-blinded-baseline-public/index.html`

Serve it on localhost only. Never serve the private `fresh-blinded-baseline/` directory, because it contains `private-mapping.json`.

```sh
npx live-server --host=127.0.0.1 --port=8080 --no-browser \
  "outputs/diagnosis/codexa-934m-foundation-20261011-v1/review/fresh-blinded-baseline-public"
# then open http://127.0.0.1:8080/
```

Alternative: `python3 -m http.server 8080 --bind 127.0.0.1 -d <public dir>`. Opening `index.html` directly from disk also works.

Regenerate it into a new directory, because publishing refuses to overwrite:

```sh
.venv/bin/python scripts/evaluation/review_dashboard.py publish \
  --source outputs/diagnosis/codexa-934m-foundation-20261011-v1/review/fresh-blinded-baseline \
  --output <new public dir> \
  --mapping outputs/diagnosis/codexa-934m-foundation-20261011-v1/review/fresh-blinded-baseline/private-mapping.json \
  --title "Codexa 934M — Model Evaluation"
```

The publish step does the following:

- Recomputes `package_sha256` and refuses a modified package.
- Refuses an output directory inside the private source directory.
- Writes only `index.html`.
- Drops `mapping_sha256` and `comparison_sha256`. With five known candidate names, the mapping hash could be reversed by trying all 120 permutations.
- When `--mapping` is given, refuses if a checkpoint name appears in any structural field (ids, aliases, metadata, template). It never copies the mapping.

## Using it

- **Review tab:**
  - Shows one response at a time: earlier conversation turns, the user prompt, and the exact model response, rendered as text only.
  - Shows the stop reason.
  - The reference answer is hidden until you reveal it.
  - Machine diagnostics are collapsed by default and are labelled as not human judgment.
- **Sidebar:**
  - Filters for status, category and anonymous model, each with counts.
  - Search by text or ID.
  - "By question" order, which compares one prompt across screens.
  - Jump to a question number.
- **Model summary tab:** per-screen counts and coherence breakdown, aggregated automated checks, and separate three-way overall judgments (Not decided / Yes / No). Overall judgments are never derived from the counts.
- **Keys:**
  - `A`/`R`/`U`/`P` set the decision.
  - `2`/`1`/`0` set coherence.
  - `J`/`K` (or the arrow keys) move to the next or previous response.
  - `N` goes to the next unreviewed response; `Enter` saves and moves on.
  - `/` opens search, `D` toggles diagnostics, `V` toggles the reference, `?` shows help.
  - Shortcuts are ignored while you are typing in a text field.

## Export and import semantics (`codexa-human-review-export-v1`, unchanged)

| UI state | Exported as |
|---|---|
| Accept / Reject | `decision: "approve"` / `"reject"` |
| Pending | `decision: "pending"` |
| Unsure | `decision: "pending"`, `needs_review: true` |
| Overall Not decided | `null` |

- The existing importers (`import_decisions`, `reviewed_comparison`) ignore the additive `needs_review` and `extensions` keys. They refuse pending decisions and `null` overall judgments as incomplete, so nothing is treated as approved.
- A complete dashboard export passes `import_decisions` unchanged; the tests cover this.
- Files from the previous page (they have no `extensions.dashboard` key) still import. Their overall `false` values came from unticked checkboxes, so they import as Not decided, with a visible notice. `true` is kept as Yes.
- Import refuses invalid JSON, a different `package_sha256`, a wrong method, unknown ids or aliases, and invalid values. When it refuses, nothing changes.
- Replacing an existing draft asks for confirmation first and keeps a local backup.

## Local drafts

- Progress autosaves in browser `localStorage` under `codexa-review:<package_sha256>:<reviewer>`, so it survives a refresh.
- The draft is a convenience, labelled `local-draft-not-submitted`. It is never uploaded. The exported JSON file is the review record.
- If local storage is unavailable, the page shows this and warns before you leave with unexported changes.
