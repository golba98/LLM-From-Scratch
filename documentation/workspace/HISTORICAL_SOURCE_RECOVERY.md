# Historical foundation source recovery

The completed `codexa-934m-foundation-5h-v1` job is immutable. Its original
source is preserved in the private, content-addressed archive identified by
`HISTORICAL_SOURCE_V1.json`. The inventory is suitable for Git; the archive is
ignored because it also contains working-tree diffs and installed-package
metadata. Ten of the twelve job-bound source files were not committed to Git.
Each frozen source entry in the inventory records its owner repository, original
relative path, and a byte-matching Git commit when one exists. Only
`llm_architecture.model` and `llm_data.token_data` have such a commit; the other
ten require the private archive.

From `LLM-From-Scratch`, verify using any Python 3.11+ interpreter:

```sh
python3 tools/historical_source_archive.py verify
python3 tools/historical_source_archive.py verify --restore /tmp/codexa-historical-source-v1
```

The destination must not already exist. Verification extracts all 231 files to
a temporary directory, checks every SHA-256 against the inventory, and checks
all twelve source identities against the completed job. It never imports a
model, uses `.venv`, or edits the current owner repositories. The restored
`code-identity/` tree preserves the job's paths. `venv-components/` contains
the historical installed Codexa packages; `integration-src/` contains the
integration package; `pip-freeze.txt` records the original package set. A
separate environment needs third-party packages before historical code can
execute. Do not install those packages over the existing `.venv`.

The archive and original snapshot are on the same physical disk. This is
source recovery independent of `.venv`, but not off-device disaster recovery.
Copy the archive and the inventory to independent protected storage, verify
the archive SHA-256 there, and record that verification before claiming an
independent backup. Do not push the private archive or session diffs.
