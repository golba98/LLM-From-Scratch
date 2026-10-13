# Consolidation report — implementation ready for review

Canonical PyTorch and NumPy code and asset relocation are implemented; retirement remains pending review/merge and independent backup. See [migration mapping and recovery](MIGRATION.md), [validation](../../VALIDATION.md), [environment limits](../workspace/ENVIRONMENTS.md), and [module ownership](module-mapping.json).

## Original repository inventory

Ten repositories: original 31 (old LLM-Codexa-v1 remote redirects to LLM-Codexa-v1-PyTorch), integration LLM-From-Scratch-PyTorch-Codexa-v1, seven LLM-{Architecture,Tokenizer,Data,Training,Inference,Memory,Specialist}-PyTorch-Codexa-v1 components, and NumPy (old trailing-hyphen remote redirects to LLM-NumPy-CuPy). All existing remotes belong to golba98; default branches are main with active protection rulesets. Original uncommitted source and feature branches are captured by recovery metadata/patches. An obsolete missing registered worktree was pruned only after history and metadata preservation.

## Final layout

```text
3-Python/
├── 37-LLM From Scratch (PyTorch Codexa v1)/  # integration Git root
│   ├── .gitmodules, compatibility.json
│   ├── LLM-{Architecture,Tokenizer,Data,Training,Inference,Memory,Specialist}/
│   │   └── src/llm_<component>/             # pinned independent submodules
│   ├── src/codexa_workspace/               # integration package
│   ├── src/, scripts/                     # legacy adapters / research
│   ├── configs/, tests/, tools/, documentation/
│   ├── outputs/, .venv/, .venv-specialist/  # ignored local runtime
│   └── LLM-From-Scratch/RETIREMENT_PENDING.md
├── 32-LLM (NumPy)/                         # independent Git monorepo
│   ├── src/llm_numpy/
│   │   ├── backend/, ops/, nn/, losses/, optim/
│   │   ├── data/, tokenization/, training/, generation/, cli/, utils/
│   │   └── tensor.py, parameter.py, config.py, assets.py
│   ├── src/                               # legacy adapters
│   ├── configs/, tests/, reference/, scripts/, examples/, benchmarks/, cpp/, docs/
│   └── outputs/, .venv/                    # ignored
└── retained original/deprecated directories with retirement notices
Development/LLM-Assets-Staging/
├── pytorch/{data,checkpoints,exports,logs}/
├── numpy/{current,historical-local,deprecated}/
└── recovery/2026-10-08-consolidation/
```

NumPy preserves its own Tensor, autograd, RMSNorm, attention, RoPE, SwiGLU, Transformer, optimizers and CuPy backend. No PyTorch implementation/dependency was substituted. Package ownership and adapters replace duplicate executable implementations. No new NumPy component repositories were needed.

## Review and remaining blockers

Nine existing repositories require review. Feature branches preserve all existing history and do not overwrite main. Component commits must merge before the integration PR's pinned gitlinks are adopted. The pre-existing integration README PR remains open and may overlap documentation; review its unique changes before resolving it. The existing Specialist PR is reused if its head can advance without history loss.

No remote repositories, project directories, unique experiments or large artifacts were deleted. Single-store relocation avoids permanent duplicate working assets; independent backup remains necessary before retirement. Fresh hermetic environment installation, full large optimizer-resume validation and GitHub Project association are outstanding as documented. No expensive training or automatic merge occurred.

## Publication ledger

Eight new PRs were created and the existing Specialist PR was updated. No PR was merged. Published implementation commits are listed below; subsequent documentation commits may advance the integration/NumPy branch tips. Upstream extraction squash/merge commits were reconciled on feature branches while preserving their original local histories. Origin/main content was byte-identical to the pre-consolidation component trees; add/add conflicts retained the reviewed consolidation versions.

| Repository | Implementation commit | Review PR |
|---|---|---|
| LLM-Architecture | `a9519764b148257372cbe16db363870a942d3165` | [PR](https://github.com/golba98/LLM-Architecture-PyTorch-Codexa-v1/pull/2) |
| LLM-Tokenizer | `b4cefaff09c5c4c17b0efe8b2dc4c13b0fdf7276` | [PR](https://github.com/golba98/LLM-Tokenizer-PyTorch-Codexa-v1/pull/2) |
| LLM-Data | `77adbf98786298f6732201d627be33438b4c98c5` | [PR](https://github.com/golba98/LLM-Data-PyTorch-Codexa-v1/pull/3) |
| LLM-Training | `40f4d848122f7eb997506a2c3ae82bbb03761999` | [PR](https://github.com/golba98/LLM-Training-PyTorch-Codexa-v1/pull/2) |
| LLM-Inference | `465bfd107577b3492f6a794e62ea7a24d96096b2` | [PR](https://github.com/golba98/LLM-Inference-PyTorch-Codexa-v1/pull/2) |
| LLM-Memory | `58e947aabe184acd257a1b0a6d5315e29e4ed1cc` | [PR](https://github.com/golba98/LLM-Memory-PyTorch-Codexa-v1/pull/2) |
| LLM-Specialist | `1a3089d36dcb2038e50be85b921c9426fa8bca15` | [PR](https://github.com/golba98/LLM-Specialist-PyTorch-Codexa-v1/pull/1) |
| integration | `1993904a81d879ff342f0de0c0d4cf32c03ebd2c` | [PR](https://github.com/golba98/LLM-From-Scratch-PyTorch-Codexa-v1/pull/2) |
| NumPy | `c9caadec9aaa8dcbef23e217c13639180e9c1faf` | [PR](https://github.com/golba98/LLM-NumPy-CuPy/pull/2) |

## Dependency diagram

Arrows mean “depends on”; specialist IPC is a runtime boundary rather than a reverse package dependency.

```text
Integration ──> Training ──> Architecture
     │              ├─────> Data ──> Tokenizer
     │              └──────────────> Tokenizer
     ├────────> Inference ──> Architecture + Tokenizer
     │              └─────> Memory (optional adapter)
     └────────> Specialist worker (separate environment, JSON-lines IPC)
NumPy CLI ──> Training / Generation ──> NN / Optim / Losses
                                            └─> Tensor / Ops / Backend
NumPy Data / Tokenization remain internal subpackages; no PyTorch imports.
```

## Retirement sequence

1. Review component PRs, then integration pins and NumPy PR; resolve the overlapping existing integration README PR by review, not automatic closure.
2. Add all nine PRs to Build an LLM From Scratch manually while project scopes are unavailable.
3. Obtain an independently recoverable storage copy of all required artifacts/recovery records and verify hashes.
4. Validate canonical input/output paths, checkpoint resume requirements and clean-host environments for the intended deployment machine.
5. Only after merges, backup and path validation: inventory retained original directories again, preserve any intervening work, verify resolved absolute paths and remove only redundant directories. Do not delete remote repositories automatically.

## Initial Git audit records

| Checkout | Original branch | Original HEAD | Original remote |
|---|---|---|---|
| 31-LLM (PyTorch) | `agent/rebuild-1b-base` | `66eec752688afad348206f22db4c90f8f2a47eda` | https://github.com/golba98/LLM-Codexa-v1.git |
| 32-LLM (NumPy) | `main` | `a83bc14bbd246b07e77d922ea6c26340f2fa4c5b` | https://github.com/golba98/LLM-NumPy-CuPy-.git |
| LLM-Architecture | `save-complete-architecture` | `5cada37fdc9011abdfba351d076910517b0fbbd6` | https://github.com/golba98/LLM-Architecture-PyTorch-Codexa-v1.git |
| LLM-Data | `feat/save-data-project` | `479f248dcecb493b7142b6e20ce527f9f3e34af0` | https://github.com/golba98/LLM-Data-PyTorch-Codexa-v1.git |
| LLM-From-Scratch | `workspace/component-extraction` | `66eec752688afad348206f22db4c90f8f2a47eda` | https://github.com/golba98/LLM-From-Scratch-PyTorch-Codexa-v1.git |
| LLM-Inference | `docs/repository-heading` | `e14df720ea9bc5752f0bfde53634fa60684e3ba2` | https://github.com/golba98/LLM-Inference-PyTorch-Codexa-v1.git |
| LLM-Memory | `save-project` | `a60703a35992ab6f8595ea3610915bc463ad8e96` | https://github.com/golba98/LLM-Memory-PyTorch-Codexa-v1.git |
| LLM-Specialist | `save-specialist-project` | `d161bc1823ff02b68da32298be1828464bee6180` | https://github.com/golba98/LLM-Specialist-PyTorch-Codexa-v1.git |
| LLM-Tokenizer | `save-tokenizer-project` | `0d8e08e799a7da10e6246982da57c5267de3bd06` | https://github.com/golba98/LLM-Tokenizer-PyTorch-Codexa-v1.git |
| LLM-Training | `feat/save-training-project` | `33effac8ef9ff18e39253af40b10b3b0e6438a55` | https://github.com/golba98/LLM-Training-PyTorch-Codexa-v1.git |

Original 31 had 30 modified tracked files and 77 untracked entries; integration had the uncommitted extraction; NumPy had four tracked changes plus unique configs/scripts/reference fixtures. These were archived before consolidation and included where appropriate rather than reset. Seven component working trees were clean at audit start. Their previously merged extraction PRs were fetched and reconciled before publishing. Original/deprecated source remains locally with retirement notices, and no deletion is asserted.

Fresh published-clone checks: PyTorch 119 passed / 7 skipped; NumPy 126 passed, with every pinned submodule fetched from GitHub. Two private reference checks explain the PyTorch difference from the canonical configured workspace.
