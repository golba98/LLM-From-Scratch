# Validated environments

PyTorch's existing generative and specialist environments were relocated on the same machine, generated entry points repaired and locally built component/integration wheels installed with --no-deps. Runtime imports, model loading and package dependency checks validate relocation. This avoids re-downloading large binary libraries and preserves exact installed versions. Virtual environments are not general-purpose portable backups: rebuild when moving to another host using the recorded external pins and matching CUDA wheel source.

Install local component wheels together; their distributions are not assumed to exist on PyPI. Keep specialist dependencies in its separate environment. Component dependency metadata permits the existing 4.70.x tqdm patch versions; each environment's exact version remains pinned in its lock.

NumPy uses a project-local environment inheriting existing system scientific libraries, with the project installed locally. Its CPU/CUDA baseline versions are recorded in its lock files and environment report. An isolated clean-host dependency installation is a remaining reproducibility check; current runtime validation is not evidence of a hermetic installation.
