# Environment definitions

These files describe the software to install; they are not installed software.
Run `setup_environments.py` from the repository root to install `analysis`
and `policy`. Add `--roles cellpose` for the microscopy example:

| Name | Purpose | Python |
| --- | --- | --- |
| `analysis` | Main analysis, statistics and plots | 3.12.7 |
| `cellpose` | CZI image segmentation demonstration | 3.10.9 |
| `policy` | Saved-policy inference and related plots | 3.10.9 |

`manifest.json` maps each name to its exact dependency lock file. The runner
selects the correct interpreter automatically.

Generated software stays together under the ignored `.local/` subfolder:

- `.local/envs/`: the three installed package environments.
- `.local/python/`: their required Python versions.
- `.local/bootstrap/`: the installation tool.
- `.local/cache/`: downloaded packages and other installation caches.
- `.local/installed.json`: the generated interpreter registry.

Only the definitions and lock files belong in Git. The local development
interpreter `.venv/`, when present, is separate and is not created by this setup.
