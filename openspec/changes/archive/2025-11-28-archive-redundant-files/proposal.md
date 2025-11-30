# Change: Archive Redundant Files

## Why

The root directory contains several obsolete scripts and configuration files that were superseded by the recent core architecture refactor. These files:

1. **Clutter the project root** - Making it harder to identify active entry points
2. **Use deprecated patterns** - Old config files bypass the new `.env`-based configuration
3. **Duplicate functionality** - `create.py` and `dataset_generator.py` are nearly identical
4. **Reference removed code** - `sandbox.py` imports deprecated `config/server_config.py`

Moving these to `archive/` preserves them for reference while cleaning up the project structure.

## What Changes

- Move 6 obsolete root-level scripts to `archive/deprecated-scripts/`
- Move entire `config/` folder to `archive/deprecated-config/`
- Update `.gitignore` if needed to exclude any generated files

## Files to Archive

**Root scripts → `archive/deprecated-scripts/`:**
| File | Reason |
|------|--------|
| `main.py` | Empty stub, never used |
| `create.py` | Superseded by pipeline components |
| `dataset_generator.py` | Duplicate of `create.py` |
| `sandbox.py` | Dev scratch file, uses deprecated imports |
| `create_part_of_day_files.py` | One-off utility, completed its purpose |
| `legacy_live_plot.py` | Already marked legacy, uses removed `db.py` |

**Config folder → `archive/deprecated-config/`:**
| File | Reason |
|------|--------|
| `config/mongo_config.py` | Replaced by `src/config.py` + `.env` |
| `config/mongo_config.py.template` | Documentation only, outdated |
| `config/server_config.py` | Replaced by `src/config.py` + `.env` |
| `config/server_config.py.template` | Documentation only, outdated |
| `config/simulation_config.py` | Unused |
| `config/simulation_config.py.template` | Unused |

## Impact

- **No code changes** - Only file relocation
- **No breaking changes** - These files are not imported by active code
- **Preserves history** - Files remain accessible in `archive/`
