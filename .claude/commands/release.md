# Release Procedure

Execute the release procedure for this project. Follow these steps in order:

## 1. Determine Version Bump

Check what has changed since the last release:
```bash
git log --oneline $(git describe --tags --abbrev=0 2>/dev/null || git rev-list --max-parents=0 HEAD)..HEAD
```

Determine version bump based on changes:
- **PATCH** (x.x.X): Bug fixes, minor improvements
- **MINOR** (x.X.0): New features, non-breaking changes
- **MAJOR** (X.0.0): Breaking changes

## 2. Update CHANGELOG.md

Add a new version section under `## [Unreleased]` with today's date:
- Group changes under `### Added`, `### Changed`, `### Fixed`, `### Removed` as appropriate
- Use clear, concise descriptions with **bold** feature names
- Reference any relevant issues or components

## 3. Update Documentation

Review and update any documentation affected by the changes:

- **`docs/` folder**: Update relevant docs (SERVICES.md, API_REFERENCE.md, DATABASE_SCHEMA.md, ARCHITECTURE.md) if:
  - New service methods or API endpoints were added
  - Database schema changed
  - Architecture or patterns changed
- **`README.md`**: Update if there are user-facing feature changes or setup changes
- **`.claude/reports/_registry.md`**: Add entry if any investigation reports were created during this work
- **ADRs**: Create new ADR in `.claude/reports/arch/adr/` if significant architectural decisions were made

## 4. Update Version Files

Update the version in `pyproject.toml`:
```toml
version = "X.Y.Z"
```

## 5. Update Lock File

Run uv sync to update the lock file:
```bash
uv sync --all-extras
```

## 6. Consider Squashing Commits

If there are many small fix commits since the last release, consider squashing them:
```bash
git log --oneline <last-release-tag>..HEAD
git reset --soft <last-release-commit>
git add -A
git commit -m "feat: <summary> (vX.Y.Z)"
```

## 7. Commit, Tag, and Push

Commit all changes, create a version tag, and push:
```bash
git add -A
git commit -m "feat: <brief summary of changes> (vX.Y.Z)

- Change 1
- Change 2
- Change 3"
git tag vX.Y.Z
git push && git push --tags
```

## 8. Verify

Confirm the release:
- Check git log shows the new commit with tag
- Verify CHANGELOG.md has the new version
- Verify pyproject.toml has the updated version
- Verify tag exists: `git tag -l "vX.Y.*"`
