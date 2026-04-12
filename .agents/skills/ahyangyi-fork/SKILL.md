---
name: ahyangyi-fork
description: Bump version for ahyangyi's fork branches.
---

This skill handles cherry-picking and merging the ahyangyi fork branches when the upstream Kimi Code CLI version is bumped.

## Parameters

- `{old_version}` — the previous upstream version (e.g. `1.29.0`)
- `{new_version}` — the new upstream version (e.g. `1.30.0`)

## Procedure

The remote for the fork is `git@github.com:ahyangyi/kimi-cli.git`.

### 1. Cherry-pick the base branch

Create `{new_version}-ahyangyi-base` from the new upstream version, then cherry-pick the commits from `{old_version}-ahyangyi-base` that are not in the old upstream:

```bash
git checkout -b {new_version}-ahyangyi-base {new_version}
git cherry-pick {old_version}..{old_version}-ahyangyi-base
```

### 2. Fork into the integration branch

Create `{new_version}-ahyangyi` from the new base:

```bash
git checkout -b {new_version}-ahyangyi {new_version}-ahyangyi-base
```

### 3. Cherry-pick and merge each feature branch

Find all branches on the `ahyangyi` remote matching `head-{foo}`:

```bash
git fetch git@github.com:ahyangyi/kimi-cli.git 'refs/heads/head-*:refs/remotes/ahyangyi/head-*'
```

For each `head-{foo}` branch:

1. Create a local branch from the new base and cherry-pick the feature commits:

   ```bash
   git checkout -b {new_version}-{foo} {new_version}-ahyangyi-base
   git cherry-pick {old_version}-ahyangyi-base..ahyangyi/head-{foo}
   ```

2. Merge it into the integration branch:

   ```bash
   git checkout {new_version}-ahyangyi
   git merge {new_version}-{foo}
   ```

### 4. Push results

Push all updated branches to the fork remote.

## Notes

- Resolve any cherry-pick conflicts as they arise; prefer the upstream version for build/config changes and the fork version for custom feature code.
- Always use `--no-verify` with git commits to skip commit hooks.
- Always verify the build passes after the final merge into `{new_version}-ahyangyi`.
