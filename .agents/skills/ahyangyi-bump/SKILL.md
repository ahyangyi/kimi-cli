---
name: ahyangyi-bump
description: Rebase all ahyangyi fork head-* branches onto upstream HEAD.
---

This skill rebases all `head-*` branches on the ahyangyi fork against the current upstream HEAD.

## Procedure

The remote for the fork is `git@github.com:ahyangyi/kimi-cli.git`.

### 1. Fetch

```bash
git fetch origin
git fetch git@github.com:ahyangyi/kimi-cli.git 'refs/heads/head-*:refs/remotes/ahyangyi/head-*'
```

### 2. Rebase each branch

For each `head-{foo}` branch:

```bash
git checkout -b head-{foo} ahyangyi/head-{foo}
git rebase origin/main
```

Resolve any conflicts, then continue.

### 3. Push

Force-push all rebased branches back to the fork:

```bash
git push git@github.com:ahyangyi/kimi-cli.git head-{foo} --force-with-lease
```

## Notes

- Resolve rebase conflicts as they arise; prefer upstream for build/config, fork for custom feature code.
- Always use `--no-verify` with git commits to skip commit hooks.
- Verify the build passes on each branch after rebasing.
