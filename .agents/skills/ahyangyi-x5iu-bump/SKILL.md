---
name: ahyangyi-x5iu-bump
description: Rebase all ahyangyi fork x5iu-* branches onto upstream HEAD.
---

This skill rebases all `x5iu-*` branches on the ahyangyi fork against the current upstream HEAD.

## Procedure

The remote for the fork is `git@github.com:ahyangyi/kimi-cli.git`.

### 1. Fetch

```bash
git fetch origin
git fetch git@github.com:ahyangyi/kimi-cli.git 'refs/heads/x5iu-*:refs/remotes/ahyangyi/x5iu-*'
```

### 2. Rebase each branch

For each `x5iu-{foo}` branch:

```bash
git checkout -b x5iu-{foo} ahyangyi/x5iu-{foo}
git rebase origin/main
```

Resolve any conflicts, then continue.

### 3. Push

Force-push all rebased branches back to the fork:

```bash
git push git@github.com:ahyangyi/kimi-cli.git x5iu-{foo} --force-with-lease
```

## Notes

- Resolve rebase conflicts as they arise; prefer upstream for build/config, fork for custom feature code.
- Always use `--no-verify` with git commits to skip commit hooks.
- Verify the build passes on each branch after rebasing.
