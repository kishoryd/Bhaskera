# Maintaining and Releasing Bhaskera

## Branches and releases

`main` holds released code only: the tip of `main` is always the latest release. Every release is a tag
(`v1.0.0`) on a commit of `main`, with a matching GitHub release. Work in progress lives on a short-lived
branch and goes into `main` only when it is ready to be released.

Before a change goes into `main` it must:

- pass the CPU tests (`pytest src`),
- have its `CHANGELOG.md` entry under **[Unreleased]**,
- if it changes training behaviour, run on the cluster with at least one recipe
  (`./submit.sh slurm/recipes/<recipe>.yaml <nodes> 1`).

A fix to a released version is made on `main` and shipped as the next patch release.

## Versions

Bhaskera follows [Semantic Versioning](https://semver.org/):

| Change | Bump | Example |
|---|---|---|
| A config key or CLI option removed or changed so existing runs break | MAJOR | 1.0.0 → 2.0.0 |
| New feature; existing configs still work | MINOR | 1.0.0 → 1.1.0 |
| Bug fix only | PATCH | 1.0.0 → 1.0.1 |

The version lives in one place, `pyproject.toml`; code reads it from the installed package.

## Cutting a release

1. On the release branch, move the **[Unreleased]** entries in `CHANGELOG.md` under a new `## [X.Y.Z] - YYYY-MM-DD`
   heading and update the links at the bottom.
2. Set `version = "X.Y.Z"` in `pyproject.toml`.
3. Run the CPU tests, and run at least one recipe on the cluster.
4. Commit (`Release X.Y.Z`), fast-forward `main` to that commit, then tag and push:

   ```bash
   git tag -a vX.Y.Z -m "Bhaskera X.Y.Z"
   git push origin main vX.Y.Z
   ```

5. On GitHub, open **Releases → Create a new release**, choose the tag, and paste the changelog section as
   the notes. A release is the tag plus this page; pushing the tag alone leaves the Releases panel empty.
6. Central installation (PARAM Rudra): set `REF=vX.Y.Z` and run `slurm/install-bhaskera.sh` with
   `PREFIX=/home/apps/bhaskera/X.Y.Z`, then add `slurm/modulefiles/bhaskera/X.Y.Z.lua` (copy the previous
   one and change `root` and the version).

## Dependency lock

`slurm/requirements-paramrudra.lock` holds the exact versions of the environment the release was
tested with. When upgrading a dependency, rebuild the environment, rerun a recipe on the
cluster, and regenerate the lock from that environment. Do not loosen it without that test: newer Ray and
grpcio releases have broken multi-node start-up before.
