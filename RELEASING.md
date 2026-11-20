# Maintaining and Releasing Bhaskera

## Branches

| Branch | Purpose |
|---|---|
| `main` | Released, tested code. Changes arrive only through reviewed pull requests. |
| `feature/<name>`, `fix/<name>`, `docs/<name>` | One change each, branched from `main`, merged by pull request. |

Every pull request: runs the CPU tests (`pytest src`), updates `CHANGELOG.md` under **[Unreleased]**,
and — if it changes training behaviour — is checked on the cluster with at least one recipe smoke test
(`./submit.sh slurm/recipes/<recipe>.yaml <nodes> 1`).

## Versions

Bhaskera follows [Semantic Versioning](https://semver.org/):

| Change | Bump | Example |
|---|---|---|
| A config key or CLI option removed or changed so existing runs break | MAJOR | 2.3.0 → 3.0.0 |
| New feature; existing configs still work | MINOR | 2.3.0 → 2.4.0 |
| Bug fix only | PATCH | 2.3.0 → 2.3.1 |

The version lives in one place, `pyproject.toml`; code reads it from the installed package.

## Cutting a release

1. On `main`, move the **[Unreleased]** entries in `CHANGELOG.md` under a new `## [X.Y.Z] - YYYY-MM-DD`
   heading and update the links at the bottom.
2. Set `version = "X.Y.Z"` in `pyproject.toml`.
3. Run the CPU tests and the recipe smoke tests on the cluster.
4. Commit (`Release X.Y.Z`), then tag and push:

   ```bash
   git tag -a vX.Y.Z -m "Bhaskera X.Y.Z"
   git push origin main vX.Y.Z
   ```

5. On GitHub, create a release from the tag and paste the changelog section as its notes.
6. Central installation (PARAM Rudra): set `REF=vX.Y.Z` and run `slurm/install-bhaskera.sh` with
   `PREFIX=/home/apps/bhaskera/X.Y.Z`, then add `slurm/modulefiles/bhaskera/X.Y.Z.lua` (copy the previous
   one and change `root` and the version).

## Dependency lock

`slurm/requirements-paramrudra.lock` holds the exact versions of the environment the release was
validated with. When upgrading a dependency, rebuild the environment, rerun the recipe smoke tests on the
cluster, and regenerate the lock from that environment. Do not loosen it without that test: newer Ray and
grpcio releases have broken multi-node start-up before.
