# A central Bhaskera service for every user of a cluster

This is the plan for one Bhaskera service that everyone on a cluster signs in to with their cluster login (LDAP), where each person
sees and runs only their own work. It says what exists today, what has to be added, what you need from other people, and what is hard.
Nothing here is built yet. Facts about the code are checked against this repository; facts about the cluster are marked **to confirm**.

## 1. The idea in one paragraph

Today every user starts their own copy of the home page (`bhaskera-dashboard ray`) on a login node and reaches it through an SSH tunnel.
The page is safe because it belongs to one Unix user: it listens on `127.0.0.1`, refuses any connection that is not from the same user,
and starts jobs with that user's own Slurm account. A central service keeps that property and moves the front door: one HTTPS address,
a sign-in with the cluster's LDAP account, and behind it **one small Bhaskera process per signed-in user, running as that user**. The
central part only proves who you are and forwards your browser to your own process. It never holds a user's cluster credentials, and it
never runs a job itself.

## 2. What exists today (and why it matters)

| Today | Consequence for a central service |
|---|---|
| The page (`ray_hub.py`) binds `127.0.0.1` and checks the peer's Unix user (`hub_guard.same_user_peer`) | It cannot be put behind a shared proxy as it is: every request would come from the proxy's user |
| Jobs go through `submit.sh` as the user, with `SLURM_ACCOUNT` taken from the environment | A central service must find each user's Slurm account itself (`sacctmgr`), not read one variable |
| MLflow runs and the model registry are files in the user's own `~/mlflow-runs` | Separation between users comes free; sharing between users needs a deliberate design (section 6) |
| Links to MLflow and to each job's Ray Dashboard use host names like `mlflow.localhost:8265` and `job-<id>.localhost:8265` | A shared address needs either wildcard DNS and a wildcard certificate, or links under one path (section 7) |
| A chat server runs on a compute node, and the page reaches it with an SSH forward from the login node | The central host must be able to reach compute nodes, or go through a login node as the user |
| Everything is configured by environment variables and a cluster profile file | The service must keep the same rule: cluster facts in a config file, never in code |
| On a shared account (like `dlapp`) every person sees every job | Real per-person sign-in fixes this; it is the strongest reason to build the service |

## 3. Three ways to build it

| | A. Inside Open OnDemand | B. Own gateway, one process per user | C. One server that acts for everyone |
|---|---|---|---|
| What it is | Bhaskera as an app in the portal the cluster already has | A small Bhaskera gateway that signs people in and proxies to their own page | One multi-user web app holding a service credential that can submit as any user |
| Sign-in | The portal's (whatever it uses; **to confirm**) | LDAP directly, or OIDC in front of LDAP | LDAP directly |
| Runs as the user? | Yes: the portal starts a per-user process | Yes, if a launcher can start processes as the user (section 5) | No: it acts for the user |
| If the service is broken into | Attacker gets the portal's reach, which already exists | Attacker gets the gateway, which holds no cluster credentials | Attacker can submit and read as every user |
| Work to build | Least: an OnDemand app plus the changes in section 5 | Medium: the gateway, a launcher, the same changes | Medium, then a large security review |
| Verdict | **Start here** if the admins agree | The fallback if OnDemand cannot host it | Do not build this |

The cluster's user manual says Open OnDemand is running at `ood.paramrudra.cdacb.in`, with a file browser, job submission and a shell. That
means the hardest parts (sign-in, session cookies, TLS, starting a process as the user) may already be solved. Option A reuses them.
Option B is the same design built by us. Option C is listed only so the reason not to is written down.

## 4. The architecture (A and B are the same shape)

```
 browser ──HTTPS──▶ front door ──▶ Bhaskera page of THIS user ──▶ submit.sh / Slurm  (as the user)
                    (sign-in,       (a process owned by the user,    MLflow files, registry (the user's)
                     sessions,       listening on a Unix socket       Ray / chat on compute nodes
                     TLS, logs)      only that user can open)
```

- **Front door.** Option A: OnDemand's own web server. Option B: our gateway (nginx or a small Python service) with LDAP sign-in. It checks the
  session, then forwards to the socket of that user's page and adds nothing the user could forge.
- **Per-user page.** The existing hub, changed to listen on a Unix socket, to accept the front door as its only client, and to serve under a path prefix.
  It starts when someone signs in and stops after an idle time.
- **State** stays where it is: in the user's home and scratch. The service itself holds almost nothing, so it can be restarted or
  replaced without losing anyone's work, and "permanent" means the files of each user, which already survive.

## 5. What has to change in Bhaskera

1. **Listen on a Unix socket** (path `/run/user/<uid>/bhaskera.sock` or inside the user's home with mode 0600) as well as the loopback port. The kernel then enforces
   "only this user", without relying on a port lookup. Keep loopback for the tunnel use that exists today.
2. **Trust the front door, and only it.** The page accepts a request only from the socket and takes the user name from the front door's header, checking it equals the
   Unix user the process runs as. A header from anywhere else is ignored.
3. **Serve under a prefix** (`/bhaskera/`). Today the page fetches `/api/...` and links as if it lived at `/`; those must become relative to a base path.
4. **Replace the `*.localhost` host names.** Pick one: (a) wildcard DNS and a wildcard certificate (`*.bhaskera.<domain>`), simple for the page, needs the network team; (b) path
   routing (`/mlflow/`, `/job/<id>/`) with the proxy rewriting links, no DNS work but MLflow and Ray assume they live at `/`.
5. **Find the user's Slurm account** from `sacctmgr show assoc user=<user>`, offer a choice when there are several, and stop reading `SLURM_ACCOUNT` from the shell.
6. **Start and stop** the per-user page: a launcher that starts it as the user on sign-in and ends it when idle (section 8).
7. **A config file** (`gateway.yaml`): LDAP address and search base, group names for roles, the cluster profile to use, idle time, log folder. None of it in code.
8. **An audit log** the user cannot edit: who signed in, from where, and every job started or cancelled through the page.

The first four are the real work. They touch the page's HTML and the proxy, not the training code.

## 6. Separation between users

| Layer | How users are kept apart |
|---|---|
| Identity | LDAP sign-in; the Unix user comes from the directory, never from anything the browser sends |
| Process | One page process per user, started as that user; no process ever serves two users |
| Files | Ordinary Unix permissions on home and scratch; the socket is mode 0600; nothing is written to a shared folder |
| Slurm | Jobs are submitted as the user, with an account and QoS that `sacctmgr` says the user has; the existing partition and QoS checks still run |
| Runs and models | Each user's own MLflow folder. Cancel works only on jobs whose Slurm comment starts with `bhaskera`, and only the user's own jobs are listed |
| Network | The front door reaches only the per-user sockets and the compute-node ports it must; compute nodes never accept the browser directly |
| Secrets | The service stores no passwords and no cluster credentials. LDAP passwords are checked and forgotten |
| Logs | Per-user job logs stay in the user's folder; the audit log is central and append-only |

**Sharing on purpose.** A team registry ("the model we use") is a separate decision. It needs a shared folder with a group, one MLflow store per project, or a small registry
service with its own permissions. Build the per-user service first; add team features after, with an explicit group-to-project mapping from LDAP.

## 7. Sign-in with LDAP: what is needed

- **Reachability and TLS.** The front door host must reach the directory over LDAPS or StartTLS and trust its certificate (the CA file goes in `gateway.yaml`).
- **How to check a password.** Either *bind as the user* (build the user's DN, try to bind with the password they typed) or *search then bind* (a read-only service
  account finds the DN from the username, then bind as the user). Search-then-bind copes with directories where the DN is not guessable. Escape the username in the filter, or it
  becomes an LDAP injection.
- **A read-only service account** for searches, with a password kept in a file only the service reads. Nothing else needs write access to the directory.
- **Groups to roles.** `bhaskera-users` may sign in; `bhaskera-admins` may see the audit log and the service's health. Per-project groups come later.
- **Same identities as the cluster.** The front door host should resolve users the way the cluster does (SSSD against the same directory), so that name, UID and groups are
  identical. If they differ, the launcher starts a process as the wrong person. This is the most common mistake in setups like this.
- **Sessions.** Secure, HTTP-only, same-site cookies; a server-side session store; a short idle time and a longer absolute time; CSRF protection (the page already requires a
  header only its own script sends); rate limits and lockout against password guessing.
- **Better than raw passwords.** If the site has an identity provider (Keycloak, Dex, a university SSO) in front of LDAP, sign in with OIDC through it: the service never sees a password,
  and multi-factor sign-in comes with it. Open OnDemand is usually set up this way.
- **Leaving.** When someone is removed from the directory or the group, their running page and their sessions must end within minutes, not days.

## 8. What is hard (and what to do about it)

| Challenge | Why it hurts | What to do |
|---|---|---|
| Starting a process as another user | Needs privilege, which is exactly what an attacker wants | Use what exists: OnDemand's per-user server launcher, or a `sudo` rule that allows one fixed command for the service account, nothing else. Never run the whole service as root |
| Idle processes pile up | Hundreds of users, one process each, on a shared host | Idle timeout, a cap per user, a memory limit through systemd or cgroups |
| Streaming through the proxy | Chat answers stream (server-sent events); some proxies buffer them | Turn buffering off for those routes; test with a long answer |
| Ray Dashboard and MLflow behind a proxy | Both assume they live at `/`, and the Ray Dashboard is read-only only because the page enforces it | Wildcard host names (4a) or careful rewriting; keep the read-only rule in the page |
| Reaching compute nodes | A chat server listens on a compute node; a firewall may block the front door | Go through a login node as the user (as today), or open a documented port range from the front door only |
| Same filesystem paths | The page reads `/scratch` and `/home`; a service VM that does not mount them sees nothing | Run the front door on a host that mounts the cluster filesystems, or keep the per-user page on a login node and only proxy to it |
| Slurm client on the host | `sbatch` needs the cluster's config and authentication | Run the per-user page on a login node (it has both) and keep the front door thin |
| Shared accounts | Everyone in a project shares one Slurm account, so per-person limits do not apply | Per-person sign-in solves who sees what; limits stay per account unless Slurm is changed |
| Uploads in chat | Image, audio and video files are large | Raise the proxy's body limit for that route only; keep the page's own size caps |
| Timeouts | Long requests (a 600 s upstream today) hit proxy defaults | Set them per route |
| Upgrades | One service for everyone, so a bad release is everyone's problem | Version in the path, a staging copy, and roll back by pointing the launcher at the old release |
| Being the single sign-in point | A hole in the gateway exposes every user | Keep it small, patch it, log everything, put it behind the site's firewall and review it with the security team |

## 9. A plan in phases

0. **Ask the admins** (section 10) and get a short yes or no on OnDemand, a service host and the directory details. No code until then.
1. **Spike inside OnDemand.** One user, one app that starts the existing page behind OnDemand's sign-in. Learn what OnDemand gives (cookies, user name, path prefix) and what breaks.
2. **Make the page proxy-ready** (section 5, items 1 to 4). Tests: a request from the wrong user, a forged header, and the same page under a prefix.
3. **Gateway for option B** only if option A is refused: LDAP sign-in, launcher, idle reaper, audit log.
4. **Per-user Slurm account selection and the audit log**, then hardening: rate limits, session lifetime, log shipping, restart behaviour.
5. **Team features:** shared registry, per-project groups, quotas. Separate branch of work, separate review.

## 10. Questions for the cluster admins

1. Is Open OnDemand the portal people already use for sign-in, and may Bhaskera be added to it as an app? How does it authenticate (LDAP, OIDC, something else)?
2. Is there a service host (not a login node) for the front door, and does it mount `/home` and `/scratch`, run SSSD against the cluster directory, and have the Slurm client?
3. Which LDAP server, search base and groups? LDAPS or StartTLS? Can we get a read-only service account? Is there an identity provider in front of it?
4. May a service account start a process as a user (a `sudo` rule for one fixed command), or must it go through OnDemand's launcher?
5. Can the front door host reach compute nodes (which ports), or must everything go through a login node?
6. Can we have a DNS name and a certificate, including a wildcard one for `*.bhaskera.<domain>`?
7. How should Slurm accounts map to people: one account per person, or per project? Is `slurmrestd` enabled, and do we want to use it?
8. What must be logged, for how long, and who may read it? Is there an existing log store?
9. Who patches and runs the service: the admins, or us?

## 11. Where the work lives

This branch (`central-service`). The central parts go in a new package, `src/bhaskera/gateway/`, next to `launcher/`: `auth` (LDAP and sessions), `proxy` (front door to
per-user socket), `agent` (start and stop a user's page), `audit`. The changes to the per-user page (sections 5.1 to 5.5) stay in `launcher/`, because the same page keeps working
for people who run it by hand. As in the rest of the repository, cluster facts live in `gateway.yaml` and the cluster profile, never in the code or the page.
