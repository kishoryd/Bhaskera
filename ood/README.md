# Bhaskera in Open OnDemand: setting it up (for the administrators)

A user opens **Interactive Apps, Bhaskera**, presses **Launch**, then **Open Bhaskera**. That is all they do: no form, no account to type, no folder to make, nothing to
export. The portal signs them in (Keycloak), starts a small job as them, and the Open button leads to Bhaskera's home page in a new tab, where they create recipes,
train, chat, benchmark and register models. Only that user can open their page.

This is the whole of your side. It takes about ten minutes, and the only parts that need thought are steps 1, 5 and 6.

## What you need before you start

| Need | Why | How to check |
|---|---|---|
| Open OnDemand with batch_connect apps (any version that runs Jupyter or RStudio sessions) | The app is a standard interactive app | Another interactive app already works for you |
| The `bhaskera/1.0.0` module on the compute nodes, from the build that has the `portal` command | The session job loads it and runs the page | `module load bhaskera/1.0.0 && bhaskera-dashboard --help` lists `portal` |
| A CPU partition whose time limit is at least the hours you choose | The page needs 2 CPUs and 4 GB, no GPU. It starts GPU jobs itself, in the GPU partitions | `sinfo -p <partition> -o '%P %l'` |
| Slurm client tools on the compute nodes | The page runs `sbatch`, `squeue` and `sacctmgr` as the user | They already work in any batch job |
| The web nodes can reach the compute nodes on high ports | The portal's `/rnode/` proxy forwards to the session | Another interactive app already works |
| SSH between compute nodes for a user who has a job on both (only for the chat window) | The page reaches a chat server on another node through an SSH link | `ssh <node with the user's job> true` from a compute node |

## Steps

Do steps 2 to 4 on **every** web node (the portal runs on at least two: its cookie names `ood_node2`).

**1. Choose the partition and the hours.** The session needs 2 CPUs and 4 GB for as long as the user keeps the page open. Pick a CPU partition and a number of hours that fits its limit.
From what the cluster shows today: `terai` allows 24 h, `himachal` 12 h, `himadri` 6 h; the default partition `debug` allows only 1 h, which is too short. Confirm with `sinfo -o '%P %l'`.

**2. Get the app onto the web node.** From a checkout of the Bhaskera repository (the `ood/` folder), or copy it:

```bash
git clone https://github.com/kishoryd/Bhaskera.git && cd Bhaskera/ood     # or: scp -r ood/ <webnode>:
```

**3. Install it.**

```bash
sudo ./install.sh --partition terai --hours 12
```

It copies the app to `/var/www/ood/apps/sys/bhaskera` and writes three settings into the form once: the cluster id (read from `/etc/ood/config/clusters.d/` when there is exactly one
file there, otherwise pass `--cluster <id>`), the partition and the hours. It checks that the partition exists. Run it again to change a setting. Nothing else is touched.
Users see none of these settings.

**4. Let the portal's proxy reach the compute nodes.** Edit `/etc/ood/config/ood_portal.yml` so that `host_regex` matches the names of the compute nodes a session can land on, and
the proxy paths are set. Node names seen on this cluster are `cbcn####`, `cbgpu####` and `cbhm####`, so for example:

```yaml
host_regex: 'cb[a-z]+[0-9]+'
node_uri: '/node'
rnode_uri: '/rnode'
```

If it already matches (another interactive app works), leave it. Then apply it, as the OnDemand documentation describes for your version:

```bash
sudo /opt/ood/ood-portal-generator/sbin/update_ood_portal
sudo systemctl try-restart httpd
```

**5. Run the checker** on the web node:

```bash
./check.sh --nodes "cbcn0176 cbgpu0049 cbhm0001"      # one real name of each node type
```

It checks that the app and the cluster file are there and that `host_regex` accepts each name (a name that is rejected would give the user "Forbidden" on the Open button). It prints
a last line about the firewall, which it cannot check.

**6. Check the network once.** If Jupyter, RStudio or the desktop session already work through the portal, the web nodes already reach the compute nodes and you are done. If this is the
first interactive app, open a session, then from the web node run `curl -s -o /dev/null -w '%{http_code}\n' http://<compute node>:<port>/` with the port shown on the session card: any answer
(Bhaskera answers 403 without the token) means the path is open; a timeout means a firewall rule is needed from the web nodes to the compute nodes (TCP, ports 1024 and up).

**7. Try it as a user.** Sign in to the portal, Interactive Apps, Bhaskera, Launch. When the card says Running, press Open Bhaskera. You should see the home page with your runs. If you are
new on the cluster it says "Nothing is running", and that is normal.

## If something goes wrong

| What you see | Cause | Fix |
|---|---|---|
| Open Bhaskera gives "Forbidden" or a proxy error | `host_regex` does not match the node name | Step 4, then `./check.sh --nodes "<that node>"` |
| The card stays "Starting" for minutes | The partition is busy or the limit is smaller than the hours | `squeue -u <user>`; lower `--hours`; pick another partition (rerun `install.sh`) |
| The card goes to "Completed" at once | The module or the `portal` command is missing on that node | Open the session's `output.log` from the card |
| The page says "Open Bhaskera from the Sessions page" | The address was opened without the Open button (a bookmark, a copied link) | Use the Open button; the token changes with each session |
| The page says "SLURM_ACCOUNT is not set" under Start | The user has no default account in Slurm | `sacctmgr modify user <user> set defaultaccount=<account>` |
| The page opens but nothing loads | The proxy dropped a header or cookie | Look at the web server's error log; the page needs `Set-Cookie` back and `Cookie` forward |
| Chat says the server cannot be reached | SSH between compute nodes is not allowed for this user | Allow SSH between compute nodes for a user who has a job there (the last row of the first table) |

## What runs, and what it can touch

- The session is an ordinary job of the user: their own Slurm account, quota and limits. Bhaskera adds no privilege. Nothing is shared between users.
- The page listens on the node's network address (so the portal can reach it) and answers only a request that carries the session's random 24-character token, handed to the user's browser by the
  Open button and kept in an `HttpOnly` cookie. Changes also need a header that only the page's own script sends. The token lives in the job's environment, as for the portal's other interactive apps.
- On first use the session makes `bhaskera-runs` in the user's `$SCRATCH` (or `$HOME` if there is none), copied from the shared kit. Recipes, caches and checkpoints go there.
- The links to MLflow's own pages and to each job's Ray Dashboard are hidden behind the portal (they need `*.localhost` names). Runs, Models, Compare, logs, Start, Chat and Benchmark work.

## Updating and removing

- A new Bhaskera version: update the module on the cluster. Nothing to do in the portal unless the app files changed; then run `install.sh` again on each web node.
- Removing the app: `sudo rm -rf /var/www/ood/apps/sys/bhaskera`.

## What was tested, and what was not

Tested on PARAM Rudra, without the portal itself: the page as a job on a compute node, reached from a login node over the network (no or wrong token: 403; right token: cookie, then the page,
the user's runs, models and tests; *Check first* for a training recipe from inside the job); the page in a real browser behind a proxy that strips a `/rnode/<host>/<port>` prefix; the app's
templates rendered and parsed; `install.sh` and `check.sh` against a temporary folder.

Not tested, because it needs the portal: the app inside Open OnDemand (the proxy passing the cookie and the form post, the card turning Running), a session on two web nodes, many users at once,
and the chat window from inside a session.
