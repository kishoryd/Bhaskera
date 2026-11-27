# gateway

Planned, not built. One front door for every user of a cluster: sign in with the cluster login (LDAP), then reach your own copy of the dashboard.
The plan, the choices and the questions for the admins are in [docs/central-service/DESIGN.md](../../../docs/central-service/DESIGN.md).

## What you will control

A `gateway.yaml` (nothing about a particular cluster is in the code):

| Key | Meaning |
|---|---|
| `ldap.uri`, `ldap.ca_file`, `ldap.base_dn`, `ldap.bind_dn`, `ldap.bind_password_file` | The directory, how to trust it, where users are, and the read-only account that searches |
| `roles.users`, `roles.admins` | The directory groups that may sign in, and that may read the audit log |
| `session.idle_minutes`, `session.max_hours` | How long a sign-in lasts |
| `agent.idle_minutes`, `agent.max_per_user` | When a user's page is stopped, and how many one user may have |
| `cluster_profile` | The cluster profile file, as for the rest of ExaAI |
| `audit.dir` | Where the audit log is written |

## What it will do itself

- **Sign-in** (`auth`): checks the password against the directory and forgets it, keeps a server-side session, limits attempts.
- **Forwarding** (`proxy`): sends each request to the Unix socket of the signed-in user's page, and nothing else.
- **Starting pages** (`agent`): starts a user's page as that user, stops it when idle, ends it when the user leaves the directory.
- **Audit** (`audit`): an append-only record of sign-ins and of every job started or cancelled.
