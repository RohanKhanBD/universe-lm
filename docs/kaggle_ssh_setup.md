# Kaggle SSH via KaggleLink

Real bash shell into a Kaggle notebook over a Zrok tunnel. Pairs with
`scripts/kg_ssh.sh` (this repo) for connect / rsync / status.

## TL;DR — copy-paste cheatsheet

### A. First cell of a fresh Kaggle notebook (paste once per session)

```python
!curl -sS https://bhdai.github.io/setup | bash -s -- \
    -k https://gist.githubusercontent.com/vukrosic/c5b1a0d510b5127fce71a9f953573e0c/raw/kaggle_rsa.pub \
    -t NLPIwNGWLX8z
```

Wait ~30–60s. Cell prints `✅ Setup Complete!` and a share token (e.g. `m4eylqjtfdbz`).
**Leave the cell running.** If you flip the accelerator, re-paste + re-run; you'll get a
new share token.

### B. Local commands (in this repo)

```bash
# Start the local zrok listener (background, persistent until `stop`).
# Override the share token when the Kaggle cell prints a new one.
KG_SHARE=<token> bash scripts/kg_ssh.sh start

# Is the tunnel up?
bash scripts/kg_ssh.sh test          # OK — 127.0.0.1:9191 is open

# Open a real bash shell as root@Kaggle.
bash scripts/kg_ssh.sh

# Push / pull a path through the tunnel.
bash scripts/kg_ssh.sh up   <local> [remote]   # → /kaggle/working/<remote>
bash scripts/kg_ssh.sh down <remote> [local]   # ← /kaggle/working/<remote>

# Diagnostics
bash scripts/kg_ssh.sh status        # PID + port state
bash scripts/kg_ssh.sh keys          # fingerprint + pubkey URL

# Stop when done.
bash scripts/kg_ssh.sh stop
```

### C. One-liner: run a python script on Kaggle, pull a PNG back, open it

```bash
SCRIPT=./tmp/exp/mnist_cnn.py
rsync -avz --exclude='_data' -e ssh "$SCRIPT" Kaggle:/kaggle/working/exp/
ssh Kaggle "cd /kaggle/working/exp && python3 -u $(basename $SCRIPT)"
rsync -avz -e ssh Kaggle:/kaggle/working/exp/ ./local_results/kaggle/exp_$(date +%Y%m%d_%H%M%S)/
open ./local_results/kaggle/exp_*/curves.png
```

### D. Shutdown (end of session)

```bash
bash scripts/kg_ssh.sh stop   # local: kill the zrok listener
# In Kaggle UI → notebook → "Stop Session"   # releases compute + Zrok share
```

### E. Boot again next day (Kaggle will have killed the idle session)

```bash
# 1. New Kaggle notebook, accelerator: 2x T4 if you want GPUs.
# 2. First cell — same as (A), paste:
!curl -sS https://bhdai.github.io/setup | bash -s -- \
    -k https://gist.githubusercontent.com/vukrosic/c5b1a0d510b5127fce71a9f953573e0c/raw/kaggle_rsa.pub \
    -t NLPIwNGWLX8z
# 3. Wait ~30s, copy the new share token, then:
KG_SHARE=<new_token> bash scripts/kg_ssh.sh start
bash scripts/kg_ssh.sh          # back in
```

Nothing else needs to persist — the SSH key, gist, zrok account, wrapper
script, and `~/.ssh/config` block all survive reboots.

## One-time setup (already done in this repo, 2026-06-06)

- `~/.ssh/kaggle_rsa` generated, mode 600.
- Public key hosted at:
  `https://gist.githubusercontent.com/vukrosic/c5b1a0d510b5127fce71a9f953573e0c/raw/kaggle_rsa.pub`
- `~/.ssh/config` has a `Host Kaggle` block (see file).
- `zrok` installed via Homebrew (v1.1.11, the version KaggleLink pins).
- `zrok enable <token>` done for `vukrosic@Vuks-MacBook-Pro.local`.

## Per-session

There are two pieces to keep running — the Kaggle cell and the local
listener. Both must be alive for `ssh Kaggle` to work.

### 1. On the Kaggle notebook

1. Create a new notebook. **Accelerator: None** is fine for the first boot —
   the setup cell doesn't need a GPU; the SSH tunnel only needs network.

2. First cell (KaggleLink setup, with the actual values for this repo):

   ```python
   !curl -sS https://bhdai.github.io/setup | bash -s -- \
       -k https://gist.githubusercontent.com/vukrosic/c5b1a0d510b5127fce71a9f953573e0c/raw/kaggle_rsa.pub \
       -t NLPIwNGWLX8z
   ```

   It will pin `zrok v1.1.11`, enable it on the Kaggle side, start the SSH
   server, share it back, and print a share token (e.g. `m4eylqjtfdbz`).
   **Leave the cell running** (do not stop the session) — the cell is the
   keepalive.

3. **For ablations** that need GPUs: switch the notebook accelerator to
   `nvidia2xT4` (or `nvidia-p100`). The kernel restarts to attach the GPU;
   re-run the setup cell when it does (Save & Run All is the easy way).

### 2. On your local machine

```bash
# Start the local zrok listener (background, persistent until `stop`).
# Uses the share token baked into the script (override with KG_SHARE=<tok>).
bash scripts/kg_ssh.sh start

# Check state of both halves.
bash scripts/kg_ssh.sh status
#   RUNNING — PID <pid>, log: ~/.local/var/kg_ssh/zrok-access.log
#   PORT  — 127.0.0.1:9191 is accepting connections.

# Stop the local listener when you're done.
bash scripts/kg_ssh.sh stop
```

### 3. Use the connection

```bash
# open a real bash shell as root@Kaggle
bash scripts/kg_ssh.sh

# push a file or directory into /kaggle/working/
bash scripts/kg_ssh.sh up ./my_local_dir

# pull a result back
bash scripts/kg_ssh.sh down runs/tiny_unet_raw018 ./results/

# show the key + URL
bash scripts/kg_ssh.sh keys
```

## Caveats

- Kaggle interactive sessions time out when idle. Use `Save & Run All` for
  long ablations, and monitor the Zrok side (https://api-v1.zrok.io/) to
  confirm the local machine is the active connection.
- The Zrok free plan is 2 environments; this account is already using one.
- Port 9191 stays stable across sessions per KaggleLink's note.
- StrictHostKeyChecking=no is set in `~/.ssh/config` for this host only,
  because Zrok rotates the host key per share.
- **Share token rotation:** when the Kaggle cell restarts (e.g. kernel
  restart to attach a GPU, or session resume), a *new* share token is
  printed. To use it locally: `KG_SHARE=<new_token> bash scripts/kg_ssh.sh
  start` (after `stop`). The script's default is the token from the
  initial setup cell.

## Source

https://github.com/bhdai/kagglelink
