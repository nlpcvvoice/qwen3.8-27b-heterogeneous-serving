# P3 engine-cache

Cache the `llama-server` binary so GPU kernels skip the ~25 min source build.

```
┌─────────────────────┐   upload binary   ┌─────────────────────────────────┐  auto-mount on push  ┌──────────────────────────────┐
│  machine with GPU    │ ────────────────→ │  Private Dataset                │ ←──────────────────── │  GPU kernel (serve_qwen38_gpu) │
│  builds llama-server │  upload_cache    │  llama-server-qwen38-cache       │  dataset_sources      │  cache-hit → skip 25-min build │
└─────────────────────┘                   └─────────────────────────────────┘                      └──────────────────────────────┘
         │                                          ↑                                              │
         │ kaggle kernels output                   │ download                                     identity
         └────→ binary from kernel output          │                                                ↓
                                                   │                                    /kaggle/input/.../llama-server
                                                   │                                     (skip 25-min build)
                                          ┌─────────────────────┐
                                          │  any machine / agent │
                                          │  start.sh fetch      │
                                          └─────────────────────┘
```

## One-command entrypoint (start.sh)

| Command | Purpose | Quota |
|---|---|---|
| `./start.sh test` | full local validation (kernel-decision simulation + static asserts) | 0 |
| `./start.sh static` | static only (kernel/script hook integrity) | 0 |
| `./start.sh status` | cache dataset status (private / files / size) | 0 (read) |
| `./start.sh upload BIN` | upload llama-server binary → private cache dataset | dataset quota |
| `./start.sh fetch` | download cached llama-server and verify SHA | 0 (read) |
| `./start.sh live-test` | real Kaggle end-to-end (requires API token) | dataset quota |

## Local validation output (test)

```
[P3 validate] engine-cache full test
  [local] cache dataset mounted      -> engine=llama-server (cache-hit)  OK
  [local] no cache dataset           -> source-build                    OK
  [local] tiny/garbage binary        -> rejected -> source-build        OK
  [local] force no-cache flag        -> source-build                   OK
  [local] datasets/*/<slug> layout   -> cache-hit                      OK
  [static] kernel  : skip-compile hook + else-build + cache-hit publish   OK
  [static] push    : existence probe + dataset_sources injection        OK
  [static] kernel  : py_compile clean                                     OK
[P3 validate] ALL PASS
```

## Production flow

```
step  action                                output
────────────────────────────────────────────────────────────────────────────
1  run the GPU kernel without a cache hit   llama-server binary
2  pull `kaggle kernels output` to tmp/     kernel output
3  ./start.sh upload tmp/llama-server       private cache dataset
4  later pushes auto-detect the dataset     kernel skips build, <2 min start
```

## Layout

```
p3/
├── start.sh                one-command entrypoint
├── requirements.txt        dependencies
├── validate_cache.py       full validation (local + optional live)
├── upload_cache.py         binary upload
├── fetch_cache.py          download + verify
├── status.py               dataset status
├── common.py               shared constants + helpers
├── kaggle_login.py         auth (in-memory, no plaintext)
├── kernel/                 kernel copy (includes cache-hit logic)
│   └── serve_qwen38_gpu.py
└── reference/
    └── API-Token           auth token (never committed)
```

## Copy to a new machine

1. Copy `p3/` + `reference/API-Token` to the target machine.
2. `pip install -r requirements.txt`.
3. `./start.sh test` → verify.
4. `./start.sh fetch` → pull the cached binary.

## Constraints

- `reference/API-Token` is read-only, in-memory, never printed.
- A cache hit only copies the binary; never writes to `/kaggle/working`.
- All pushes to Kaggle must use `is_private=true`.
- `upload_cache` only mutates the private dataset.