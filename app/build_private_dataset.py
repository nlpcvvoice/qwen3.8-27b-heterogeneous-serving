import os, json, sys, glob, traceback
from kaggle.api.kaggle_api_extended import KaggleApi

USER = "tentenshishi"
SLUG = "qwen3-8-27b-bf16-private"
RELEASE = "/kaggle/working/release"
LOG = "/kaggle/working/build.log"


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(str(msg) + "\n")


def write_done(text):
    with open("/kaggle/working/DONE.txt", "w") as f:
        f.write(text)


def write_err(text):
    with open("/kaggle/working/ERROR.txt", "w") as f:
        f.write(text)


try:
    hits = glob.glob("/kaggle/input/qwen3-8-27b-bf16") + \
           glob.glob("/kaggle/input/datasets/rahim3/qwen3-8-27b-bf16")
    if not hits:
        raise RuntimeError("weights dataset not found: " +
                           str(glob.glob("/kaggle/input/**/qwen3-8-27b-bf16", recursive=True)))
    SRC = hits[0]
    log("using source mount: " + SRC)

    os.makedirs(RELEASE, exist_ok=True)
    files = sorted(os.listdir(SRC))
    log("input top-level files: " + str(len(files)))
    log("has config.json: " + str(os.path.exists(os.path.join(SRC, "config.json"))))
    total = 0
    for fn in files:
        fp = os.path.join(SRC, fn)
        if os.path.isfile(fp):
            ln = os.path.join(RELEASE, fn)
            if not os.path.exists(ln):
                os.symlink(fp, ln)
            total += os.path.getsize(fp)
    log("release symlinks: " + str(len(os.listdir(RELEASE))) +
        "  total bytes: " + str(total))

    meta = {"id": f"{USER}/{SLUG}",
            "title": "Qwen3.8-27B bf16 private mirror",
            "licenses": [{"name": "Apache 2.0"}],
            "description": "Private mirror of rahim3/qwen3-8-27b-bf16 (Qwen3.8-27B, Apache-2.0). "
                           "Used by kaggle-tpu-lab to avoid a network weight download at startup."}
    with open(os.path.join(RELEASE, "dataset-metadata.json"), "w") as f:
        json.dump(meta, f)

    api = KaggleApi()
    try:
        api.authenticate()
        log("authenticated via platform implicit creds")
    except Exception as e:
        log("implicit auth failed: " + str(e)[:300])
        tok = os.environ.get("KAGGLE_API_TOKEN")
        if tok:
            api = KaggleApi()
            api.set_config_value(api.CONFIG_NAME_TOKEN, tok)
            api.set_config_value(api.CONFIG_NAME_USER, USER)
            api.authenticate()
            log("authenticated via KAGGLE_API_TOKEN env")
        else:
            raise RuntimeError("no kaggle auth available in kernel")

    try:
        resp = api.dataset_create_new(RELEASE, public=False, quiet=True, convert_to_csv=False)
        log("dataset_create_new returned: " + json.dumps(resp.to_dict() if hasattr(resp, 'to_dict') else str(resp)))
        write_done("created id=tentenshishi/" + SLUG)
        log("SUCCESS (dataset create request accepted)")
    except Exception as e:
        log("dataset_create_new error: " + repr(e))
        try:
            import requests
            log("possibly already exists -> trying create_version")
            resp2 = api.dataset_create_version(RELEASE, version_notes="update", quiet=True,
                                               convert_to_csv=False)
            log("dataset_create_version returned: " +
                json.dumps(resp2.to_dict() if hasattr(resp2, 'to_dict') else str(resp2)))
            write_done("updated id=tentenshishi/" + SLUG)
            log("SUCCESS (dataset updated)")
        except Exception as e2:
            write_err("create failed: " + repr(e) + "\nversion failed: " + repr(e2))
            log("VERSION ALSO FAILED")
except Exception:
    tb = traceback.format_exc()
    try:
        write_err(tb)
    except Exception:
        pass
    log(tb)
    sys.exit(1)
