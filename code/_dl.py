"""测试 hf-mirror 能否替 tabpfn 拉到权重。"""
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from huggingface_hub import hf_hub_download

for repo, fn in [
    ("Prior-Labs/TabPFN-v2-clf", "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"),
    ("Prior-Labs/tabpfn_2_5", "tabpfn-v2.5-classifier-v2.5_default.ckpt"),
]:
    try:
        p = hf_hub_download(repo_id=repo, filename=fn)
        print("OK", repo, fn, "->", p)
    except Exception as exc:  # noqa: BLE001
        print("FAIL", repo, fn, repr(exc)[:300])
