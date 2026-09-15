"""Teach ComfyUI-Manager to fetch huggingface.co model files from ModelScope.

The Manager rewrites a model URL with HF_ENDPOINT and nothing else, so the
image cannot point its downloads at ModelScope without also pointing
huggingface_hub there — and ModelScope does not implement the Hub API
(list_repo_files returns HTML, hf_hub_download fails). This patch adds a
second, download-only mirror: COMFY_MODEL_MIRROR, tried first, with
HF_ENDPOINT as the fallback.

Applied at image build against the pinned comfyui_manager. If the upstream
source moves, the anchor stops matching and the build fails loudly rather
than silently shipping an unpatched Manager.
"""

import pathlib

from comfyui_manager.common import manager_downloader

OLD = """def download_url(model_url: str, model_dir: str, filename: str):
    if HF_ENDPOINT:
        model_url = model_url.replace('https://huggingface.co', HF_ENDPOINT)
        logging.info(f"model_url replaced by HF_ENDPOINT, new = {model_url}")
"""

NEW = '''MODEL_MIRROR = os.getenv('COMFY_MODEL_MIRROR')


def _mirror_url(model_url: str) -> str:
    """Prefer COMFY_MODEL_MIRROR for huggingface.co files, else HF_ENDPOINT.

    Measured from this cluster on one 100 MB file: ModelScope 28 MB/s,
    hf-mirror 12 MB/s. ModelScope serves the huggingface path shape verbatim
    (/<org>/<repo>/resolve/main/<file>), but it does not carry every repo, so
    a HEAD that does not come back OK falls back to HF_ENDPOINT.
    """
    if not model_url.startswith('https://huggingface.co'):
        return model_url

    if MODEL_MIRROR:
        candidate = model_url.replace('https://huggingface.co', MODEL_MIRROR)
        try:
            res = requests.head(candidate, allow_redirects=True, timeout=10)
            if res.status_code < 400:
                logging.info(f"model_url served by COMFY_MODEL_MIRROR, new = {candidate}")
                return candidate
            logging.info(
                f"COMFY_MODEL_MIRROR lacks {candidate} (HTTP {res.status_code}), falling back to HF_ENDPOINT"
            )
        except Exception as e:
            logging.info(f"COMFY_MODEL_MIRROR unreachable ({e!r}), falling back to HF_ENDPOINT")

    if HF_ENDPOINT:
        model_url = model_url.replace('https://huggingface.co', HF_ENDPOINT)
        logging.info(f"model_url replaced by HF_ENDPOINT, new = {model_url}")
    return model_url


def download_url(model_url: str, model_dir: str, filename: str):
    model_url = _mirror_url(model_url)
'''


def main() -> None:
    path = pathlib.Path(manager_downloader.__file__)
    src = path.read_text()

    if "COMFY_MODEL_MIRROR" in src:
        print(f"already patched: {path}")
        return

    count = src.count(OLD)
    if count != 1:
        raise SystemExit(
            f"cannot patch {path}: expected 1 occurrence of the download_url anchor, found {count}. "
            "comfyui_manager changed — update docker/patch-manager-model-mirror.py."
        )

    path.write_text(src.replace(OLD, NEW))
    print(f"patched: {path}")


if __name__ == "__main__":
    main()
