import sys
import types

import pytest

torch = pytest.importorskip("torch")

import comfy.utils


class FakeStreamer:
    streamed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def stream_file(self, path):
        self.streamed.append(("file", path))

    def stream_files(self, paths):
        self.streamed.append(("files", tuple(paths)))

    def get_tensors(self):
        return [("weight", torch.ones(1))]


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self, size=-1):
        return self.data


def patch_autumn_loader(monkeypatch, endpoint):
    module = types.SimpleNamespace(
        SafetensorsStreamer=FakeStreamer,
        list_safetensors=lambda path: [path.rstrip("/") + "/part-00001.safetensors"],
    )
    monkeypatch.setitem(sys.modules, "runai_model_streamer", module)
    monkeypatch.setenv("AWS_ENDPOINT_URL", endpoint)
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("ALL_PROXY", raising=False)
    monkeypatch.delenv("http_proxy", raising=False)
    monkeypatch.delenv("https_proxy", raising=False)
    monkeypatch.delenv("all_proxy", raising=False)

    header = b'{"__metadata__":{"format":"pt"},"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    calls = []

    class FakeOpener:
        def open(self, req, timeout=None):
            return urlopen(req, timeout=timeout)

    def urlopen(req, timeout=None):
        calls.append((req.full_url, req.headers["Range"]))
        if req.headers["Range"] == "bytes=0-7":
            return FakeResponse(len(header).to_bytes(8, "little"))
        return FakeResponse(header)

    monkeypatch.setattr(comfy.utils.urllib.request, "build_opener", lambda *args: FakeOpener())
    return calls


def test_autumn_uri_converts_to_s3():
    assert comfy.utils.autumn_to_s3_uri("autumn://autumn/models/checkpoints/a.safetensors") == "s3://autumn/models/checkpoints/a.safetensors"
    assert comfy.utils.autumn_to_s3_uri("s3://autumn/models/checkpoints/a.safetensors") == "s3://autumn/models/checkpoints/a.safetensors"


def test_autumn_loader_requires_local_endpoint(monkeypatch):
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)

    try:
        comfy.utils.load_torch_file("autumn://autumn/models/checkpoints/a.safetensors")
    except ValueError as e:
        assert "local autumn-s3 endpoint" in str(e)
    else:
        assert False, "expected ValueError"


def test_autumn_loader_uses_runai_streamer(monkeypatch):
    FakeStreamer.streamed = []
    calls = patch_autumn_loader(monkeypatch, "http://127.0.0.1:9100")

    sd, metadata = comfy.utils.load_torch_file("autumn://autumn/models/checkpoints/a.safetensors", return_metadata=True)

    assert FakeStreamer.streamed == [("file", "s3://autumn/models/checkpoints/a.safetensors")]
    assert list(sd) == ["weight"]
    assert metadata == {"format": "pt"}
    assert calls == [
        ("http://127.0.0.1:9100/autumn/models/checkpoints/a.safetensors", "bytes=0-7"),
        ("http://127.0.0.1:9100/autumn/models/checkpoints/a.safetensors", "bytes=8-91"),
    ]


def test_autumn_loader_uses_runai_directory_listing(monkeypatch):
    FakeStreamer.streamed = []
    patch_autumn_loader(monkeypatch, "http://localhost:9100")

    sd = comfy.utils.load_torch_file("autumn://autumn/models/checkpoints/flux")

    assert FakeStreamer.streamed == [("files", ("s3://autumn/models/checkpoints/flux/part-00001.safetensors",))]
    assert list(sd) == ["weight"]
