import os
import sys
import shutil
import tempfile
import subprocess
from typing import Any, Dict, List, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, Header
from dotenv import load_dotenv
import requests
import boto3
from botocore.client import Config

# --- env ---
load_dotenv()

S3_ENDPOINT   = os.getenv('S3_ENDPOINT', 'http://localhost:9000')
S3_REGION     = os.getenv('S3_REGION', 'us-east-1')
S3_ACCESS_KEY = os.getenv('S3_ACCESS_KEY', 'minioadmin')
S3_SECRET_KEY = os.getenv('S3_SECRET_KEY', 'minioadmin')
INPUT_BUCKET  = os.getenv('S3_INPUT_BUCKET', 'facefusion-input')
OUTPUT_BUCKET = os.getenv('S3_OUTPUT_BUCKET', 'facefusion-output')

NEST_BASE_URL        = os.getenv('NEST_BASE_URL', 'http://127.0.0.1:3000')
WORKER_SHARED_SECRET = os.getenv('WORKER_SHARED_SECRET', 'supersecret')
DEFAULT_DEVICE_ID    = os.getenv('EXECUTION_DEVICE_ID', '0')

s3 = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    region_name=S3_REGION,
    config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}),
)

router = APIRouter()

def _ext_from_key(key: str, fallback: str) -> str:
    base = os.path.basename(key)
    _, ext = os.path.splitext(base)
    return ext or fallback

def _content_type_for_ext(ext: str) -> str:
    e = ext.lower()
    if e in ('.jpg', '.jpeg'): return 'image/jpeg'
    if e == '.png': return 'image/png'
    if e in ('.mp4', '.m4v'): return 'video/mp4'
    if e == '.mov': return 'video/quicktime'
    return 'application/octet-stream'

def _project_paths() -> tuple[Path, Path]:
    project_root = Path(__file__).resolve().parent
    run_py_path = project_root / 'facefusion.py'
    if not run_py_path.is_file():
        run_py_path = Path('facefusion.py')
    return project_root, run_py_path

def _diagnostics() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        import onnxruntime as ort
        info['ort_providers'] = ort.get_available_providers()
    except Exception as e:
        info['ort_error'] = str(e)
    try:
        import torch
        info['torch_cuda'] = bool(torch.cuda.is_available())
        info['torch_cuda_version'] = getattr(torch.version, 'cuda', None)
    except Exception as e:
        info['torch_error'] = str(e)
    try:
        res = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
        info['nvidia_smi'] = res.stdout if res.returncode == 0 else 'unavailable'
    except Exception:
        info['nvidia_smi'] = 'unavailable'
    print('[DIAG]', info)
    return info

def run_facefusion(
    source_paths, target_path, output_path,
    *, processors=None, face_swapper_model=None,
    force_cuda=False, execution_device_id=None, extra_args=None,
):
    project_root, run_py_path = _project_paths()

    device_id = str(execution_device_id or DEFAULT_DEVICE_ID or '0')

    # ✔ providers harus dipisah per token, bukan satu string pakai koma
    providers: list[str] = ['cpu']
    if force_cuda:
        providers = ['cuda']          # CUDA only (biar fail kalau gak tersedia)
    else:
        # kalau mau fallback: cuda lalu cpu
        providers = ['cuda', 'cpu']

    cmd = [
        sys.executable, str(run_py_path), 'headless-run',
        '--source', *source_paths,
        '--target', target_path,
        '--output-path', output_path,
        '--execution-providers', *providers,        # <— ini kuncinya
        '--execution-device-id', device_id,
    ]

    proc_list = processors or ['face_swapper']
    cmd += ['--processors', *proc_list]
    if 'face_swapper' in [p.strip() for p in proc_list]:
        cmd += ['--face-swapper-model', (face_swapper_model or 'inswapper_128')]

    if extra_args:
        cmd += [str(x) for x in extra_args]

    env = os.environ.copy()
    env.setdefault('ORT_CUDA_UNAVAILABLE_LOGGING', '1')
    env.setdefault('ORT_LOG_SEVERITY_LEVEL', '1')
    if device_id.isdigit():
        env.setdefault('CUDA_VISIBLE_DEVICES', device_id)

    result = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=str(project_root), env=env, check=False
    )
    print("== STDOUT ==\n", result.stdout)
    print("== STDERR ==\n", result.stderr)
    result.check_returncode()

def _callback(url: str, body: Dict[str, Any]):
    try:
        r = requests.post(url, json=body, headers={'X-Worker-Secret': WORKER_SHARED_SECRET}, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"[callback error] {url} -> {e}")

@router.post("/worker/facefusion")
def start_facefusion_job(
    payload: dict,
    x_worker_secret: Optional[str] = Header(None),
):
    if x_worker_secret != WORKER_SHARED_SECRET:
        raise HTTPException(status_code=401, detail="unauthorized")

    job_id: Optional[str]     = payload.get('jobId')
    source_key: Optional[str] = payload.get('sourceKey')   # pastikan camelCase di Nest
    target_key: Optional[str] = payload.get('targetKey')
    options: dict             = payload.get('options', {}) or {}

    if not job_id or not source_key or not target_key:
        raise HTTPException(status_code=400, detail="missing fields")

    processors          = options.get('processors') or ['face_swapper']
    face_swapper_model  = options.get('faceSwapperModel') or 'inswapper_128'
    use_cuda: bool      = bool(options.get('useCuda', False))
    device_id           = options.get('deviceId', None)
    extra_args          = options.get('extraArgs') or []

    print("---- FACEFUSION JOB ----")
    print("job_id        :", job_id)
    print("source_key    :", source_key)
    print("target_key    :", target_key)
    print("processors    :", processors)
    print("swapper_model :", face_swapper_model)
    print("use_cuda      :", use_cuda)
    print("device id     :", device_id if device_id is not None else DEFAULT_DEVICE_ID)
    print("------------------------")

    tmpdir = tempfile.mkdtemp(prefix=f"ff_{job_id}_")
    try:
        src_ext = _ext_from_key(source_key, ".jpg")
        tgt_ext = _ext_from_key(target_key, ".mp4")
        out_ext = ".mp4"

        source_path = os.path.join(tmpdir, f"source{src_ext}")
        target_path = os.path.join(tmpdir, f"target{tgt_ext}")
        output_path = os.path.join(tmpdir, f"output{out_ext}")

        # validasi & download
        s3.head_object(Bucket=INPUT_BUCKET, Key=source_key)
        s3.head_object(Bucket=INPUT_BUCKET, Key=target_key)
        s3.download_file(INPUT_BUCKET, source_key, source_path)
        s3.download_file(INPUT_BUCKET, target_key, target_path)

        # jalankan
        use_cuda = bool(options.get('useCuda', False))
        run_facefusion(
            [source_path], target_path, output_path,
            processors=processors,
            face_swapper_model=face_swapper_model,
            force_cuda=use_cuda,                         # true → CUDA only
            execution_device_id=str(options.get('deviceId')) if options.get('deviceId') is not None else None,
            extra_args=extra_args,
        )

        if not os.path.exists(output_path):
            raise RuntimeError("facefusion finished but output file not found")

        output_key = f"results/{job_id}/result{out_ext}"
        s3.upload_file(
            output_path, OUTPUT_BUCKET, output_key,
            ExtraArgs={'ContentType': _content_type_for_ext(out_ext)}
        )

        _callback(f"{NEST_BASE_URL}/jobs/facefusion/{job_id}/callback/done", {"output_key": output_key})
        return {"ok": True, "output_key": output_key}

    except Exception as e:
        _callback(f"{NEST_BASE_URL}/jobs/facefusion/{job_id}/callback/failed", {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)