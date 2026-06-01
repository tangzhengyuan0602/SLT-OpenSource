# Modified from https://github.com/NJU-PCALab/OpenVid-1M/blob/main/download_scripts/download_OpenVid.py
import os
import subprocess
import argparse
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def _has_cmd(cmd: str) -> bool:
    try:
        return shutil.which(cmd) is not None
    except Exception:
        return False


def _run(cmd, *, check=True, quiet: bool = False):
    """Run a subprocess.

    Note: When running this script in background (e.g., remote job / CI), very
    verbose commands like `unzip` can write huge logs and the stdout/stderr pipe
    may be closed by the caller, causing the child process to crash with SIGPIPE.
    Set `quiet=True` to redirect outputs to DEVNULL.
    """

    printable = " ".join([str(c) for c in cmd])
    print(f"[cmd] {printable}")
    if quiet:
        return subprocess.run(cmd, check=check, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return subprocess.run(cmd, check=check)


def _download(url: str, dst_path: str, *, retries: int = 3, timeout_sec: int = 30):
    """Download with resume + retries.

    - Prefer aria2c if available.
    - Fallback to wget.
    """

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            if _has_cmd("aria2c"):
                # -c: continue, -x/-s: multi-connection, --auto-file-renaming=false keeps name stable
                _run([
                    "aria2c",
                    "-c",
                    "-x", "8",
                    "-s", "8",
                    "--auto-file-renaming=false",
                    "--allow-overwrite=true",
                    "--timeout", str(timeout_sec),
                    "--connect-timeout", str(timeout_sec),
                    "-o", os.path.basename(dst_path),
                    "-d", os.path.dirname(dst_path),
                    url,
                ])
            else:
                # -c: resume, --tries: retry, --timeout: socket timeout
                _run([
                    "wget",
                    "-c",
                    "--progress=dot:giga",
                    "--no-verbose",
                    "--tries=3",
                    f"--timeout={timeout_sec}",
                    "-O", dst_path,
                    url,
                ])

            # Basic sanity: non-empty file.
            if not os.path.exists(dst_path) or os.path.getsize(dst_path) == 0:
                raise RuntimeError(f"downloaded file is empty: {dst_path}")

            return
        except Exception as e:
            print(f"[warn] download failed (attempt {attempt}/{retries}): {url} -> {dst_path} | {e}")
            if attempt >= retries:
                raise


def _zip_test(zip_path: str) -> bool:
    """Return True if zip looks OK."""
    try:
        _run(["unzip", "-tq", zip_path], check=True, quiet=True)
        return True
    except Exception:
        return False

def download_files(
    output_directory,
    download_pose,
    download_label: bool = True,
    keep_zip: bool = True,
    max_rgb_archives: int = 436,
    max_pose_archives: int = 46,
    rgb_start: int = 1,
    pose_start: int = 1,
    retries: int = 3,
    base_url: str = "https://hf-mirror.com",
    jobs: int = 1,
):
    """Download CSL-News RGB and (optional) pose archives.

    Expected directory layout (matches docs/DATASET.md & config.py):
    <output_directory>/rgb_format/*.mp4
    <output_directory>/pose_format/*.pkl
    """

    output_directory = str(output_directory)
    RGB_zip_folder = os.path.join(output_directory, "RGB_download")
    video_folder = os.path.join(output_directory, "rgb_format")
    os.makedirs(RGB_zip_folder, exist_ok=True)
    os.makedirs(video_folder, exist_ok=True)

    RGB_error_log_path = os.path.join(RGB_zip_folder, "download_log.txt")
    
    base_url = str(base_url).rstrip("/")

    # Download RGB format
    rgb_start = int(rgb_start)
    max_rgb_archives = int(max_rgb_archives)
    if rgb_start < 1:
        rgb_start = 1
    jobs = int(jobs) if jobs is not None else 1
    if jobs < 1:
        jobs = 1

    def _resolve_rgb_paths(i: int):
        url = f"{base_url}/datasets/ZechengLi19/CSL-News/resolve/main/archive_{i:03d}.zip"
        padded_name = f"archive_{i:03d}.zip"
        legacy_name = f"archive_{i}.zip"  # backward compat
        padded_path = os.path.join(RGB_zip_folder, padded_name)
        legacy_path = os.path.join(RGB_zip_folder, legacy_name)
        file_path = padded_path
        if (not os.path.exists(file_path)) and os.path.exists(legacy_path):
            file_path = legacy_path
        marker = os.path.join(RGB_zip_folder, f".extracted_{i:03d}.ok")
        return url, file_path, marker

    def _ensure_zip_present(url: str, file_path: str):
        # Keep resume working: if zip exists (even partial), don't force re-download here.
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            return
        _download(url, file_path, retries=retries)

    # Build todo list (skip extracted ones).
    rgb_todo = []
    for i in range(rgb_start, max_rgb_archives + 1):
        _, _, marker = _resolve_rgb_paths(i)
        if os.path.exists(marker):
            print(f"[{i:03d}/{max_rgb_archives:03d}] extracted marker exists, skip")
            continue
        rgb_todo.append(i)

    # Download in parallel (zip-level), then unzip sequentially per batch.
    for start in range(0, len(rgb_todo), jobs):
        batch = rgb_todo[start:start + jobs]
        futures = {}
        for i in batch:
            url, file_path, marker = _resolve_rgb_paths(i)
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                continue
            print(f"[{i:03d}/{max_rgb_archives:03d}] (parallel) download rgb zip")
            futures[i] = (url, file_path, marker)

        if futures:
            with ThreadPoolExecutor(max_workers=min(jobs, len(futures))) as ex:
                fut_map = {
                    ex.submit(_ensure_zip_present, url, file_path): (i, url, file_path)
                    for i, (url, file_path, marker) in futures.items()
                }
                for fut in as_completed(fut_map):
                    i, url, file_path = fut_map[fut]
                    try:
                        fut.result()
                    except Exception as e:
                        error_message = f"file {url} download failed: {e}\n"
                        print(error_message)
                        with open(RGB_error_log_path, "a") as error_log_file:
                            error_log_file.write(error_message)

        # Unzip sequentially.
        for i in batch:
            url, file_path, marker = _resolve_rgb_paths(i)
            if os.path.exists(marker):
                continue
            if (not os.path.exists(file_path)) or os.path.getsize(file_path) == 0:
                continue
            try:
                print(f"[{i:03d}/{max_rgb_archives:03d}] unzip rgb zip")
                _run(["unzip", "-j", "-o", file_path, "-d", video_folder], check=True, quiet=True)
                Path(marker).touch()
                if not keep_zip:
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass
            except Exception as e:
                # unzip failure often indicates corrupted/partial zip. Re-download once and retry.
                print(f"[warn] unzip failed, will re-download once: {file_path} | {e}")
                try:
                    os.remove(file_path)
                except Exception:
                    pass
                try:
                    _download(url, file_path, retries=retries)
                    _run(["unzip", "-j", "-o", file_path, "-d", video_folder], check=True, quiet=True)
                    Path(marker).touch()
                    if not keep_zip:
                        try:
                            os.remove(file_path)
                        except Exception:
                            pass
                except Exception as e2:
                    error_message = f"file {url} unzip failed after re-download: {e2}\n"
                    print(error_message)
                    with open(RGB_error_log_path, "a") as error_log_file:
                        error_log_file.write(error_message)
    
    # Download pose format (Optional)
    if download_pose:
        pose_zip_folder = os.path.join(output_directory, "pose_download")
        pose_folder = os.path.join(output_directory, "pose_format")
        os.makedirs(pose_zip_folder, exist_ok=True)
        os.makedirs(pose_folder, exist_ok=True)

        pose_error_log_path = os.path.join(pose_zip_folder, "download_log.txt")
        
        pose_start = int(pose_start)
        max_pose_archives = int(max_pose_archives)
        jobs = int(jobs) if jobs is not None else 1
        if jobs < 1:
            jobs = 1
        if pose_start < 1:
            pose_start = 1

        # Build todo list (skip extracted ones).
        todo = []
        for i in range(pose_start, max_pose_archives + 1):
            marker = os.path.join(pose_zip_folder, f".extracted_{i:03d}.ok")
            if os.path.exists(marker):
                print(f"[{i:03d}/{max_pose_archives:03d}] extracted marker exists, skip")
                continue
            todo.append(i)

        def _resolve_paths(i: int):
            url = f"{base_url}/datasets/ZechengLi19/CSL-News_pose/resolve/main/archive_{i:03d}.zip"
            padded_name = f"archive_{i:03d}.zip"
            legacy_name = f"archive_{i}.zip"
            padded_path = os.path.join(pose_zip_folder, padded_name)
            legacy_path = os.path.join(pose_zip_folder, legacy_name)
            file_path = padded_path
            if (not os.path.exists(file_path)) and os.path.exists(legacy_path):
                file_path = legacy_path
            marker = os.path.join(pose_zip_folder, f".extracted_{i:03d}.ok")
            return url, file_path, padded_path, marker

        def _ensure_zip_present(url: str, file_path: str):
            # We rely on unzip failure to detect corruption. This avoids scanning
            # multi-GB zips twice (zip test + unzip).
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                return
            _download(url, file_path, retries=retries)

        # Download in parallel (zip-level), then unzip sequentially to reduce IO contention.
        for start in range(0, len(todo), jobs):
            batch = todo[start:start + jobs]

            futures = {}
            for i in batch:
                url, file_path, padded_path, marker = _resolve_paths(i)

                # If zip exists (even partial), we will attempt unzip later; wget -c can resume if needed.
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    continue

                print(f"[{i:03d}/{max_pose_archives:03d}] (parallel) download pose zip")
                futures[i] = (url, file_path, padded_path, marker)

            if futures:
                with ThreadPoolExecutor(max_workers=min(jobs, len(futures))) as ex:
                    fut_map = {
                        ex.submit(_ensure_zip_present, url, file_path): (i, url, file_path)
                        for i, (url, file_path, padded_path, marker) in futures.items()
                    }
                    for fut in as_completed(fut_map):
                        i, url, file_path = fut_map[fut]
                        try:
                            fut.result()
                        except Exception as e:
                            error_message = f"file {url} download failed: {e}\n"
                            print(error_message)
                            with open(pose_error_log_path, "a") as error_log_file:
                                error_log_file.write(error_message)

            # Unzip sequentially for this batch.
            for i in batch:
                url, file_path, padded_path, marker = _resolve_paths(i)
                if os.path.exists(marker):
                    continue
                if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                    continue
                try:
                    print(f"[{i:03d}/{max_pose_archives:03d}] unzip pose zip")
                    _run(["unzip", "-j", "-o", file_path, "-d", pose_folder], check=True, quiet=True)
                    Path(marker).touch()
                    if not keep_zip:
                        try:
                            os.remove(file_path)
                        except Exception:
                            pass
                except Exception as e:
                    # unzip failure often indicates corrupted/partial zip. Re-download once and retry.
                    print(f"[warn] unzip failed, will re-download once: {file_path} | {e}")
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass
                    try:
                        _download(url, file_path, retries=retries)
                        _run(["unzip", "-j", "-o", file_path, "-d", pose_folder], check=True, quiet=True)
                        Path(marker).touch()
                        if not keep_zip:
                            try:
                                os.remove(file_path)
                            except Exception:
                                pass
                    except Exception as e2:
                        error_message = f"file {url} unzip failed after re-download: {e2}\n"
                        print(error_message)
                        with open(pose_error_log_path, "a") as error_log_file:
                            error_log_file.write(error_message)
        
    # download label (optional)
    # config.py expects: ./data/CSL_News/CSL_News_Labels.json
    if download_label:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        data_folder = os.path.join(repo_root, "data", "CSL_News")
        os.makedirs(data_folder, exist_ok=True)
        data_url = f"{base_url}/datasets/ZechengLi19/CSL-News/resolve/main/data/train/CSL_News_Labels.json"
        data_path = os.path.join(data_folder, "CSL_News_Labels.json")
        if not os.path.exists(data_path) or os.path.getsize(data_path) == 0:
            _download(data_url, data_path, retries=retries)
        else:
            print(f"label exists, skip: {data_path}")

    # optionally delete zip files to save disk
    if not keep_zip:
        try:
            subprocess.run(["rm", "-rf", RGB_zip_folder], check=True)
        except Exception:
            pass
        if download_pose:
            try:
                subprocess.run(["rm", "-rf", os.path.join(output_directory, "pose_download")], check=True)
            except Exception:
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument(
        '--output_directory',
        type=str,
        help='Path to the CSL-News dataset directory (will create rgb_format/pose_format inside).',
        default=str((Path(__file__).resolve().parents[1] / "dataset" / "CSL_News").resolve()),
    )
    parser.add_argument('--download_pose', action='store_true', help='Whether to download pose or not')
    parser.add_argument('--max_rgb_archives', type=int, default=436, help='Max number of RGB zip archives to download (default: 436)')
    parser.add_argument('--max_pose_archives', type=int, default=46, help='Max number of pose zip archives to download (default: 46)')
    parser.add_argument('--rgb_start', type=int, default=1, help='Start index (1-based) for RGB archives')
    parser.add_argument('--pose_start', type=int, default=1, help='Start index (1-based) for pose archives')
    parser.add_argument('--retries', type=int, default=3, help='Retry times per archive')
    parser.add_argument('--jobs', type=int, default=1, help='Parallel download jobs (zip-level). Unzip is still sequential per batch.')
    parser.add_argument(
        '--base_url',
        type=str,
        default='https://hf-mirror.com',
        help='HuggingFace base URL. In CN networks, https://hf-mirror.com is usually faster/accessible.',
    )
    parser.add_argument('--no_download_label', action='store_true', help='Do not download CSL_News_Labels.json into ./data/CSL_News')
    parser.add_argument('--delete_zip', action='store_true', help='Delete downloaded zip folders after extraction')
    args = parser.parse_args()
    download_files(
        args.output_directory,
        args.download_pose,
        download_label=(not args.no_download_label),
        keep_zip=(not args.delete_zip),
        max_rgb_archives=args.max_rgb_archives,
        max_pose_archives=args.max_pose_archives,
        rgb_start=args.rgb_start,
        pose_start=args.pose_start,
        retries=args.retries,
        base_url=args.base_url,
        jobs=args.jobs,
    )
