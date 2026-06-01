from pathlib import Path

# Make paths independent of current working directory.
_REPO_ROOT = Path(__file__).resolve().parent


def _rel(p: str) -> str:
    return str((_REPO_ROOT / p).resolve())


def _env_or_rel(env_name: str, relative_path: str) -> str:
    return os.environ.get(env_name, _rel(relative_path))


mt5_path = _rel("pretrained_weight/mt5-base")
_csl_news_label_path = _env_or_rel("UNI_SIGN_CSL_NEWS_LABEL_PATH", "data/CSL_News/CSL_News_Labels.json")

# label paths
train_label_paths = {
                    "CSL_News": _csl_news_label_path,
                    "CSL_Daily": _rel("data/CSL_Daily/labels.train"),
                    "CE-CSL": _rel("dataset/CE-CSL/CE-CSL/label/train.csv"),
                    "WLASL": _rel("data/WLASL/labels-2000.train"),
                    "How2Sign": _rel("data/How2Sign/labels.train"),
                    "OpenASL": _rel("data/OpenASL/labels.train"),
                    }

dev_label_paths = {
                    "CSL_News": _csl_news_label_path,
                    "CSL_Daily": _rel("data/CSL_Daily/labels.dev"),
                    "CE-CSL": _rel("dataset/CE-CSL/CE-CSL/label/dev.csv"),
                    "WLASL": _rel("data/WLASL/labels-2000.dev"),
                    "How2Sign": "",
                    "OpenASL": _rel("data/OpenASL/labels.dev"),
                    }

test_label_paths = {
                    "CSL_News": _csl_news_label_path,
                    "CSL_Daily": _rel("data/CSL_Daily/labels.test"),
                    "CE-CSL": _rel("dataset/CE-CSL/CE-CSL/label/test.csv"),
                    "WLASL": _rel("data/WLASL/labels-2000.test"),
                    "How2Sign": _rel("data/How2Sign/labels.test"),
                    "OpenASL": _rel("data/OpenASL/labels.test"),
}


# video paths
rgb_dirs = {
            # Recommended to place CSL-News on /tmp (large local NVMe) to avoid filling rootfs.
            "CSL_News": _env_or_rel("UNI_SIGN_CSL_NEWS_RGB_DIR", "dataset/CSL_News/rgb_format"),
            "CSL_Daily": _rel('dataset/CSL_Daily/sentence-crop'),
            # CE-CSL videos are organized as CE-CSL/video/{train,dev,test}/{Translator}/*.mp4
            "CE-CSL": _rel('dataset/CE-CSL/CE-CSL/video'),
            "WLASL": _rel("dataset/WLASL/rgb_format"),
            "How2Sign": _rel("dataset/How2Sign/rgb_format"),
            "OpenASL": _rel("dataset/OpenASL/rgb_format"),
            }

# pose paths
pose_dirs = {
            "CSL_News": _env_or_rel("UNI_SIGN_CSL_NEWS_POSE_DIR", "dataset/CSL_News/pose_format"),
            "CSL_Daily": _rel('dataset/CSL_Daily/pose_format'),
            # Expected pose path: ./dataset/CE-CSL/pose_format/{train,dev,test}/{Translator}/*.pkl
            "CE-CSL": _rel('dataset/CE-CSL/pose_format'),
            "WLASL": _rel("dataset/WLASL/pose_format"),
            "How2Sign": _rel("dataset/WLASL/pose_format"),
            "OpenASL": _rel("dataset/WLASL/pose_format"),
}
