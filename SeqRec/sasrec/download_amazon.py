import os
import gzip
import shutil
import requests
from tqdm import tqdm

BASE_2018_REVIEW = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/"
BASE_2018_META   = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/"

BASE_2023_REVIEW = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/"
BASE_2023_META   = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/"

def download_file(url, out_path, timeout=30):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 最终文件存在，直接跳过
    if os.path.exists(out_path):
        print(f"[√] Exist: {out_path}")
        return

    tmp_path = out_path + ".part"

    print(f"[↓] Downloading {url}")
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()

    total = int(r.headers.get("content-length", 0) or 0)
    with open(tmp_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True
    ) as pbar:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    # 下载完成才替换
    os.replace(tmp_path, out_path)

def is_valid_gzip(gz_path):
    try:
        with gzip.open(gz_path, "rb") as f:
            while f.read(1024 * 1024):
                pass
        return True
    except Exception:
        return False


def unzip_gz(gz_path):
    out_path = gz_path[:-3]

    # 已解压
    if os.path.exists(out_path):
        print(f"[√] Extracted exists: {out_path}")
        return out_path

    if not os.path.exists(gz_path):
        raise FileNotFoundError(f"Missing gz file: {gz_path}")

    # gzip 损坏，删除并报错
    if not is_valid_gzip(gz_path):
        print(f"[!] Corrupted gzip detected, removing: {gz_path}")
        os.remove(gz_path)
        raise RuntimeError(f"Corrupted gzip file: {gz_path}")

    print(f"[↪] Extracting {gz_path}")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    return out_path

def ensure_amazon_2018(category, root="data/amazon_2018"):
    raw_dir = os.path.join(root, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    review_path = os.path.join(raw_dir, f"{category}.json")
    meta_path   = os.path.join(raw_dir, f"meta_{category}.json")

    # 最终文件存在，直接返回
    if os.path.exists(review_path) and os.path.exists(meta_path):
        print(f"[√] Amazon 2018 {category} already prepared.")
        return review_path, meta_path

    review_gz = review_path + ".gz"
    meta_gz   = meta_path + ".gz"

    download_file(BASE_2018_REVIEW + os.path.basename(review_gz), review_gz)
    download_file(BASE_2018_META   + os.path.basename(meta_gz), meta_gz)

    review_path = unzip_gz(review_gz)
    meta_path   = unzip_gz(meta_gz)

    return review_path, meta_path


def ensure_amazon_2023(category, root="data/amazon_2023"):
    raw_dir = os.path.join(root, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    review_path = os.path.join(raw_dir, f"{category}.jsonl")
    meta_path   = os.path.join(raw_dir, f"meta_{category}.jsonl")

    # 最终文件存在，直接返回
    if os.path.exists(review_path) and os.path.exists(meta_path):
        print(f"[√] Amazon 2023 {category} already prepared.")
        return review_path, meta_path

    review_gz = review_path + ".gz"
    meta_gz   = meta_path + ".gz"

    download_file(BASE_2023_REVIEW + os.path.basename(review_gz), review_gz)
    download_file(BASE_2023_META   + os.path.basename(meta_gz), meta_gz)

    review_path = unzip_gz(review_gz)
    meta_path   = unzip_gz(meta_gz)

    return review_path, meta_path


def ensure_amazon_dataset(year, category):
    """
    year: 2018 或 2023
    category: 类别名称
    """
    if year == 2018:
        return ensure_amazon_2018(category)
    elif year == 2023:
        return ensure_amazon_2023(category)
    else:
        raise ValueError(f"Unsupported Amazon year: {year}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--category", type=str, required=True)
    args = parser.parse_args()

    paths = ensure_amazon_dataset(args.year, args.category)
    print("Done:", paths)
