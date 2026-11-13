
import glob
import os
import sysconfig
import random

stdlib = sysconfig.get_paths()["stdlib"]
paths = sorted(glob.glob(
    os.path.join(stdlib, "**", "*.py"),
    recursive=True
))

# 파일 단위로 셔플 후 90:10 분리
random.seed(42)
random.shuffle(paths)

n_val = int(len(paths) * 0.1)
val_paths = paths[:n_val]
train_paths = paths[n_val:]


def save_files(paths, output_path):
    count = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for p in paths:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    text = f.read()

                out.write(text)
                out.write("\n\n")
                count += 1

            except (OSError, UnicodeError):
                continue

    size = os.path.getsize(output_path) / 1e6
    print(f"{output_path}: {count} files, {size:.1f} MB")


save_files(train_paths, "train.txt")
save_files(val_paths, "val.txt")
