import random
from pathlib import Path

for i in range(3):
    dataset_dir = Path("/home/liuyang/datasets/yanshou_video/")
    video_list = [str(f) for f in dataset_dir.iterdir()]

    selected_videos = random.sample(video_list, 10)

    with open(f"selected_videos_self_{i}.txt", "w") as f:
        for v in selected_videos:
            f.write(v)
            f.write("\n")

    dataset_dir = Path("/home/liuyang/datasets/yanshou_video_public/")
    video_list = [str(f) for f in dataset_dir.iterdir()]

    selected_videos = random.sample(video_list, 10)

    with open(f"selected_videos_public-{i}.txt", "w") as f:
        for v in selected_videos:
            f.write(v)
            f.write("\n")
