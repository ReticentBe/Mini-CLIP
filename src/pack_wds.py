import os
import json
import webdataset as wds
from collections import defaultdict
import config

"""
Pack raw Flickr8k into WebDataset .tar shards.

Each shard sample contains one .jpg image and one .json file with a list
of 5 captions. ShardWriter splits output into files of maxcount=1000 samples.
"""

def main():
    os.makedirs("./data/wds", exist_ok=True)

    image_to_captions = defaultdict(list)

    with open(config.CAPTIONS_PATH, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                img_name, caption = parts[0], parts[1]
                image_to_captions[img_name].append(caption)

    pattern = "./data/wds/flickr8k-%05d.tar"

    with wds.ShardWriter(pattern, maxcount=1000) as sink:
        for idx, (img_name, captions_list) in enumerate(image_to_captions.items()):
            img_path = os.path.join(config.IMAGE_DIR, img_name)
            if not os.path.exists(img_path):
                continue

            with open(img_path, 'rb') as stream:
                image_bytes = stream.read()

            sink.write({
                "__key__": f"sample_{idx:06d}",
                "jpg": image_bytes,
                "json": captions_list
            })

    print(f"Pack finished with sample number:{len(image_to_captions)}")

if __name__ == "__main__":
    main()