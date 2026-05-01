import os
import webdataset as wds
import config

def main():
    os.makedirs("./data/wds", exist_ok=True)

    samples = []
    with open(config.CAPTIONS_PATH, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                samples.append((parts[0], parts[1]))

    pattern = "./data/wds/flickr8k-%05d.tar"

    with wds.ShardWriter(pattern, maxcount=1000) as sink:
        for idx, (img_name, caption) in enumerate(samples):
            img_path = os.path.join(config.IMAGE_DIR, img_name)
            if not os.path.exists(img_path):
                continue

            with open(img_path, 'rb') as stream:
                image_bytes = stream.read()

            sink.write({
                "__key__": f"sample_{idx:06d}",
                "jpg": image_bytes,
                "txt": caption.encode('utf-8')
            })

    print(f"Pack finished with sample number:{len(samples)}")

if __name__ == "__main__":
    main()