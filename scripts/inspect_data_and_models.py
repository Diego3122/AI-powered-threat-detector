import json,collections,pathlib,sys,hashlib,os
paths=["data/unsw_nb15_train.jsonl","data/unsw_nb15_test.jsonl","data/windows_dataset.jsonl","data/windows_dataset_sample.jsonl"]
for p in paths:
    pt=pathlib.Path(p)
    if pt.exists():
        cnt=0
        labels=collections.Counter()
        with pt.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r=json.loads(line)
                    labels.update([r.get("label")])
                except Exception:
                    pass
                cnt+=1
        print(p, "n=",cnt, "labels=", dict(labels))
    else:
        print(p, "MISSING")

d=pathlib.Path("models/distilbert_finetuned")
if d.is_dir():
    print("MODEL_DIR:", d)
    for fn in sorted(d.iterdir()):
        try:
            sz=fn.stat().st_size
        except Exception:
            sz='?'
        print(" -", fn.name, sz)
        if fn.suffix in ('.bin','.safetensors'):
            try:
                h=hashlib.sha1(fn.read_bytes()).hexdigest()
                print("   sha1:",h)
            except Exception as e:
                print("   sha1 failed:",e)
else:
    print("MODEL_DIR missing")
