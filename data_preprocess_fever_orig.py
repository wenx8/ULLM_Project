import json, glob, random
from collections import defaultdict

random.seed(42)
N_PER_LABEL = 3000

claims = [json.loads(l) for l in open("/home/scur0470/ULLMPROJ/ULLM_Project/data/fever-original/shared_task_dev.jsonl", encoding="utf-8")]
claims = [c for c in claims if c["label"] in ("SUPPORTS", "REFUTES")]

by_label = defaultdict(list)
for c in claims:
    by_label[c["label"]].append(c)
sample = [c for lab in by_label for c in random.sample(by_label[lab], N_PER_LABEL)]

needed = {ev[2] for c in sample for ev_set in c["evidence"] for ev in ev_set}

wiki = {}
for fn in glob.glob("/home/scur0470/ULLMPROJ/ULLM_Project/data/fever-original-wikipages/wiki-pages/*.jsonl"):
    for line in open(fn, encoding="utf-8"):
        d = json.loads(line)
        if d["id"] in needed:
            sents = {}
            for l in d["lines"].split("\n"):
                parts = l.split("\t")
                if len(parts) >= 2 and parts[1].strip():
                    sents[int(parts[0])] = parts[1]
            wiki[d["id"]] = sents

with open("/home/scur0470/ULLMPROJ/ULLM_Project/data/processed/pilot_fever.jsonl", "w", encoding="utf-8") as out:
    for c in sample:
        for ev_set in c["evidence"]:
            texts = [wiki.get(p, {}).get(s) for _, _, p, s in ev_set]
            if all(texts):
                text = " ".join(texts)
                out.write(json.dumps({
                    "id": c["id"], "claim": c["claim"],
                    "evidence_text": text, "label": c["label"],
                    "evidence_length_words": len(text.split()),
                }) + "\n")
                break