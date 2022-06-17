import json
import jsonlines
import gzip
import gzip
import shutil

with jsonlines.open('output-dev-full.jsonl', 'w') as writer:
    with gzip.open("nq-dev-00.jsonl.gz", "rb") as f:
        for line in f:
            my_json = line.decode('utf8')
            example = json.loads(my_json)
            writer.write(example)
    
    with gzip.open("nq-dev-01.jsonl.gz", "rb") as f:
        for line in f:
            my_json = line.decode('utf8')
            example = json.loads(my_json)
            writer.write(example)

    with gzip.open("nq-dev-02.jsonl.gz", "rb") as f:
        for line in f:
            my_json = line.decode('utf8')
            example = json.loads(my_json)
            writer.write(example)
    
    with gzip.open("nq-dev-03.jsonl.gz", "rb") as f:
        for line in f:
            my_json = line.decode('utf8')
            example = json.loads(my_json)
            writer.write(example)

    with gzip.open("nq-dev-04.jsonl.gz", "rb") as f:
        for line in f:
            my_json = line.decode('utf8')
            example = json.loads(my_json)
            writer.write(example)

with open('output-dev-full.jsonl', 'rb') as f_in:
    with gzip.open('nq-dev-all.jsonl.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)