import json
import jsonlines
import gzip
import gzip
import shutil

num_train = 20
output_file = "nq-train-" + str(num_train) + ".jsonl.gz"


with jsonlines.open('output-train-full.jsonl', 'a') as writer:
    for n in range(10, num_train):
        print(n)
        train_file = "nq-train-" + str(n) + ".jsonl.gz"
        with gzip.open(train_file, "rb") as f:
            for line in f:
                my_json = line.decode('utf8')
                example = json.loads(my_json)
                writer.write(example)

with open('output-train-full.jsonl', 'rb') as f_in:
    with gzip.open(output_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)