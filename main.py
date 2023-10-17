import torch
from databases import *

databases = list_databases()
print(databases)

fasta = parse_fasta(open_database(databases[1]))

dick = {}

for fasta_record in fasta:
    for word in fasta_record.class_name:
        if not word in dick:
            dick[word] = 1
        else:
            dick[word] += 1
# sort descendingly both arrays based on results array (numerical)

srtd = sorted(dick.items(), key=lambda x:x[1], reverse=True)
for i in srtd[:10]:
    print(i)


