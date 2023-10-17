import os

def list_databases(extension='.fasta'):
    return [f for f in os.listdir('databases') if f.endswith(extension)]

def open_database(database):
    with open(os.path.join('databases', database), 'r') as f:
        return f.read()

def parse_fasta(fasta):
    records = []
    splitted = fasta.split('\n')
    current_record_name = None
    current_record_sequence = ''
    for line in splitted:
        if line.startswith('>'):
            if current_record_name:
                records.append(FastaRecord(current_record_name, current_record_sequence))
            current_record_name = line[1:]
            current_record_sequence = ''
        else:
            current_record_sequence += line
    return records

    

class FastaRecord:
    def __init__(self, raw_name: str, raw_sequence: str):
        self.raw_name = raw_name
        self.raw_sequence = raw_sequence

        self.sequence = self.chunkify_sequence()
        self.class_name = self.extract_class_name_from_raw_name()


    def __str__(self):
        return f"[FastaRecord] {self.class_name}\n\t{self.sequence[:256]}..."
    
    def extract_class_name_from_raw_name(self):
        abc = [i.split('\x01') for i in self.raw_name.split(' ')]
        flattened = []
        for i in abc:
            flattened += i
        return flattened
    
    def chunkify_sequence(self, chunk_size=5):
        # return [self.raw_sequence[i:i+chunk_size] for i in range(0, len(self.raw_sequence), chunk_size)]
        return self.raw_sequence

