cd databases
# wget https://ftp.ncbi.nlm.nih.gov/blast/db/FASTA/swissprot.gz
# wget https://ftp.ncbi.nlm.nih.gov/blast/db/FASTA/pdbaa.gz
wget https://ftp.ncbi.nlm.nih.gov/blast/db/mouse_genome.00.tar.gz
wget https://ftp.ncbi.nlm.nih.gov/blast/db/human_genome.00.tar.gz

# unzip
gunzip ./mouse_genome.00.tar.gz
gunzip ./human_genome.00.tar.gz

# change extenstion to fasta
mv ./mouse_genome.00 ./mouse_genome.00.fasta
mv ./pdhuman_genome.00baa ./pdbaahuman_genome.00.fasta

# remove zip files
rm ./mouse_genome.00.tar.gz
rm ./human_genome.00.tar.gz
cd ..
rm *.gz
