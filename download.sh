FROM=1
TO=5


for i in $(seq -f "%g" $FROM $TO)
do
    echo $i
    cd databases
    wget https://ftp.ncbi.nlm.nih.gov/genbank/gbbct$i.seq.gz
    gunzip gbbct$i.seq.gz

    # remove first 10 lines (macos)
    sed -i '' 1,10d gbbct$i.seq

    # change name, overwrite force
    mv gbbct$i.seq gbbct$i.gb
    cd ..
done
