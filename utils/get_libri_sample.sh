files=$(echo /mnt/diskb/corpora/LibriSpeech/test-clean/*/*/*.flac)
rm tmp.txt
for file in $files
do
	length=$(soxi -s $file)
	echo $file,$length >> tmp.txt
done
sort -t, -k2 tmp.txt -n -o tmp.txt --parallel=8
awk "NR % 82==1" tmp.txt > result.txt
rm tmp.txt