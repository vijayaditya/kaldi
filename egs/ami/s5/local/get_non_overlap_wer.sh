

for overlap_html in $*; do
  wer=$(cat $overlap_html | \
    grep '<td style="vertical-align: center; text-align: center;"><b>#Ref Speaker<br>Overlap</b></td>' -A39 | tail -1 | sed -e 's/.*center...//g;s/% (.*//g')
  echo $overlap_html $wer
done | python -c "import sys
min_wer=10000000000000000000000
min_wer_line=None
for line in sys.stdin:
  print line.strip()
  parts = line.split()
  if float(parts[1]) <= min_wer:
    min_wer_line=line
print min_wer_line.strip()
print '=='*10
"
