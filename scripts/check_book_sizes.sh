#!/bin/bash

books=${1:-books}

for fsrc in $books/{dev,eval}/*.txt; do 
  test -e $fsrc || continue
  echo -e "$(cat $fsrc | wc -l)\t$(cat $fsrc | grep -v "^\s*$" | wc -l)\t$fsrc"
  base=$(basename $fsrc .txt | rev | cut -c3- | rev)

  for fout in $books/MT/pipeline{1,2}/*{,/*}/$base??.txt; do 
    test -e $fout || continue
    echo -e "$(cat $fout | wc -l)\t$(cat $fout | grep -v "^\s*$" | wc -l)\t$fout"
  done
  echo ""
done
