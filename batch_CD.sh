#!/bin/bash

for INITIAL in {236..240}
do
  for FINAL in {241..245}
  do
    vaspberry -kx 1 -ky 1 -cd 1 -ii $INITIAL -if $FINAL -o CD_${INITIAL}_${FINAL}
  done
done

