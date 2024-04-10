#!/bin/bash
fnum=(100 200 300 400 500 600)
for i in {0..5}
do
    python secure-feature-select.py ${fnum[$i]}
done
