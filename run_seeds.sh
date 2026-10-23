#!/bin/bash
cd D:/Sehoon/AES
for s in 0 1 2; do python bert.py klue/roberta-base klue_rb6_s$s 6 $s > log_klue_rb6_s$s.txt 2>&1; done
