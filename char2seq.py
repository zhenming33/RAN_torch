# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 14:27:35 2018

@author: ensur
"""

import os
import numpy as np
import pickle

# generate all characters dict
lines = open('cjkvi-ids/ids.txt',encoding='UTF-8').readlines()[2:]
char_seq = {}
char_seq['⿰'] = '⿰'
char_seq['⿱'] = '⿱'
char_seq['⿵'] = '⿵'
char_seq['⿻'] = '⿻'
char_seq['⿺'] = '⿺'
char_seq['⿹'] = '⿹'
char_seq['⿶'] = '⿶'
char_seq['⿳'] = '⿳'
char_seq['⿴'] = '⿴'
char_seq['⿸'] = '⿸'
char_seq['⿷'] = '⿷'
char_seq['⿲'] = '⿲'
char_seq['A'] = 'A'
char_seq['H'] = 'H'
char_seq['U'] = 'U'
char_seq['X'] = 'X'
for i in range(len(lines)):
    a = lines[i].split(' ')[0].replace('\n','').split('\t')
    seq = a[2].replace(' ','').replace('[','').replace(']','')\
                .replace('G','').replace('T','').replace('J','')\
                .replace('K','').replace('V','')
    char_seq[a[1]] = seq

for i in range(len(lines)):
    a = lines[i].split(' ')[0].replace('\n','').split('\t')
    seq = a[2].replace(' ','').replace('[','').replace(']','')\
                .replace('G','').replace('T','').replace('J','')\
                .replace('K','').replace('V','')
    for k in seq:
        char_seq[k]

# analysis all seq
def is_all(seq):
    all_len = [len(char_seq[c]) for c in seq]
    if max(all_len) > 1:
        return False
    else:
        return True

char_seq_all = {}
for i in range(len(lines)):
    print(i)
    a = lines[i].split(' ')[0].replace('\n','').split('\t')
    char = a[1]
    seq_tmp = char_seq[a[1]]
    while not is_all(seq_tmp):
        for k in range(len(seq_tmp)):
            if len(char_seq[seq_tmp[k]]) > 1:
                seq_tmp = seq_tmp.replace(seq_tmp[k],char_seq[seq_tmp[k]])
        print(seq_tmp)
    char_seq_all[char] = seq_tmp

alphabet = ''
for value in char_seq_all.values():
    alphabet += value
alphabet = list(set(alphabet))
alphabet = ''.join(alphabet)

print(len(alphabet))

char_seq_index = {}
for keys in char_seq_all.keys():
    char_seq_index[keys] = [alphabet.index(c) for c in list(char_seq_all[keys])]

#保存序列
save_file = open('char2seq_dict_real.pkl', 'wb')
pickle.dump(char_seq_all, save_file)
save_file.close()

#保存序列
save_file = open('char2seq_dict.pkl', 'wb')
pickle.dump(char_seq_index, save_file)
save_file.close()


#保存字典
f = open('radical_alphabet.txt','w',encoding='utf-8')
f.write(alphabet)
f.close()