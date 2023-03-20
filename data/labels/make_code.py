f = open('label-gb2312-level1.txt', encoding='utf-8')
i = 0
f_o = open('label-gb2312-level1.c', mode="w", encoding='utf-8')
f_o.write('unsigned short g_labels[] =\n')
f_o.write('{\n')
for each in f:
    f_o.write('   {}, // {} {}\n'.format(hex(ord(each[0])), i, each[0]))
    i+=1
f_o.write('}\n')
