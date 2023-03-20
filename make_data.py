import os
import json
import platform

def pot2txt(pot_path, txt_path):
   print('write {}'.format(txt_path))
   for home, dirs, files in os.walk(pot_path):
      for filename in files:
         txt_home = home.replace(pot_path, txt_path)
         if (not os.path.isdir(txt_home)):
            os.makedirs(txt_home)

         fullname = os.path.join(home, filename)
         txt_filename = filename.replace('.pot', '.txt')
         txt_fullname = os.path.join(txt_home, txt_filename)
         if platform.system().lower() == 'windows':
            cmd = '{} {} {}'.format('pot2txt.exe', fullname, txt_fullname)
         else:
            cmd = '{} {} {}'.format('./pot2txt', fullname, txt_fullname)
         print('  {} {}'.format(fullname, txt_fullname), end='\r')
         os.system(cmd)
   print()

def txtfile2jsonfile(in_file, out_file):
   print('  {} {}'.format(in_file, out_file), end='\r')
   f = open(in_file, 'r', encoding='gbk')
   ink_data = {}
   if platform.system().lower() == 'windows':
      name = in_file.split('\\')
   else:
      name = in_file.split('/')
   
   ink_data['name'] = name[-3] + ' - ' + name[-1]
   ink_data['data'] = '2023-01-01'
   ink_data['samples'] = []
   for each in f:
      each = each.strip()
      items = each.split()
      sample_data = {}
      sample_data['code'] = items[0]
      data_lst = items[1].split(',')
      stroke_data = []
      
      l = 10000
      t = 10000
      r = 0
      b = 0

      if '-1,0,0,-1,-1,0' in items[1]:
         pass
      else:
         stroke_idx = 0
         point_data = []
         for i in range(int(len(data_lst) / 2) - 1):
            x = int(data_lst[i * 2])
            y = int(data_lst[i * 2 + 1])

            if x == -1 and y == -1:
               continue
            if x != -1:
               l = x if x<l else l
               t = y if y<t else t
               r = x if x>r else r
               b = y if y>b else b
               point_data.append({'x':x, 'y':y})
            else:
               stroke_data.append(point_data.copy())
               point_data.clear()
               stroke_idx += 1
         for stroke in stroke_data:
            for point in stroke:
               point['x'] -= l
               point['y'] -= t

         sample_data['rect'] = {'left':l, 'top':t, 'width':r-l, 'height':b-t}
         sample_data['strokes'] = stroke_data
         ink_data["samples"].append(sample_data.copy())
   
   f.close()
   f_out = open(out_file, 'w', encoding='utf-8')
   json.dump(ink_data, f_out, indent=4)
   f_out.close()


def txt2json(txt_path, json_path):
   print('write {}'.format(json_path))
   for home, dirs, files in os.walk(txt_path):
      for filename in files:
         json_home = home.replace(txt_path, json_path)
         if (not os.path.isdir(json_home)):
            os.makedirs(json_home)

         fullname = os.path.join(home, filename)
         out_filename = filename.replace('.txt', '.json')
         out_fullname = os.path.join(json_home, out_filename)
         txtfile2jsonfile(fullname, out_fullname)
   print()

def load_label(filename):
    f = open(filename, 'r', encoding='utf-8')
    labels = {}
    set_label = set()
    for each in f:
        each = each.strip()
        labels[each] = len(labels)
        set_label.add(each)
    f.close()
    return labels

g_map_stroke_point_num = {}
g_map_han_point_num = {}
g_map_han_stroke_num = {}

def jsonfile2samplefile(map_charset, in_file, out_file):
   print('  {}'.format(in_file), end='\r')
   f = open(in_file, 'r', encoding='utf-8')
   json_data = json.load(f)
   for sample in json_data['samples']:
      out_sample = {}
      han_point_num = 0
      if sample['code'] in map_charset:
         out_sample['code'] = sample['code']
         out_sample['strokes'] = []
         max_length = sample['rect']['width'] if sample['rect']['width'] > sample['rect']['height'] else sample['rect']['height']
         for stroke in sample['strokes']:
            out_points = []
            last_x = -1
            last_y = -1
            for point in stroke:
               x = int(point['x'] * 64 / max_length)
               y = int(point['y'] * 64 / max_length)
               if (last_x == x and last_y == y):
                  continue
               else:
                  out_points.append({'x':x, 'y':y})
                  last_x = x
                  last_y = y
            
            out_sample['strokes'].append(out_points)
            stroke_point_num = len(out_points)
            g_map_stroke_point_num[stroke_point_num] = g_map_stroke_point_num.get(stroke_point_num, 0) + 1
            han_point_num += stroke_point_num
         han_stroke_num = len(out_sample['strokes'])
         g_map_han_point_num[han_point_num] = g_map_han_point_num.get(han_point_num, 0) + 1
         g_map_han_stroke_num[han_stroke_num] = g_map_han_stroke_num.get(han_stroke_num, 0) + 1

         out_data = {}
         out_data['label'] = out_sample['code']
         out_data['data'] = []
         stroke_index = 0
         point_index = 0
         for stroke in out_sample['strokes']:
            stroke_index += 1
            for i in range(len(stroke)):
               point_index += 1
               x = stroke[i]['x']
               y = stroke[i]['y']
               s = stroke_index
               t = point_index
               out_data['data'].append([x, y, s, t])
         if len(out_data['data']) <= 320:
            out_file.write('{}\n'.format(json.dumps(out_data)))
   f.close()

def json2sample(map_charset, in_path, out_file):
   out_path = os.path.split(out_file)[0]
   if out_path and not os.path.isdir(out_path):
      os.makedirs(out_path)
   f_out = open(out_file, 'w', encoding='utf-8')
   print('write {}'.format(out_file))
   for home, dirs, files in os.walk(in_path):
      for filename in files:
         fullname = os.path.join(home, filename)
         jsonfile2samplefile(map_charset, fullname, f_out)
   f_out.close()
   print()

   stat_file = out_file + '.han_point_stat.txt'
   print('write', stat_file)
   lst_word = sorted(g_map_han_point_num.items(), key = lambda x: x[0], reverse = False)
   f = open(stat_file, 'w', encoding='utf-8')
   for each in lst_word:
      s = '{}\t{}'.format(each[0], each[1])
      f.write('{}\n'.format(s))
   f.close()

   stat_file = out_file + '.stroke_point_stat.txt'
   print('write', stat_file)
   lst_word = sorted(g_map_stroke_point_num.items(), key = lambda x: x[0], reverse = False)
   f = open(stat_file, 'w', encoding='utf-8')
   for each in lst_word:
      s = '{}\t{}'.format(each[0], each[1])
      f.write('{}\n'.format(s))
   f.close()

   stat_file = out_file + '.han_stroke_stat.txt'
   print('write', stat_file)
   lst_word = sorted(g_map_han_stroke_num.items(), key = lambda x: x[0], reverse = False)
   f = open(stat_file, 'w', encoding='utf-8')
   for each in lst_word:
      s = '{}\t{}'.format(each[0], each[1])
      f.write('{}\n'.format(s))
   f.close()

def make_casia_data(label_file, pot_path, result_file, temp_path):
   map_charset = load_label(label_file)
   txt_path = os.path.join(temp_path, 'txt')
   json_path = os.path.join(temp_path, 'json')
   pot2txt(pot_path, txt_path)
   txt2json(txt_path, json_path)
   json2sample(map_charset, json_path, result_file)
 
#make_casia_data('./data/labels/label-gb2312-level1.txt', './raw/casia-pot/Pot1.0Test', './data/casia-sample/Pot1.0Test.v1.json', './tmp/casia-pot/Pot1.0Test')
make_casia_data('./data/labels/label-gb2312-level1.txt', './raw/casia-pot/Pot1.0Train', './data/casia-sample/Pot1.0Train.v1.json', './tmp/casia-pot/Pot1.0Train')
