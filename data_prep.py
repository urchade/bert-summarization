import os
from tqdm import tqdm

def save_data(input_path=r'data_french\raw\all_src_sep.txt',
              oracle_path=r'data_french\oracle\oracle_file.txt',
              out_folder=r'data_french\preprocessed',
              cls='<s>',
              sep='</s>'):
    text = []
    label = []

    with open(input_path, 'r', encoding='utf8') as f:
        for t in f:
            text.append(t.replace('##SENT##', sep + cls))

    with open(oracle_path, 'r', encoding='utf8') as f:
        for t in f:
            label.append(t.split('\t')[0])

    for i, (t, l) in tqdm(enumerate(zip(text, label)), total=len(text)):
        with open(os.path.join(out_folder, f'data_{"0"*(6-len(str(i)))}{i}.txt'), 'w', encoding='utf8') as f:
            f.write(t)
            f.write(l.replace('(', '').replace(')', '').replace(' ', '').rstrip(','))


if __name__ == "__main__":
    save_data()
