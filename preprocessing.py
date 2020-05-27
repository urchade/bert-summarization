import glob
import os

link_files = glob.glob(os.path.join(r'docs_with_summaries\en', '*.txt'), recursive=True)

texts = []
for link in link_files:
    with open(link, 'r', encoding='utf8') as f:
        try:
            texts.append(f.read())
        except UnicodeDecodeError:
            pass

texts[1].split('\n\n@highlight\n\n')

orig_text = []

for txt in texts:
    orig_text.append(txt.split('\n\n@highlight\n\n')[0])

