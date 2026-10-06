# Phase 0 labelling aid: one middle frame per folder, tiled 6x5 with the folder
# name, so environment labels can be assigned by eye. Writes
# restart/phase0/labelling/sheet_NN.jpg. Reads restart/data/inventory.csv.
import os, csv
import cv2, numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA = os.path.join(ROOT, '3rd_Anti-UAV_train_val')
OUT = os.path.join(ROOT, 'restart', 'phase0', 'labelling')
COLS, ROWS, TW, TH = 6, 5, 320, 256
os.makedirs(OUT, exist_ok=True)

rows = list(csv.DictReader(open(os.path.join(ROOT, 'restart', 'data', 'inventory.csv'))))
todo = [r for r in rows if r['environment'] == 'unlabelled']
per = COLS * ROWS
for k in range(0, len(todo), per):
    sheet = np.full((ROWS * (TH + 22), COLS * TW, 3), 255, np.uint8)
    for i, r in enumerate(todo[k:k + per]):
        d = os.path.join(DATA, r['official_split'], r['folder'])
        imgs = sorted(x for x in os.listdir(d) if x.endswith('.jpg'))
        im = cv2.imread(os.path.join(d, imgs[len(imgs) // 2]))
        im = cv2.resize(im, (TW, TH))
        y, x = (i // COLS) * (TH + 22), (i % COLS) * TW
        sheet[y + 22:y + 22 + TH, x:x + TW] = im
        cv2.putText(sheet, '%d %s' % (k + i, r['folder'][:30]), (x + 4, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(OUT, 'sheet_%02d.jpg' % (k // per + 1)), sheet,
                [cv2.IMWRITE_JPEG_QUALITY, 80])
print('%d folders on %d sheets -> %s' % (len(todo), -(-len(todo) // per), OUT))
