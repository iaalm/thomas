import json
import h5py
import os

n_densecap = 10
max_len = 16

with open('/s/coco/cocotalk.json') as fd:
    cocotalk = json.load(fd)
ix_w = cocotalk['ix_to_word']
imgs = cocotalk['images']
w_ix = {ix_w[k]: int(k) for k in ix_w}
unk_ix = w_ix['UNK']


def cap2ix(sent):
    tun = sent.strip().lower().split(' ')[:max_len]
    res = [0] * (16 - len(tun))
    for i in tun:
        res.append(w_ix.get(i, unk_ix))
    return res


cocotalk = h5py.File('/s/coco/cocotalk.h5', 'r')
output = h5py.File('cocodens.h5', 'w')

n_imgs = len(imgs)
print(n_imgs)

densecap = {}
path_template = \
    '/home/whz/densecap-master/densecap-master/vis/test/results_coco_%s.json'

filenames = ['test']\
    + ['train%d' % i for i in range(1, 10)] \
    + ['val%d' % i for i in range(1, 6)]
for filename in filenames:
    print('reading', filename)
    with open(path_template % filename) as fd:
        data = json.load(fd)
        for i in data['results']:
            densecap[i['img_name']] = i

h5_d = output.create_dataset('densecaps',
                             (n_imgs, n_densecap, max_len), 'int16')
h5_l = output.create_dataset('locations', (n_imgs, n_densecap, 4), 'float')
h5_s = output.create_dataset('scores', (n_imgs, n_densecap), 'float')
# output.create_dataset('labels', data=cocotalk['labels'])
# output.create_dataset('label_start_ix', data=cocotalk['label_start_ix'])
# output.create_dataset('label_end_ix', data=cocotalk['label_end_ix'])

for ix, img in enumerate(imgs):
    if ix % 100 == 0:
        print('processing %d/%d' % (ix, n_imgs))
    name = os.path.basename(img['file_path'])
    try:
        data = densecap[name]
    except KeyError:
        print(name, 'not found')
        continue
    for j in range(n_densecap):
        h5_d[ix, j, :] = cap2ix(data['captions'][j])
        h5_l[ix, j, :] = data['boxes'][j]
        h5_s[ix, j] = data['scores'][j]

output.close()
