import sys, glob, imageio, tqdm
from multiprocessing import Pool, cpu_count
imgdir = sys.argv[1]
num_proc = cpu_count()

def process(x):
    for idx, f in enumerate(tqdm.tqdm(glob.glob(imgdir + '/*.jpg'))):
        if idx % num_proc == x:
            imageio.imsave(f.replace('jpg', 'png'), imageio.imread(f)[:, -512:])

with Pool(num_proc) as p:
    p.map(process, range(num_proc))

print('done')