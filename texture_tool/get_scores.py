import sys, glob, imageio
import skimage.metrics
import numpy as np
import torch
import tqdm
import cv2
import os
import tempfile

output = sys.argv[1]
target = sys.argv[2]
ratio = int(sys.argv[3]) if len(sys.argv) > 3 else 1
use_mask = sys.argv[4] if len(sys.argv) > 4 else None
fname = sys.argv[5] if len(sys.argv) > 5 else "*"

g_files = sorted(glob.glob(output + f'/{fname}.png'))
t_files = sorted(glob.glob(target + f'/{fname}.png'))

# def fnc(g): return int(g.split('_')[-1].split('.')[0])
# g_files = [g for g in g_files if fnc(g) >= 100 and fnc(g) < 7300]  # 17000
# t_files = [t for t in t_files if fnc(t) >= 100 and fnc(t) < 7300]  # 17000

print(output, target)
print(len(g_files), len(t_files))

# def trs(g):
#     a, b = g.split('/')[-1].split('.')[0].split('_')
#     # f_name = f'/private/home/jgu/work/neuralbody/data/valid/vlad_full/marc_output/{a}/0{b}.png'
#     # f_name = f'/private/home/jgu/work/neuralbody/data/valid/vlad_full/marc_training/{a}/0{b}.png'
#     # f_name = f'/private/home/jgu/work/neuralbody/data/valid/vlad_full/marc_training/{a}/0{b}.png'
#     # f_name = f'/private/home/jgu/work/neuralbody/data/valid/vlad_full/marc_testing/useNoGTTex/{a}/0{b}.png'
#     f_name = f'/private/home/jgu/work/neuralbody/data/valid/vlad_full/ddc_testing/{a}/0{b}.png'
#     return f_name

# def tr0(g):
#     a, b = g.split('/')[-1].split('.')[0].split('_')
#     f_name = f'/private/home/jgu/work/neuralbody/data/valid/vlad_full/marc_testing/{a}/0{b}.png'
#     return f_name

# g_files = [trs(g) for g in g_files]
# import pdb;pdb.set_trace()

tmpsrc = tempfile.mkdtemp() if (ratio != 1) or (use_mask is not None) else output
tmptgt = tempfile.mkdtemp() if (ratio != 1) or (use_mask is not None) else target
print(tmpsrc, tmptgt)

from lpips_pytorch import LPIPS
lpips = LPIPS(net_type='alex', version='0.1')
lpips = lpips.cuda()

from multiprocessing import Pool, cpu_count
num_proc = cpu_count()

psnrs = []
lpipses = []

# oleks: 2~6900
if use_mask is not None:
    H, W = [int(a) for a in use_mask.split(',')]
    print(H, W)

def func(x):
    psnrs, ssims = [], []
    for i in tqdm.tqdm(range(len(g_files))):
        if i % num_proc == x:
            g = imageio.imread(g_files[i]).astype('float32') / 255.
            t = imageio.imread(t_files[i]).astype('float32') / 255.
            g = cv2.resize(g, (t.shape[1], t.shape[0]))
            h0, w0 = h, w = g.shape[0], g.shape[1]
            
            if use_mask is not None:
                ii, jj = np.where(~(t == 1).all(-1))
                try:
                    hmin, hmax = np.min(ii), np.max(ii)
                    uu = (H - (hmax + 1 - hmin)) // 2
                    vv = H - (hmax - hmin) - uu
                    if hmin - uu < 0:
                        hmin, hmax = 0, H
                    elif hmax + vv > h:
                        hmin, hmax = h - H, h
                    else:
                        hmin, hmax = hmin - uu, hmax + vv

                    wmin, wmax = np.min(jj), np.max(jj)
                    uu = (W - (wmax + 1 - wmin)) // 2
                    vv = W - (wmax - wmin) - uu
                    if wmin - uu < 0:
                        wmin, wmax = 0, W
                    elif wmax + vv > w:
                        wmin, wmax = w - W, w
                    else:
                        wmin, wmax = wmin - uu, wmax + vv
                except ValueError:
                    print(f"target is empty {i}")
                    continue

                g = g[hmin: hmax, wmin: wmax]
                t = t[hmin: hmax, wmin: wmax]
                h, w = g.shape[0], g.shape[1]
                assert (h == H) and (w == W), f"error {hmin} {hmax} {wmin} {wmax} {h0} {w0} {uu} {vv}"

            if i == 0:
                print(h, w, ratio)
            if ratio != 1:
                try:
                    g = cv2.resize(g, (w//ratio, h//ratio))
                    t = cv2.resize(t, (w//ratio, h//ratio))
                except Exception:
                    import pdb; pdb.set_trace()

            if (ratio != 1) or (use_mask is not None):
                imageio.imsave("{}/{}.png".format(tmpsrc, i), (g * 255).astype('uint8'))
                imageio.imsave("{}/{}.png".format(tmptgt, i), (t * 255).astype('uint8'))

            psnrs += [skimage.metrics.peak_signal_noise_ratio(g, t, data_range=1)]
            ssims += [skimage.metrics.structural_similarity(g, t, multichannel=True, data_range=1)]
    return np.asarray(psnrs), np.asarray(ssims)

with Pool(num_proc) as p:
    results = p.map(func, range(num_proc))
psnr  = np.concatenate([r[0] for r in results]).mean()
print(f"PSNR {psnr}")
ssim  = np.concatenate([r[1] for r in results]).mean()
print(f"SSIM {ssim}")

g_files = sorted(glob.glob(tmpsrc + '/*.png'))
t_files = sorted(glob.glob(tmptgt + '/*.png'))
for i in tqdm.tqdm(range(len(g_files))):
    g = imageio.imread(g_files[i]).astype('float32') / 255.
    t = imageio.imread(t_files[i]).astype('float32') / 255.
    lpipses += [lpips(
        2 * torch.from_numpy(g).cuda().unsqueeze(-1).permute(3,2,0,1) - 1,
        2 * torch.from_numpy(t).cuda().unsqueeze(-1).permute(3,2,0,1) - 1
    ).item()]
lpips = np.mean(lpipses)
print(f"LPIPS {lpips}")
os.system('python -m pytorch_fid --device cuda {} {}'.format(tmpsrc, tmptgt))