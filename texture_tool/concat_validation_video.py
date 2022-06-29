import sys, glob, tqdm
import torch, math, imageio
from torchvision.utils import make_grid

image_dir = sys.argv[1]
out_imgs  = sorted(glob.glob(f'{image_dir}/output/*.png'))
tgt_imgs  = sorted(glob.glob(f'{image_dir}/target/*.png'))


def get_names(imgs):
    name_dict = {}
    for name in imgs:
        a, _ = name.split('/')[-1][:-4].split('_')
        if a not in name_dict:
            name_dict[a] = [name]  
        else: 
            name_dict[a].append(name)
    return name_dict, [v for v in name_dict]

out_imgs, cams = get_names(out_imgs)
tgt_imgs, _    = get_names(tgt_imgs)
n_cameras, n_examples = len(out_imgs), len(out_imgs[cams[0]])

writer = imageio.get_writer(f'{image_dir}/comparison.mp4', fps=40)
for i in tqdm.tqdm(range(n_examples)):
    out_img = torch.stack([torch.from_numpy(imageio.imread(out_imgs[v][i])) for v in out_imgs]).permute(0,3,1,2)
    tgt_img = torch.stack([torch.from_numpy(imageio.imread(tgt_imgs[v][i])) for v in tgt_imgs]).permute(0,3,1,2)
    out_img = make_grid(out_img, nrow=int(math.sqrt(n_cameras)), padding=20)
    tgt_img = make_grid(tgt_img, nrow=int(math.sqrt(n_cameras)), padding=20)
    ful_img = torch.cat([out_img, tgt_img], 2).permute(1,2,0).numpy()
    writer.append_data(ful_img)
writer.close()
