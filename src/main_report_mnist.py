import glob
from PIL import Image

# filepaths
fp_in = "/images/reconstruction_PSMVAE_a_*.png"
fp_out = "/iamges/reconstruction_PSMVAE_a.gif"

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=200, loop=0)



fp_in = "/images/reconstruction_PSMVAE_b_*.png"
fp_out = "/iamges/reconstruction_PSMVAE_b.gif"

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=200, loop=0)