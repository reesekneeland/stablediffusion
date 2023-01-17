"""make variations of input image"""

import argparse, os
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
import PIL
import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
from imwatermark import WatermarkEncoder


from scripts.txt2img import put_watermark
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

from nsd_access import NSDAccess
import open_clip



def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

# Takes in numpy array of image from nsd hdf5 file
def load_img(im_array):

    image = Image.fromarray(im_array)
    w, h = image.size

    # print(f"loaded input image of size ({w}, {h}) from array")

    w, h = 512, 512  # resize to integer multiple of 64
    imagePil = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(imagePil).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1., imagePil


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v2-inference.yaml",
        help="path to config which constructs model",
    )
    
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model",
    )

    # First URL: This is the original read-only NSD file path (The actual data)
    # Second URL: Local files that we are adding to the dataset and need to access as part of the data
    # Object for the NSDAccess package
    nsda = NSDAccess('/home/naxos2-raid25/kneel027/home/surly/raid4/kendrick-data/nsd', '/home/naxos2-raid25/kneel027/home/kneel027/nsd_local')

    # Setting up the device to use the GPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Sets the default parameters for LDM
    opt = parser.parse_args()

    # Loading the default stable diffusion model. 
    # This is the configuration for the stable diffusion model.
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    model = model.to(device)
    
    # Initialize OpenClip model
    openclip_model, _, oc_preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k', device=device)
    
    # Importing the caption object with 5 captions per image
    # Captions is a lists of lists, containing the 5 captions. 
    captions = nsda.read_image_coco_info([i for i in range(73000)], info_type='captions', show_annot=False)
    
    # Iterate through all images and captions in nsd sampled from COCO
    for i in range(0, 73000):
        
        # Array of image data 1 x 425 x 425 x 3 (Stores pixel intensities)
        img = nsda.read_images([i], show=False)
        
        # Variable to store the captions of the image. 
        prompts = []
        
        # Load the 5 prompts for each image
        for j in range(len(captions[i])):

            # Index into the caption list and get the corresponding 5 captions. 
            prompts.append(captions[i][j]['caption'])
        
        # Converting images to the format openClip likes
        init_image, img_pil = load_img(img[0])
        init_image = repeat(init_image, '1 ... -> b ...', b=1)
        image = oc_preprocess(img_pil).unsqueeze(0).to(device)
        
        # Tokenize the image 1 x 77
        token = open_clip.tokenize(prompts).to(device)
        
        # Generate clip embeddings for each prompt to find the best one
        img_embed = openclip_model.encode_image(image) # 1 x 1024
        text_embed = openclip_model.encode_text(token) # 1 x 1024
        
        # Normalize the embeddings so that we can calculate probability
        img_embed_norm = img_embed/img_embed.norm(dim=-1, keepdim=True)
        text_embed_norm = text_embed/text_embed.norm(dim=-1, keepdim=True)
        
        # Calculate the probability of a particular text embedding matching each image
        probs = (100.0 * img_embed_norm @ text_embed_norm.T).softmax(dim=-1)
        
        # Pick a singular prompt thats the best
        prompt = prompts[torch.argmax(probs)]

        # Get expanded clip embedding (1 x 77 x 1024) for best prompt using Stable Diffusion methods
        precision_scope = autocast
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():

                    # Stable diffusion encoder. Converts the text into a stable diffusion encoded clip vector. 
                    c = model.get_learned_conditioning(prompt) #1x77x1024
                    
        # Use the Stable Diffusion method to move to latent space to create latent z vector
        z = model.get_first_stage_encoding(model.encode_first_stage(init_image.to(device))) # 1 x 4 x 64 x 64
        
        # Save the c and z vectors into there corresponding files. 
        torch.save(c, "/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/c/" + str(i) + ".pt")
        torch.save(z, "/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/z/" + str(i) + ".pt")


if __name__ == "__main__":
    main()
