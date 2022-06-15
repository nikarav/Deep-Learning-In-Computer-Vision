import os
from pathlib import Path
import tqdm
import numpy as np
import click
from PIL import Image
from typing import List, Optional
import torch
import legacy
import dnnlib

def make_latent_control_animation(start_amount, end_amount, step_size):

    amounts = np.linspace(start_amount, end_amount, abs(end_amount-start_amount)/step_size)
    return amounts

    
    


#----------------------------------------------------------------------------
@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--dir-path', 'direction_path', help='Direction npy file', required=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', 'projected_w', help='Projection result file', type=str, metavar='FILE', required=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def direction(
    ctx: click.Context,
    network_pkl: str,
    direction_path: str,
    noise_mode: str,
    outdir: str,
    projected_w: str
):

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)
    person='tsouv'
    feature='smile'
    start_amount=-5
    end_amount=5

    files = [x for x in Path(direction_path).iterdir() if str(x).endswith('.npy')]
    latent_controls  = {f.name[:-4]:np.load(f) for f in files}
    feature = list(latent_controls.keys())[0]

    #latent_code_to_use = ws[0]

    if projected_w is not None:
        
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)['w']
        ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        modified_latent_code_start = ws.cpu() + latent_controls[feature]*start_amount
        modified_latent_code_end = ws.cpu() + latent_controls[feature]*end_amount

        #all_imgs = []
        
        #amounts = make_latent_control_animation(start_amount=-5, end_amount=5, step_size=0.1)
        #for amount_to_move in tqdm(amounts):
         #   modified_latent_code = np.array(latent_code_to_use)
          #  modified_latent_code += latent_controls[feature]*amount_to_move
           # img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            #img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            #all_imgs.append(get_concat_h(img, w))
        #save_name = outdir/'{0}_{1}.gif'.format(person, feature)
        #all_imgs[0].save(save_name, save_all=True, append_images=all_imgs[1:], duration=5, loop=0)
        for idx, w in enumerate(modified_latent_code_end.cuda()):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj{idx:02d}.png')

        for idx, w in enumerate(modified_latent_code_start.cuda()):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj{idx:02d}.png')
        return

if __name__ == "__main__":
    direction() # pylint: disable=no-value-for-parameter

#--------------