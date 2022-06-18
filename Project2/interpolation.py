import click
import numpy as np
import torch

@click.command()
@click.pass_context
@click.option('--latent1', 'latent1', help='Path to latent space of the 1st image', required=True)
@click.option('--latent2', 'latent2', help='Path to latent space of the 2nd image', required=True)
@click.option('--per', 'per', help='How much to keep from 1st image, the rest is taken from the 2nd.', required=True)
@click.option('--outdir', help='Where to save the output latent space', type=str, required=True, metavar='DIR')
def interpolation(
    ctx: click.Context,
    latent1: str,
    latent2: str,
    per: str,
    outdir: str,
):
    device = torch.device('cuda')
    ws_one = np.load(latent1)['w']
    ws_two = np.load(latent2)['w']
    ws_one = torch.tensor(ws_one, device=device)
    ws_two = torch.tensor(ws_two, device=device)
    per = float(per)
    ws_one = torch.mul(ws_one, per)
    ws_two = torch.mul(ws_two, (1 - per))
    new_latent = ws_one.add(ws_two)
    np.savez(f'{outdir}/joined_projected.npz', w=new_latent.cpu().numpy())
    


if __name__ == "__main__":
    interpolation() # pylint: disable=no-value-for-parameter
