import click
import matplotlib.pyplot as plot
import numpy as np
import os
import soundfile as sf

from sdft.stft import STFT

def read(path):

    name, ext = os.path.splitext(path)
    path = name + (ext or '.wav')

    data, samplerate = sf.read(path)

    click.echo(f'READ {path} (shape={data.shape}, dtype={data.dtype}, sr={samplerate})')

    data = np.atleast_2d(data)
    assert (data.ndim == 2) and (data.shape[-1] == 2), f'Invalid data shape {data.shape}!'

    return data, samplerate

def write(path, data, samplerate):

    name, ext = os.path.splitext(path)
    path = name + (ext or '.wav')

    click.echo(f'WRITE {path} (shape={data.shape}, dtype={data.dtype}, sr={samplerate})')

    data = np.atleast_1d(data)
    assert data.ndim == 1, f'Invalid data shape {data.shape}!'

    sf.write(path, data, samplerate, format='wav', subtype='PCM_24')

@click.command('azimuth', help='stereo sound source separation', no_args_is_help=True, context_settings=dict(help_option_names=['-h', '--help']))
@click.option('-i', '--input', required=True, help='input stereo .wav file name')
@click.option('-o', '--output', required=True, help='output mono .wav file name')
@click.option('-s', '--swap', is_flag=True, default=False, help='swap source channels')
@click.option('-w', '--window', default='4k', show_default=True, help='stft window size')
@click.option('-v', '--overlap', default='4', show_default=True, help='stft window overlap')
@click.option('-d', '--debug', is_flag=True, default=False, help='plot spectrograms before and after processing')
def main(input, output, swap, window, overlap, debug):

    def kilo(value): return int(value[:-1]) * 1024 if value.lower().endswith('k') else int(value)

    framesize = kilo(window)
    hopsize = framesize // int(overlap)
    stft = STFT(framesize, hopsize)

    x, sr = read(input)
    x = np.flip(x, -1) if swap else x

    l = x[..., 0]
    r = x[..., 1]
    m = (l + r) / 2
    d = (l - r)

    l = stft.stft(l)
    r = stft.stft(r)
    m = stft.stft(m)
    d = stft.stft(d)

    mag = (np.abs(l) - np.abs(r)) / 1
    phi = np.angle(m)

    y = stft.istft(mag * np.exp(1j * phi))
    y = np.clip(y, -1, +1)

    write(output, y, sr)

if __name__ == '__main__':

    main()
