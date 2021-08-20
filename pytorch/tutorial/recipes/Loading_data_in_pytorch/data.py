# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/4/16 16:43

import torchaudio
import matplotlib.pyplot as plt


def load_data(data_dir=r'data'):
    yesno_data = torchaudio.datasets.YESNO(
        root=data_dir,
        url='http://www.openslr.org/resources/1/waves_yesno.tar.gz',
        download=True
    )
    return yesno_data


def visualize(waveform):
    wave_numpy = waveform.t().numpy()
    plt.figure()
    plt.plot(wave_numpy)
    plt.show()


def __test():
    yesno_data = load_data()
    waveform, sample_rate, label = yesno_data[2]
    print(f'Waveform_shape: {waveform.shape}\nSample_rate: {sample_rate}\nLabel: {label}')
    visualize(waveform)


if __name__ == '__main__':
    __test()