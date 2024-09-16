import numpy as np

class MHSOSystem:
    def __init__(self):
        self.params = {
            'g1': 1925, 'gkv': 1700, 'gL': 7, 'v1': 100, 'vk': -75,
            'vL': -40, 'vc': 100, 'kc': 3.3/18, 'lamn': 230, 'gkc': 12.0,
            'rao': 0.27, 'k0': 0.1, 'k1': 1.0, 'k2': 0.5, 'E': 0,
            'ohm1': 1.414, 'ohm2': 0.618
        }

    def equations(self, t, state):
        x, y, z, w = state
        p = self.params

        mp = np.tanh(w)
        alph = 0.07 * np.exp(-0.05*x - 2.5)
        betah = 1 / (1 + np.exp(-0.1*x - 2))
        alpm = (0.1 * (25 + x)) / (1 - np.exp(-0.1*x - 2.5))
        betam = 4 * np.exp(-(x + 50) / 18)
        alpn = (0.01 * (20 + x)) / (1 - np.exp(-0.1*x - 2))
        betan = 0.125 * np.exp(-(x + 30) / 80)
        taun = 1 / (p['lamn'] * (alpn + betan))
        ninf = alpn / (alpn + betan)
        hinf = alph / (alph + betah)
        minf = alpm / (alpm + betam)

        phiext = p['E'] * (np.sin(p['ohm1']*t) + np.sin(p['ohm2']*t))

        dx = p['g1']*minf**3*hinf*(p['v1']-x) + p['gkv']*y**4*(p['vk']-x) + \
             p['gkc']*(z/(1+z))*(p['vk']-x) + p['gL']*(p['vL']-x) - p['k0']*mp*x
        dy = (ninf - y) / taun
        dz = (minf**3 * hinf * (p['vc']-x) - p['kc']*z) * p['rao']
        dw = p['k1']*x - p['k2']*w + phiext

        return [dx, dy, dz, dw]
