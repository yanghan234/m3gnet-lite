import math

import scipy
import torch


def spherical_bessel(
    x: torch.Tensor,
    order: int = 0,
) -> torch.Tensor:
    """
    Compute the spherical Bessel function of the first kind.

    Args:
        x (torch.Tensor): The input tensor.
        order (int): The degree of the spherical Bessel function.
    """
    return scipy.special.spherical_jn(order, x)


def spherical_bessel_zeros(
    lmax: int = 0,
    num_zeros: int = 20,
) -> torch.Tensor:
    """
    Compute the zeros of the spherical Bessel function of the first kind.
    The zeros are precomputed for lmax = 0, 1, 2, 3, 4,
    thus no computation is needed in this function.

    Args:
        lmax (int): The maximum degree of the spherical Bessel function.
        num_zeros (int): The number of zeros to return.
    """
    if lmax < 0:
        raise ValueError("n must be greater than or equal to 0.")

    if lmax > 4:
        raise ValueError("n must be less than or equal to 4.")

    precomputed_zeros = [
        # l = 0
        [
            3.14159265358979,
            6.28318530717959,
            9.42477796076938,
            12.5663706143592,
            15.7079632679490,
            18.8495559215388,
            21.9911485751286,
            25.1327412287183,
            28.2743338823081,
            31.4159265358979,
            34.5575191894877,
            37.6991118430775,
            40.8407044966673,
            43.9822971502571,
            47.1238898038469,
            50.2654824574367,
            53.4070751110265,
            56.5486677646163,
            59.6902604182061,
            62.8318530717959,
        ],
        # l = 1
        [
            4.49340945790906,
            7.72525183693771,
            10.9041216594289,
            14.0661939128315,
            17.2207552719308,
            20.3713029592876,
            23.5194524986890,
            26.6660542588127,
            29.8115987908930,
            32.9563890398225,
            36.1006222443756,
            39.2444323611642,
            42.3879135681319,
            45.5311340139913,
            48.6741442319544,
            51.8169824872797,
            54.9596782878889,
            58.1022547544956,
            61.2447302603744,
            64.3871195905574,
        ],
        # l = 2
        [
            5.76345919689455,
            9.09501133047625,
            12.3229409705666,
            15.5146030108867,
            18.6890363553628,
            21.8538742227098,
            25.0128032022896,
            28.1678297079936,
            31.3201417074472,
            34.4704883312850,
            37.6193657535884,
            40.7671158214068,
            43.9139818113647,
            47.0601416127605,
            50.2057283367380,
            53.3508435852932,
            56.4955662618120,
            59.6399585795582,
            62.7840702561801,
            65.9279415029587,
        ],
        # l = 3
        [
            6.98793200050051,
            10.4171185473799,
            13.6980231532492,
            16.9236212852138,
            20.1218061744538,
            23.3042469889397,
            26.4767636645391,
            29.6426045403158,
            32.8037323851961,
            35.9614058047090,
            39.1164701902715,
            42.2695149777812,
            45.4209639722562,
            48.5711298516318,
            51.7202484303879,
            54.8685009575008,
            58.0160290641005,
            61.1629450448141,
            64.3093390906705,
            67.4552844798028,
        ],
        # l = 4
        [
            8.18256145257149,
            11.7049071545704,
            15.0396647076165,
            18.3012559595420,
            21.5254177333999,
            24.7275655478350,
            27.9155761994214,
            31.0939332140793,
            34.2653900861016,
            37.4317367682015,
            40.5941896534212,
            43.7536054311194,
            46.9106054900893,
            50.0656518347346,
            53.2190952897377,
            56.3712071531380,
            59.5222005873999,
            62.6722454406628,
            65.8214787430158,
            68.9700122850280,
        ],
    ]

    return precomputed_zeros[: (lmax + 1)][:num_zeros]


@torch.jit.script
def real_spherical_harmonics_m0(
    x: torch.Tensor,
    lmax: int = 4,
) -> torch.Tensor:
    r"""Compute the m = 0 compoent of the real spherical harmonics.
    $$Y_{00} = \frac{1}{\sqrt{4\pi}}$$
    $$Y_{10} = \sqrt{\frac{3}{4\pi}}x$$
    $$Y_{20} = \sqrt{\frac{5}{16\pi}}(3x^2 - 1)$$
    $$Y_{30} = \sqrt{\frac{7}{16\pi}}(5x^3 - 3x)$$

    Args:
        x (torch.Tensor): The input tensor.
        lmax (int): The maximum degree of the spherical harmonics.

    Returns:
        torch.Tensor: The spherical harmonics of degree 0.
    """
    if lmax < 0:
        raise ValueError("lmax must be greater than or equal to 0.")

    if lmax > 4:
        raise ValueError("lmax must be less than or equal to 4.")

    results = []

    sh_0_0 = torch.ones_like(x) / math.sqrt(4.0 * math.pi)
    results.append(sh_0_0)

    if lmax >= 1:
        sh_1_0 = torch.sqrt(3.0 / (4.0 * math.pi)) * x
        results.append(sh_1_0)

    if lmax >= 2:
        sh_2_0 = torch.sqrt(5.0 / (16.0 * math.pi)) * (3.0 * x**2 - 1.0)
        results.append(sh_2_0)

    if lmax >= 3:
        sh_3_0 = torch.sqrt(7.0 / (16.0 * math.pi)) * (5.0 * x**3 - 3.0 * x)
        results.append(sh_3_0)

    if lmax >= 4:
        sh_4_0 = torch.sqrt(9.0 / (256.0 * math.pi)) * (35.0 * x**4 - 30.0 * x**2 + 3.0)
        results.append(sh_4_0)

    return torch.stack(results, dim=-1)
