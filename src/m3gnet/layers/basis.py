import torch
from torch import nn


def spherical_jn(
    angular_l: int,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the spherical Bessel function of the first kind for torch.Tensor,
    with a stable implementation for x=0.
    """
    x_is_zero = x == 0.0
    x_no_zero = torch.where(
        x_is_zero, torch.tensor(1.0, dtype=x.dtype, device=x.device), x
    )

    if angular_l == 0:
        results = torch.sin(x_no_zero) / x_no_zero
        results = torch.where(
            x_is_zero, torch.tensor(1.0, dtype=x.dtype, device=x.device), results
        )
    elif angular_l == 1:
        results = torch.sin(x_no_zero) / x_no_zero**2 - torch.cos(x_no_zero) / x_no_zero
    elif angular_l == 2:
        results = (3 / x_no_zero**3 - 1 / x_no_zero) * torch.sin(
            x_no_zero
        ) - 3 / x_no_zero**2 * torch.cos(x_no_zero)
    elif angular_l == 3:
        results = (15 / x_no_zero**4 - 6 / x_no_zero**2) * torch.sin(x_no_zero) - (
            15 / x_no_zero**3 - 1 / x_no_zero
        ) * torch.cos(x_no_zero)
    elif angular_l == 4:
        results = (105 / x_no_zero**5 - 45 / x_no_zero**3 + 1 / x_no_zero) * torch.sin(
            x_no_zero
        ) - (105 / x_no_zero**4 - 10 / x_no_zero**2) * torch.cos(x_no_zero)
    elif angular_l == 5:
        results = (
            945 / x_no_zero**6 - 420 / x_no_zero**4 + 15 / x_no_zero**2
        ) * torch.sin(x_no_zero) - (
            945 / x_no_zero**5 - 105 / x_no_zero**3 + 1 / x_no_zero
        ) * torch.cos(x_no_zero)
    else:
        raise ValueError("Only angular_l = 0, 1, 2, 3, 4, 5 are supported.")

    # For l > 0, the value at x=0 is always 0
    if angular_l > 0:
        results = torch.where(
            x_is_zero, torch.tensor(0.0, dtype=x.dtype, device=x.device), results
        )

    return results


def spherical_bessel_zeros(
    angular_l: int = 0,
    radial_n: int = 0,
) -> torch.Tensor:
    """
    Query the zero points of the spherical Bessel function of the first kind.
    The zeros are precomputed for angular_l = 0, 1, 2, 3, 4, 5 and radial_n < 20.
    Beyond this range, the zeros need to be computed on the fly.

    Args:
        angular_l (int): The degree of the spherical Bessel function.
        radial_n (int): The number of zeros to return.
    """
    if angular_l < 0 or radial_n < 0:
        raise ValueError("angular_l and radial_n must be greater than or equal to 0.")

    if angular_l > 4 or radial_n > 4:
        raise ValueError(
            f"angular_l = {angular_l} and radial_n = {radial_n} is "
            "beyond the precomputed range. "
        )

    precomputed_zeros = torch.tensor(
        [
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
            # l = 5
            [
                9.3558121110427,
                12.9665301727743,
                16.3547096393501,
                19.6531521018210,
                22.9045506479037,
                26.1277501372255,
                29.3325625785848,
                32.5246612885788,
                35.7075769530614,
                38.8836309554631,
                42.0544164128268,
                45.2210650159239,
                48.3844038605504,
                51.5450520425884,
                54.7034825076868,
                57.8600629728451,
                61.0150837723061,
                64.1687772729670,
                67.3213317037490,
                70.4729011938089,
            ],
        ]
    )

    return precomputed_zeros[angular_l][radial_n]


def real_spherical_harmonics_m0(
    cos_theta: torch.Tensor,
    angular_l: int = 0,
) -> torch.Tensor:
    r"""Compute the m = 0 compoent of the real spherical harmonics.
    $$Y_{00} = \frac{1}{\sqrt{4\pi}}$$
    $$Y_{10} = \sqrt{\frac{3}{4\pi}}\cos\theta$$
    $$Y_{20} = \sqrt{\frac{5}{16\pi}}(3\cos^2\theta - 1)$$
    $$Y_{30} = \sqrt{\frac{7}{16\pi}}(5\cos^3\theta - 3\cos\theta)$$
    $$Y_{40} = \sqrt{\frac{9}{256\pi}}(35\cos^4\theta - 30\cos^2\theta + 3)$$
    Refer: https://en.wikipedia.org/wiki/Table_of_spherical_harmonics for more details.

    Args:
        cos_theta (torch.Tensor): The cosine of the angle theta.
        angular_l (int): The degree of the spherical harmonics.

    Returns:
        torch.Tensor: The spherical harmonics of angular momentum l
            and magnetic quantum number m = 0. Return shape the same as cos_theta.
    """
    if angular_l < 0:
        raise ValueError("lmax must be greater than or equal to 0.")

    if angular_l > 4:
        raise ValueError("lmax must be less than or equal to 4.")

    result = None
    if angular_l == 0:
        result = torch.ones_like(cos_theta) / torch.sqrt(torch.tensor(4.0 * torch.pi))
    elif angular_l == 1:
        result = torch.sqrt(torch.tensor(3.0 / (4.0 * torch.pi))) * cos_theta
    elif angular_l == 2:
        result = torch.sqrt(torch.tensor(5.0 / (16.0 * torch.pi))) * (
            3.0 * cos_theta**2 - 1.0
        )
    elif angular_l == 3:
        result = torch.sqrt(torch.tensor(7.0 / (16.0 * torch.pi))) * (
            5.0 * cos_theta**3 - 3.0 * cos_theta
        )
    elif angular_l == 4:
        result = torch.sqrt(torch.tensor(9.0 / (256.0 * torch.pi))) * (
            35.0 * cos_theta**4 - 30.0 * cos_theta**2 + 3.0
        )
    else:
        raise ValueError("l must be less than or equal to 4.")

    return result


class SphericalHarmonicAndRadialBasis(nn.Module):
    """
    Represents the radii and angles of three-body interactions using a combined radial
    and spherical harmonic basis, following the DimeNet formalism
    (see https://arxiv.org/abs/2003.03123, Eq. 6).

    Note:
        - The original DimeNet paper applies an envelope polynomial to both the
            radial and angular basis functions.
        - In contrast, the M3GNet paper uses a different envelope polynomial and
            applies it only during the computation of three-body interactions.
        - Therefore, this class does not apply the envelope polynomial;
            it should be handled separately as needed.

    Args:
        max_angular_l (int): Maximum angular momentum quantum number (l).
        max_radial_n (int): Maximum radial quantum number (n).
        cutoff (float): Cutoff radius.
    """

    def __init__(
        self, max_angular_l: int = 4, max_radial_n: int = 4, cutoff: float = 5.0
    ):
        super().__init__()

        if max_angular_l < 0 or max_angular_l > 5:
            raise ValueError("max_angular_l must be between 0 and 5.")

        if max_radial_n < 0 or max_radial_n > 5:
            raise ValueError("max_radial_n must be between 0 and 5.")

        if cutoff <= 0:
            raise ValueError("cutoff must be greater than 0.")

        self.max_angular_l = max_angular_l
        self.max_radial_n = max_radial_n
        self.cutoff = cutoff

    def forward(self, r: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute the spherical harmonic and radial basis functions.

        Args:
            r (torch.Tensor): The radial distance. Shape: (num_edges,)
            theta (torch.Tensor): The angle theta. Shape: (num_edges,)

        Returns:
            torch.Tensor: The spherical harmonic and radial basis functions.
            Return shape: (num_edges, max_radial_n * max_angular_l)
        """
        r_scaled = r / self.cutoff

        shrb = torch.zeros(self.max_radial_n, self.max_angular_l, r.shape[0])
        for angluar_l in range(self.max_angular_l):
            angular_ = real_spherical_harmonics_m0(torch.cos(theta), angluar_l)
            for radial_n in range(self.max_radial_n):
                z_ln = spherical_bessel_zeros(angluar_l, radial_n)
                prefactor = torch.tensor(
                    torch.sqrt(torch.tensor(2.0 / self.cutoff**3))
                    / spherical_jn(angluar_l + 1, z_ln)
                )
                spherical_jn_results = (
                    prefactor
                    * torch.tensor(spherical_jn(angluar_l, z_ln * r_scaled))
                    * angular_
                )
                spherical_jn_results = spherical_jn_results.squeeze(-1)
                shrb[radial_n, angluar_l, :] = spherical_jn_results

        # move the last index to the first index but keep the other
        # two indices in the same order
        return shrb.permute(2, 0, 1).reshape(
            -1, (self.max_radial_n) * (self.max_angular_l)
        )


class SmoothBesselBasis(nn.Module):
    def __init__(
        self,
        cutoff: float,
        max_radial_n: int = 3,
        device: str | torch.device = "cpu",
    ):
        super().__init__()

        self.cutoff = cutoff
        self.max_radial_n = max_radial_n
        self.device = device

        # generate temporary parameters
        en = torch.zeros(max_radial_n, dtype=torch.float32, device=device)
        for n in range(max_radial_n):
            en[n] = n**2 * (n + 2) ** 2 / (4 * (n + 1) ** 4 + 1)
        self.register_buffer("en", en)

        dn = torch.arange(max_radial_n, dtype=torch.float32, device=device)
        for n in range(max_radial_n):
            dn[n] = 1.0 - en[n] / dn[n - 1] if n > 0 else 1.0
        self.register_buffer("dn", dn)

        n_plus_1_factor = [
            (n + 1) * torch.pi / self.cutoff for n in range(max_radial_n)
        ]
        n_plus_2_factor = [
            (n + 2) * torch.pi / self.cutoff for n in range(max_radial_n)
        ]
        self.register_buffer(
            "n_plus_1_factor",
            torch.tensor(n_plus_1_factor, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "n_plus_2_factor",
            torch.tensor(n_plus_2_factor, dtype=torch.float32, device=device),
        )

        fn_factor = [
            (-1) ** n
            * torch.sqrt(torch.tensor(2))
            * torch.pi
            / self.cutoff**1.5
            * (n + 1)
            * (n + 2)
            / torch.sqrt(torch.tensor((n + 1) ** 2 + (n + 2) ** 2))
            for n in range(max_radial_n)
        ]
        self.register_buffer(
            "fn_factor", torch.tensor(fn_factor, dtype=torch.float32, device=device)
        )

        # generate parameters
        self.to(device)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate the smooth bessel basis functions at a give coordinate x.

        Args:
            x: (num_edges,)

        Returns:
            (num_edges, max_radial_n)
        """

        bessel_basis = torch.zeros(x.shape[0], self.max_radial_n, device=self.device)

        for n in range(self.max_radial_n):
            r_with_plus1_factor = x * self.n_plus_1_factor[n]
            r_with_plus2_factor = x * self.n_plus_2_factor[n]

            # please note, in numpy and torch, sinc(x) = sin(pi * x) / (pi * x)
            # but I decide to use the implementation in numpy/torch,
            # because it is more stable especially when x is close to 0
            fn = self.fn_factor[n] * (
                torch.sinc(r_with_plus1_factor / torch.pi)
                + torch.sinc(r_with_plus2_factor / torch.pi)
            )
            if n == 0:
                bessel_basis[:, n] = fn
            else:
                bessel_basis[:, n] = (
                    1
                    / torch.sqrt(self.dn[n])
                    * (
                        fn
                        + torch.sqrt(self.en[n] / self.dn[n - 1])
                        * bessel_basis[:, n - 1]
                    )
                )
        return bessel_basis
