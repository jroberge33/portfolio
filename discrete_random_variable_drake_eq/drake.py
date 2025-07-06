# import drv class
from drv import DRV


def main():
    # definition of distributions
    R = DRV(type='uniform', min=1.5, max=3, bins=5)
    f = DRV({0.94: 1.0})
    n = DRV(type='normal', mean=3, std=1, bins=5)
    f_1 = DRV(type='normal', mean=1, std=0.2, bins=5)
    f_i = DRV(type='normal', mean=0.5, std=0.1, bins=5)
    f_c = DRV(type='normal', mean=2, std=0.5, bins=5)
    L = DRV({100: 0.3, 500: 0.4, 1000: 0.2, 10000: 0.1})

    # Drake equation
    N = R * f * n * f_1 * f_i * f_c * L

    # plot distribution
    N.plot(xscale='log',
           title='Life in the Milky Way!',
           show_cumulative = True,
           savefig='drake_v2.png')

    # output estimated number of civilizations
    print(f"The estimated number of actively communicating extraterrestrial civilizations in the Milky Way is {N.E()}.")


main()