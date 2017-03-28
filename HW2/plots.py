import numpy as np
import matplotlib.pyplot as plt


def main():
    # plot_graph()
    xx = np.linspace(0, 10)
    yy_1 = 2 * xx - 2
    yy_2 = -1 * xx + 1
    yy_3 = 0.5 * xx + 3
    plt.plot(xx, yy_1, 'r-', label='z1 = 0')
    plt.plot(xx, yy_1 - 1, 'm-', label='f(z1) = 1')
    plt.plot(xx, yy_2, 'b-', label='z2 = 0')
    plt.plot(xx, yy_2 - 1, 'c-', label='f(z2) = 1')
    plt.plot(xx, yy_3, 'k-', label='F(z; theta) = 0')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    main()
