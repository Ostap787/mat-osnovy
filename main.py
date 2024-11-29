from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt

n = 11
N = 100*n


def f(t):
    return t**((2*n+1)/3)


def F(wk):
    def real_integrand(t):
        return np.real(f(t)) * np.cos(-wk * np.pi * t)

    def imag_integrand(t):
        return np.real(f(t)) * np.sin(-wk * np.pi * t)

    # Обмежуємо межі інтеграції
    real, _ = quad(real_integrand, -10, 10, limit=100)
    imag, _ = quad(imag_integrand, -10, 10, limit=100)

    return real, imag



def specter(real, imag):
    return np.sqrt(real**2 + imag**2)


def main():
    T_values = [4, 8, 16, 32, 64, 128]
    k_values = np.arange(0, 21, 1)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    for i, T in enumerate(T_values):
        Re_values = []

        for k in k_values:
            wk = 2 * np.pi * k / T
            real, imag = F(wk)
            Re_values.append(real)

        row, col = divmod(i, 2)

        axes[row, col].stem(k_values, Re_values, basefmt=" ",
                            label=f"Re(F(w_k)), T = {T}")
        axes[row, col].set_xlabel("k")
        axes[row, col].set_ylabel("Re(F(w_k))")
        axes[row, col].grid(True)
        axes[row, col].legend()

    plt.show()

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    for i, T in enumerate(T_values):
        Re_values = []
        Amplitude_values = []

        for k in k_values:
            wk = 2 * np.pi * k / T
            real, imag = F(wk)
            Re_values.append(real)
            Amplitude_values.append(specter(real, imag))

        row, col = divmod(i, 2)

        axes[row, col].stem(k_values, Amplitude_values, basefmt=" ", linefmt='orange',
                            markerfmt='o', label=f"|F(w_k)|, T = {T}")
        axes[row, col].set_xlabel("k")
        axes[row, col].set_ylabel("|F(w_k)|")
        axes[row, col].grid(True)
        axes[row, col].legend()

    plt.show()

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    for i, T in enumerate(T_values):
        Re_values = []

        for k in k_values:
            wk = 2 * np.pi * k / T
            real, imag = F(wk)
            Re_values.append(real)

        row, col = divmod(i, 2)

        axes[row, col].plot(k_values, Re_values, marker='o',
                            linestyle='-', label=f"Re(F(w_k)), T = {T}")
        axes[row, col].set_xlabel("k")
        axes[row, col].set_ylabel("Re(F(w_k))")
        axes[row, col].grid(True)
        axes[row, col].legend()

    plt.show()

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    for i, T in enumerate(T_values):
        Amplitude_values = []

        for k in k_values:
            wk = 2 * np.pi * k / T
            real, imag = F(wk)
            Amplitude_values.append(specter(real, imag))

        row, col = divmod(i, 2)

        axes[row, col].plot(k_values, Amplitude_values, marker='o',
                            linestyle='-', color='orange', label=f"|F(w_k)|, T = {T}")
        axes[row, col].set_xlabel("k")
        axes[row, col].set_ylabel("|F(w_k)|")
        axes[row, col].grid(True)
        axes[row, col].legend()

    plt.show()

    plt.figure(figsize=(10, 6))
    for T in T_values:
        Re_values = [F(2 * np.pi * k / T)[0] for k in k_values]
        plt.plot(k_values, Re_values, marker='o',
                 linestyle='-', label=f"T = {T}")
    plt.xlabel("k")
    plt.ylabel("Re(F(w_k))")
    plt.title("Real part of F(w_k) for different T values")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    for T in T_values:
        Amplitude_values = [specter(*F(2 * np.pi * k / T)) for k in k_values]
        plt.plot(k_values, Amplitude_values, marker='o',
                 linestyle='-', label=f"T = {T}")
    plt.xlabel("k")
    plt.ylabel("|F(w_k)|")
    plt.title("Amplitude |F(w_k)| for different T values")
    plt.grid(True)
    plt.legend()
    plt.show()


    t_values = np.linspace(-10, 30, 1000)  # Діапазон значень t
    f_values = [f(t) for t in t_values if
                t >= 0]

    plt.figure(figsize=(10, 6))
    plt.plot(t_values[t_values >= 0], f_values, label="f(t)", color='green')
    plt.xlabel("t")
    plt.ylabel("f(t)")
    plt.title("Input Function f(t)")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()