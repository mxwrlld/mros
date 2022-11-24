import numpy as np
import matplotlib.pyplot as plt
import utils.constants as const
from utils.vector_generator import generate_norm_vector
from modules.bayes import *
from modules.fisher import Fisher
from modules.stdmin import STDMin
from modules.robbinsmonro import RobbinsMonro


def calc_alpha(classify_sequence: np.ndarray, class_type: int):
    return len(classify_sequence[classify_sequence == class_type]) / len(classify_sequence)


def calc_bayes(xs, X_1, X_2, M_i, M_j, B_i, B_j):
    ys_b = None
    if np.array_equal(B_i, B_j):
        ys_b = d_lj_equal_B(xs, B_i, M_i, M_j, 0.5, 0.5)
    else:
        ys_b = d_lj_different_B(xs, B_i, B_j, M_i, M_j)
    Bs, Ms = np.array([B_i, B_j]), np.array([M_i, M_j])
    Ps = np.array([0.5, 0.5])
    p_0 = calc_alpha(classify_vectors(X_1, Bs, Ms, Ps), 1)
    p_1 = calc_alpha(classify_vectors(X_2, Bs, Ms, Ps), 0)
    return ys_b, p_0, p_1


def calc_fisher(xs, X_1, X_2, M_i, M_j, B_i, B_j):
    fisher = Fisher(M_i, M_j, B_i, B_j)
    ys_f = fisher.calc_decisive_boundaries(xs)
    p_0 = calc_alpha(fisher.classify_vectors(X_1, 0, 1), 1)
    p_1 = calc_alpha(fisher.classify_vectors(X_2, 1, 0), 0)
    return ys_f, p_0, p_1


def calc_stdmin(xs, X_1, X_2):
    m = 400
    stdmin = STDMin(m, X_1, X_2)
    ys_std = stdmin.calc_decisive_boundaries(xs)
    p_0 = calc_alpha(stdmin.classify_vectors(X_1, 0, 1), 1)
    p_1 = calc_alpha(stdmin.classify_vectors(X_2, 1, 0), 0)
    return ys_std, p_0, p_1


def calc_robbinsmonro(xs, X_1, X_2, m, betta, initial_w):
    rm = RobbinsMonro(m, betta, initial_w, X_1, X_2)
    index = rm.w_length - 1
    ys_rm = rm.calc_decisive_boundaries(xs, index)
    p_0 = calc_alpha(rm.classify_vectors(X_1, index, 0, 1), 1)
    p_1 = calc_alpha(rm.classify_vectors(X_2, index, 1, 0), 0)
    return ys_rm, p_0, p_1


def demonstration_plot(title: str, xs, x_1, x_2, name_ys: dict, isDiffBayes: bool = False):
    plt.title(title)
    plt.scatter(x_1[0], x_1[1], c="blue")
    plt.scatter(x_2[0], x_2[1], c="orange")
    colors = ["green", "red", "black", "magenta", "yellow", "cyan", "pink"]
    i = 0
    for name in name_ys:
        if isDiffBayes:
            plt.plot(name_ys[name], xs, label=name, c=colors[i])
            isDiffBayes = False
        else:
            plt.plot(xs, name_ys[name], label=name, c=colors[i])
        i += 1
    plt.xlim([-4, 4])
    plt.legend()


def demonstration_errors(title, classifier_errors):
    print(title)
    for classifier in classifier_errors:
        print(f"\t{classifier}: Экспериментальные ошибки классификации для классов 0 и 1: ",
              classifier_errors[classifier][0], ", ", classifier_errors[classifier][1])


if __name__ == "__main__":
    N = 200
    M_1, M_2 = const.M_1, const.M_2
    B_1, B_2 = const.B_1, const.B_2
    Bs, Bs_e = np.array([B_1, B_2]), np.array([B_1, B_1])
    Ms = np.array([M_1, M_2])
    Ps = np.array([0.5, 0.5])
    xs = np.linspace(-3, 3, N)

    config = {
        "generate": False,
        "save": False,
        "endpoints": [1, 2, 3]
    }

    if config['generate']:
        Y_1 = generate_norm_vector(N, B_1, M_1)
        Y_2 = generate_norm_vector(N, B_1, M_2)
        X_1 = generate_norm_vector(N, B_1, M_1)
        X_2 = generate_norm_vector(N, B_2, M_2)
        if config["save"]:
            np.savetxt("4/data/Y_1.txt", Y_1)
            np.savetxt("4/data/Y_2.txt", Y_2)
            np.savetxt("4/data/X_1.txt", X_1)
            np.savetxt("4/data/X_2.txt", X_2)
    else:
        Y_1 = np.loadtxt("data/Y_1.txt")
        Y_2 = np.loadtxt("data/Y_2.txt")
        X_1 = np.loadtxt("data/X_1.txt")
        X_2 = np.loadtxt("data/X_2.txt")

    exp_res = dict()
    ys, p_0, p_1 = calc_bayes(xs, Y_1, Y_2, M_1, M_2, B_1, B_1)
    ys_d, p_0_d, p_1_d = calc_bayes(xs, X_1, X_2, M_1, M_2, B_1, B_2)
    exp_res["bayes"] = {
        "B_equal": {"ys": ys, "p0": p_0, "p1": p_1},
        "B_diff": {"ys": ys_d, "p0": p_0_d, "p1": p_1_d}
    }
    ys, p_0, p_1 = calc_fisher(xs, Y_1, Y_2, M_1, M_2, B_1, B_1)
    ys_d, p_0_d, p_1_d = calc_fisher(xs, X_1, X_2, M_1, M_2, B_1, B_2)
    exp_res["fisher"] = {
        "B_equal": {"ys": ys, "p0": p_0, "p1": p_1},
        "B_diff": {"ys": ys_d, "p0": p_0_d, "p1": p_1_d}
    }
    ys, p_0, p_1 = calc_stdmin(xs, Y_1, Y_2)
    ys_d, p_0_d, p_1_d = calc_stdmin(xs, X_1, X_2)
    exp_res["stdmin"] = {
        "B_equal": {"ys": ys, "p0": p_0, "p1": p_1},
        "B_diff": {"ys": ys_d, "p0": p_0_d, "p1": p_1_d}
    }
    m, betta, initial_w = 4000, 0.99, 1
    ys, p_0, p_1 = calc_robbinsmonro(
        xs, Y_1, Y_2, m, betta, initial_w)
    ys_d, p_0_d, p_1_d = calc_robbinsmonro(
        xs, X_1, X_2, m, betta, initial_w)
    exp_res["robbinsmonro"] = {
        "B_equal": {"ys": ys, "p0": p_0, "p1": p_1},
        "B_diff": {"ys": ys_d, "p0": p_0_d, "p1": p_1_d}
    }

    if 1 in config['endpoints']:
        title = "Равные корреляционные матрицы"
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        demonstration_plot(title, xs, Y_1, Y_2,
                           {
                               "Байес": exp_res["bayes"]["B_equal"]["ys"],
                               "Фишер": exp_res["fisher"]["B_equal"]["ys"]
                           })

        demonstration_errors(title, {
            "БК": (exp_res["bayes"]["B_equal"]["p0"], exp_res["bayes"]["B_equal"]["p0"]),
            "КФ": (exp_res["fisher"]["B_equal"]["p0"], exp_res["fisher"]["B_equal"]["p0"])})

        title = "Разные корреляционные матрицы"
        ax = fig.add_subplot(1, 2, 2)
        demonstration_plot(title, xs, X_1, X_2,
                           {
                               "Байес": exp_res["bayes"]["B_diff"]["ys"][0],
                               "Фишер": exp_res["fisher"]["B_diff"]["ys"]
                           },
                           isDiffBayes=True)
        demonstration_errors(title, {
            "БК": (exp_res["bayes"]["B_diff"]["p0"], exp_res["bayes"]["B_diff"]["p1"]),
            "КФ": (exp_res["fisher"]["B_diff"]["p0"], exp_res["fisher"]["B_diff"]["p1"])})

        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show()

    if 2 in config['endpoints']:
        title = "Равные корреляционные матрицы"
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        demonstration_plot(title, xs, Y_1, Y_2,
                           {
                               "Байес": exp_res["bayes"]["B_equal"]["ys"],
                               "Мин.СКО": exp_res["stdmin"]["B_equal"]["ys"],
                               "Фишер": exp_res["fisher"]["B_equal"]["ys"]
                           })

        demonstration_errors(title, {
            "БК": (exp_res["bayes"]["B_equal"]["p0"], exp_res["bayes"]["B_equal"]["p0"]),
            "СКО": (exp_res["stdmin"]["B_equal"]["p0"], exp_res["stdmin"]["B_equal"]["p0"]),
            "КФ": (exp_res["fisher"]["B_equal"]["p0"], exp_res["fisher"]["B_equal"]["p0"])
        })

        title = "Разные корреляционные матрицы"
        ax = fig.add_subplot(1, 2, 2)
        demonstration_plot(title, xs, X_1, X_2,
                           {
                               "Байес": exp_res["bayes"]["B_diff"]["ys"][0],
                               "Мин.СКО": exp_res["stdmin"]["B_diff"]["ys"],
                               "Фишер": exp_res["fisher"]["B_diff"]["ys"]
                           },
                           isDiffBayes=True)
        demonstration_errors(title, {
            "БК": (exp_res["bayes"]["B_diff"]["p0"], exp_res["bayes"]["B_diff"]["p1"]),
            "СКО": (exp_res["stdmin"]["B_diff"]["p0"], exp_res["stdmin"]["B_diff"]["p1"]),
            "КФ": (exp_res["fisher"]["B_diff"]["p0"], exp_res["fisher"]["B_diff"]["p1"])
        })

        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show()

    if 3 in config['endpoints']:
        title = "Равные корреляционные матрицы"
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        demonstration_plot(title, xs, Y_1, Y_2,
                           {
                               "Байес": exp_res["bayes"]["B_equal"]["ys"],
                               "Робинс-Монро": exp_res["stdmin"]["B_equal"]["ys"]
                           })

        demonstration_errors(title, {
            "БК": (exp_res["bayes"]["B_equal"]["p0"], exp_res["bayes"]["B_equal"]["p1"]),
            "РМ": (exp_res["robbinsmonro"]["B_equal"]["p0"], exp_res["robbinsmonro"]["B_equal"]["p1"])
        })

        title = "Разные корреляционные матрицы"
        ax = fig.add_subplot(1, 2, 2)
        demonstration_plot(title, xs, X_1, X_2,
                           {
                               "Байес": exp_res["bayes"]["B_diff"]["ys"][0],
                               "Робинс-Монро": exp_res["robbinsmonro"]["B_diff"]["ys"]
                           },
                           isDiffBayes=True)
        demonstration_errors(title, {
            "БК": (exp_res["bayes"]["B_diff"]["p0"], exp_res["bayes"]["B_diff"]["p1"]),
            "РМ": (exp_res["robbinsmonro"]["B_diff"]["p0"], exp_res["robbinsmonro"]["B_diff"]["p1"])
        })

        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show()

        # Исследоавние зависимости скорости сходимости итерационного процесса и
        # качества классификации от выбора начальных условий и выбора последовательности
        # корректирующих коэффициентов.
        x_1, x_2 = X_1, X_2
        m, n = 1000, 7
        fig = plt.figure()

        title = "Выбор начальных условий"
        ax = fig.add_subplot(1, 2, 1)
        plt.title(title)
        plt.scatter(x_1[0], x_1[1], c="blue")
        plt.scatter(x_2[0], x_2[1], c="orange")

        yss = []
        p0s, p1s = [], []
        if 1 == 1:
            betta = 0.6
            initials_ws = [int(1 * (10 ** i / 10)) for i in range(1, n + 1)]
            print("Выбор начальных условий")
            print("Вероятности ошибочной классификации: ")
            print(
                f"w_0{len(str(initials_ws[-1]))*' '}\tp_0\tp_1")
            print(
                (len(f"w_0{len(str(initials_ws[-1]))*' '}\tp_0\tp_1") + 10) * '-')

            for i in range(n):
                rm = RobbinsMonro(m, betta, initials_ws[i], x_1, x_2)
                ys = rm.calc_decisive_boundaries(xs)
                p_0 = calc_alpha(rm.classify_vectors(x_1, 0, 1), 1)
                p_1 = calc_alpha(rm.classify_vectors(x_2, 1, 0), 0)
                p0s.append(p_0)
                p1s.append(p_1)
                yss.append(ys)
                print(
                    f"{initials_ws[i]} {(len(str(initials_ws[-1])) - len(str(initials_ws[i])))*' '}\t{p_0}\t{p_1}")
                plt.plot(xs, ys, label=f"w_0: {str(initials_ws[i])}")

        plt.xlim([-4, 4])
        plt.legend()
        title = "Выбор последовательности корректирующих коэффициентов"
        ax = fig.add_subplot(1, 2, 2)
        plt.title(title)
        plt.scatter(x_1[0], x_1[1], c="blue")
        plt.scatter(x_2[0], x_2[1], c="orange")

        yss = []
        p0s, p1s = [], []
        if 1 == 1:
            initial_w = 1000
            # bettas = np.random.uniform(low=0.51, high=0.999, size=n)
            bettas = [round(0.5 + i * 0.5 / n, 3) for i in range(n)]
            print("Выбор последовательности корректирующих коэффициентов")
            print("Вероятности ошибочной классификации: ")
            print(
                f"betta{len(str(bettas[-1]))*' '}p_0\tp_1")
            print(
                (len(f"betta{len(str(bettas[-1]))*' '}\tp_0\tp_1") + 3) * '-')

            for i in range(n):
                rm = RobbinsMonro(m, bettas[i], initial_w, x_1, x_2)
                ys = rm.calc_decisive_boundaries(xs)
                p_0 = calc_alpha(rm.classify_vectors(x_1, 0, 1), 1)
                p_1 = calc_alpha(rm.classify_vectors(x_2, 1, 0), 0)
                p0s.append(p_0)
                p1s.append(p_1)
                yss.append(ys)
                print(
                    f"{bettas[i]} {(len(str(bettas[-1])) - len(str(bettas[i])))*' '}\t{p_0}\t{p_1}")
                plt.plot(xs, ys, label=f"betta: {str(bettas[i])}")
        plt.xlim([-4, 4])
        plt.legend()
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show()
