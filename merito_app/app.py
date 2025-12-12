import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from scipy import integrate
from scipy.stats import norm

# =========================================================

def PHI(es,em, params):
    γs=params["γs"]
    γm=params["γm"]
    γms=params["γms"]
    ζ=params["ζ"]
    value = (-1)*γs*((es)**2)/2+γms*(1-es)*(1-em)-γm*((em)**2)/2
    return value

def util(e, params):
    γs=params["γs"]
    γm=params["γm"]
    γms=params["γms"]
    ζ=params["ζ"]
    value = ζ*(e-e**2/2)
    return value

def eS_vec(alpha_array, params):
    γs=params["γs"]
    γm=params["γm"]
    γms=params["γms"]
    ζ=params["ζ"]
    alpha_array = np.asarray(alpha_array)
    val = (ζ**2 + (1-alpha_array)*ζ*γm - ((1-alpha_array)**2) * γms*(γm+γms)) / (-γms*(1-alpha_array)+γs*γm*((1-alpha_array)**2)+(γs+γm)*(1-alpha_array)*ζ + ζ**2)
    return np.clip(val, 0.0, 1.0)

def eM_vec(alpha_array, params):
    γs=params["γs"]
    γm=params["γm"]
    γms=params["γms"]
    ζ=params["ζ"]
    alpha_array = np.asarray(alpha_array)
    val = (ζ**2 + (1-alpha_array)*ζ*γs - ((1-alpha_array)**2) * γms*(γs+γms)) / (-γms*(1-alpha_array)+γs*γm*((1-alpha_array)**2)+(γs+γm)*(1-alpha_array)*ζ + ζ**2)
    return np.clip(val, 0.0, 1.0)

def X1_matrix(A, params):
    γs=params["γs"]
    γm=params["γm"]
    γms=params["γms"]
    ζ=params["ζ"]
    A = np.asarray(A)
    e_s = eS_vec(A, params)          # (k,)
    e_m = eM_vec(A, params)         # (k,)
    ES = e_s[np.newaxis, :]  # shape (1,k) -> subordinate type j
    EM = e_m[:, np.newaxis]  # shape (k,1) -> manager type i
    return PHI(ES, EM, params) + util(ES, params)  # broadcastingで (k,k)

def Total_profit(P, hP, A, params):
    P = np.asarray(P)
    hP = np.asarray(hP)
    A = np.array(A)
    e_m = eM_vec(A, params)
    Profit = hP @ util(e_m, params).T+ hP @ X1_matrix(A, params) @ P.T
    return float(Profit)
    

def compute_Gamma_matrix(P, A, n, params,
                         n_grid=400, L=6):
    """
    manager type = A[i] （行）
    subordinate type = A[j] （列）の遷移行列 Gamma_mat[i,j]
    """
    P = np.asarray(P)
    A = np.asarray(A)
    k = len(A)
    sigma = params["σ"]
    theta = params["θ"]

    X1_mat = X1_matrix(A, params)  # shape (k,k)

    Gamma_mat = np.zeros((k, k))

    for m in range(k):  # manager bias = A[m]
        x_m = X1_mat[m, :]          # subordinateごとのX1
        x_min, x_max = x_m.min(), x_m.max()

        # π のグリッドを作る（±Lσ をカバー）
        pi_min = theta - L*sigma + x_min
        pi_max = theta + L*sigma + x_max
        pi_grid = np.linspace(pi_min, pi_max, n_grid)

        # shape (n_grid, k)
        diff = pi_grid[:, None] - x_m[None, :]

        pdf = norm.pdf(diff, loc=theta, scale=sigma)   # f(π − X1)
        cdf = norm.cdf(diff, loc=theta, scale=sigma)   # F(π − X1)

        # H(π) = Σ_j P_j F(π − X1_mj)
        H_vals = (P * cdf).sum(axis=1)                 # shape (n_grid,)

        # 各タイプ j についての integrand: P[j]*f * n * H^{n-1}
        integrand = pdf * P   # broadcasting → (n_grid, k)
        integrand *= n * (H_vals**(n-1))[:, None]

        # 台形公式で積分して Gamma_mat[m, :]
        Gamma_mat[m, :] = np.trapezoid(integrand, pi_grid, axis=0)

    return Gamma_mat

def updated_distribution(hP, Gamma_mat):
    hP = np.asarray(hP)
    new_raw = hP @ Gamma_mat      # shape (k,)
    new_hP  = new_raw / new_raw.sum()
    return new_hP

def fosd_relation(P, Q):
    """
    - 戻り値  1 : Q が P を一次確率支配（＝バイアス「増加」側）
    - 戻り値 -1 : P が Q を一次確率支配（＝バイアス「減少」側）
    - 戻り値  0 : どちらも支配しない
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if P.shape != Q.shape:
        return 0

    if P[0] > Q[0]:
        z = 1   # Q ≻_FSD P
    else:
        z = -1  # P ≻_FSD Q

    csum_P = 0.0
    csum_Q = 0.0
    for i in range(len(P)):
        csum_P += P[i]
        csum_Q += Q[i]
        if z * (csum_P - csum_Q) < 0:
            return 0
    return z

# ============================
#  シミュレーション関数
# ============================

def simulate_one_path(num_periods, team_size, grid_size, params):
    """
    1 本の OLG 経路を生成:
      - 下位分布 G と supp(G)=A をランダムに生成
      - G からスタートして managerial bias 分布を num_periods 期分進める
    戻り値:
      A: shape (K,)
      sub_dist: shape (K,)
      manager_dists: shape (num_periods+1, K)
      performances: shape (num_periods+1,)
    """
    K = grid_size

    # サポート A: とりあえず均等格子。ランダムにしたければ np.sort(np.random.rand(K))
    A = np.sort(np.random.rand(K))

    P1pr = [random.random() for j in range(K)]
    sP1 = sum(P1pr)
    sub_dist = [P1pr[i]/sP1 for i in range(K)]

    manager_dists = []
    performances = []
    Gamma_mat = compute_Gamma_matrix(
            sub_dist,
            A,
            n=team_size,
            params=params,
        )
    # 初期 manager 分布は G と同じにしておく
    manager_dist = sub_dist.copy()

    for t in range(num_periods + 1):
        manager_dists.append(manager_dist)
        performances.append(Total_profit(sub_dist, manager_dist, A, params))

        # 次期の manager 分布を計算（最後の期では使われないがそのまま更新）

        manager_dist = updated_distribution(manager_dist, Gamma_mat)

    return A, sub_dist, np.array(manager_dists), np.array(performances)


def run_simulations(num_sims, num_periods, team_size, grid_size, params):
    """
    複数本の経路をシミュレーションし、結果を list で返す。
    各要素は dict:
      {"A": A, "G": sub_dist, "managers": manager_dists, "perf": performances}
    """
    results = []
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    for i in range(num_sims):
        A, G, managers, perf = simulate_one_path(
            num_periods=num_periods,
            team_size=team_size,
            grid_size=grid_size,
            params=params,
        )
        results.append(
            {
                "A": A,
                "G": G,
                "managers": managers,
                "perf": perf,
            }
        )

        progress = (i + 1) / num_sims
        progress_bar.progress(progress)
        status_text.text(f"Simulation {i+1} / {num_sims} finished")

    status_text.text("Simulations completed.")
    return results


def compute_scatter_data(results, t_from, t_to):
    """
    指定した 2 期 t_from, t_to について、
    平均バイアスとパフォーマンスの変化を計算し、
    FOSD の向きごとにグループ分けして返す。
    """
    inc_x, inc_y, inc_z = [], [], []
    dec_x, dec_y, dec_z = [], [], []
    oth_x, oth_y, oth_z = [], [], []

    for res in results:
        A = res["A"]
        managers = res["managers"]
        perf = res["perf"]

        P_before = managers[t_from]
        P_after = managers[t_to]

        # 平均バイアス
        mean_before = float(P_before @ A)
        mean_after = float(P_after @ A)
        delta_mean = (mean_after - mean_before) / mean_before if mean_before != 0 else 0.0

        #バイアスの分散
        var_before = float(P_before @ (A - mean_before) ** 2)
        var_after = float(P_after @ (A - mean_after) ** 2)
        delta_var = (var_after - var_before) / var_before if var_before != 0 else 0.0

        # パフォーマンス
        perf_before = float(perf[t_from])
        perf_after = float(perf[t_to])
        delta_perf = (perf_after - perf_before) / perf_before if perf_before != 0 else 0.0

        rel = fosd_relation(P_before, P_after)

        if rel == 1:
            inc_x.append(delta_mean)
            inc_y.append(delta_perf)
            inc_z.append(delta_var)
        elif rel == -1:
            dec_x.append(delta_mean)
            dec_y.append(delta_perf)
            dec_z.append(delta_var)
        else:
            oth_x.append(delta_mean)
            oth_y.append(delta_perf)
            oth_z.append(delta_var)

    groups = {
        "inc": (np.array(inc_x), np.array(inc_y), np.array(inc_z)),
        "dec": (np.array(dec_x), np.array(dec_y), np.array(dec_z)),
        "oth": (np.array(oth_x), np.array(oth_y), np.array(oth_z)),
    }
    return groups


def add_regression(ax, xs, ys, label_suffix=""):
    """単純な一次回帰直線を図に追加（点が2個以上あるときだけ）"""
    if xs.size < 2:
        return
    coeffs = np.polyfit(xs, ys, 1)
    x_line = np.linspace(xs.min(), xs.max(), 100)
    y_line = coeffs[0] * x_line + coeffs[1]
    ax.plot(x_line, y_line, linestyle="--", label=f"OLS{label_suffix}")

# ============================
#  Streamlit UI
# ============================

st.title("Dynamics of Coordination-Neglect Bias in OLG Organizations")
st.markdown(
    r'''This app simulates the dynamics of managerial bias distributions in an overlapping generations (OLG) organization model, following the framework described in the paper, _"Meritocratic Selection and Dynamics of Coordination Neglect in Hierarchical Organizations"_ by Kohei Daido and Tomoya Tajika.''')
st.markdown(
    r''' 
    Users can adjust model parameters, run simulations, and visualize how changes in managerial bias distributions affect overall organizational performance.'''
)

# ---- 高度なパラメータ（トグルで隠す） ----
with st.expander("Model Parameters", expanded=False):
    st.write("Model parameters: We consider the following effort-performance function")
    s1 = r"""v(e_s, e_m) = -\eta_s e_s^2/2 + \eta_{ms} (1 - e_s)(1 - e_m) - \eta_m e_m^2/2"""
    st.latex(s1)
    s2 = r"""u(e) = \zeta \left( e - \frac{e^2}{2} \right)"""
    st.latex(s2 )
    cola, colb, colc, cold = st.columns(4)
    with cola:
        ζ = st.number_input("ζ", min_value=0.01, max_value=1.0, value=1.0, step=0.01)
    with colb:
        γms = st.number_input("η_ms", min_value=0.01, max_value=1.0, value=1.0, step=0.01)
    with colc:
        γm = st.number_input("η_m", min_value=0.01, max_value=1.0, value=0.05, step=0.01)
    with cold:
        γs = st.number_input("η_s", min_value=0.01, max_value=1.0, value=0.05, step=0.01)
    
    st.write("Noise distribution (dist. of θ) parameters: follows N(μ, σ²)")
    cole, colf = st.columns(2)
    with cole:
        σ = st.number_input("σ", min_value=0.01, max_value=1.0, value=.01, step=0.01)
    with colf:
        θ = st.number_input("μ", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    # 将来拡張しやすいよう dict にまとめて渡す
    params = {
        "γs": γs,
        "γm": γm,
        "γms": γms,
        "ζ": ζ,
        "σ": σ,
        "θ": θ,
    }

# ---- メイン設定（常に見える） ----
st.subheader("Main Settings for Simulation")

col1, col2, col3, col4 = st.columns(4)
with col1:
    team_size = st.number_input("team size (n)", min_value=2, max_value=10_000, value=50, step=1)
with col2:
    num_periods = st.number_input("periods", min_value=1, max_value=200, value=10, step=1)
with col3:
    grid_size = st.number_input("size of supp(G)", min_value=3, max_value=300, value=50, step=1)
with col4:
    num_sims = st.number_input("number of simulation", min_value=10, max_value=2000, value=300, step=10)

if "results" not in st.session_state:
    st.session_state.results = None
    st.session_state.num_periods = None

run_button = st.button("Run Simulations")

if run_button:


    st.session_state.results = run_simulations(
        num_sims=num_sims,
        num_periods=num_periods,
        team_size=team_size,
        grid_size=grid_size,
        params=params,
    )
    st.session_state.num_periods = num_periods

# ---- 結果の可視化 ----
if st.session_state.results is not None:
    st.subheader("Visualization of Dynamics by 2 period Change Rates")

    max_T = st.session_state.num_periods

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        t_from = st.number_input("period t (base)", min_value=0, max_value=max_T - 1, value=0, step=1)
    with col_t2:
        t_to = st.number_input("period s (comparizon)", min_value=t_from + 1, max_value=max_T, value=min(5, max_T), step=1)
    
    x_choice = st.radio("Comparizon", ["Δmean bias vs ΔPerformance", "Δvar bias vs ΔPerformance", "Δmean bias vs Δvar bias"], horizontal=True)


    show_reg = st.checkbox("Show regression lines for each FOSD group", value=True)

    groups = compute_scatter_data(st.session_state.results, t_from, t_to)

    # 件数の表示
    st.write(
        f"FOSD increase: {groups['inc'][0].size}, "
        f"decrease: {groups['dec'][0].size}, "
        f"No FOSD relation: {groups['oth'][0].size}"
    )


    inc_x, inc_y, inc_z = groups["inc"]
    dec_x, dec_y, dec_z = groups["dec"]
    oth_x, oth_y, oth_z = groups["oth"]


    fig, ax = plt.subplots()
    if x_choice == "Δmean bias vs ΔPerformance":
        ax.set_xlabel("Change rate in mean managerial bias")
        ax.set_ylabel("Change rate in overall performance")
        if inc_x.size > 0:
            ax.scatter(inc_x, inc_y, s = 1, label="FOSD increase")
        if show_reg:
            add_regression(ax, inc_x, inc_y, label_suffix=" (increase)")

        if dec_x.size > 0:
            ax.scatter(dec_x, dec_y, s = 1, label="FOSD decrease")
        if show_reg:
            add_regression(ax, dec_x, dec_y, label_suffix=" (decrease)")

        if oth_x.size > 0:
            ax.scatter(oth_x, oth_y, s = 1, label="No FOSD relation")
    elif x_choice == "Δvar bias vs ΔPerformance":
        ax.set_xlabel("Change rate in variance of managerial bias")
        ax.set_ylabel("Change rate in overall performance")
        if inc_z.size > 0:
            ax.scatter(inc_z, inc_y, s = 1, label="FOSD increase")
        if show_reg:
            add_regression(ax, inc_z, inc_y, label_suffix=" (increase)")

        if dec_z.size > 0:
            ax.scatter(dec_z, dec_y, s = 1, label="FOSD decrease")
        if show_reg:
            add_regression(ax, dec_z, dec_y, label_suffix=" (decrease)")
            
        if oth_z.size > 0:
            ax.scatter(oth_z, oth_y, s = 1, label="No FOSD relation")        
    else:  # Δmean bias vs Δvar bias
        ax.set_xlabel("Change rate in mean managerial bias")
        ax.set_ylabel("Change rate in variance of managerial bias")
        if inc_x.size > 0:
            ax.scatter(inc_x, inc_z, s = 1, label="FOSD increase")
        if show_reg:
            add_regression(ax, inc_x, inc_z, label_suffix=" (increase)")

        if dec_x.size > 0:
            ax.scatter(dec_x, dec_z, s = 1, label="FOSD decrease")
        if show_reg:
            add_regression(ax, dec_x, dec_z, label_suffix=" (decrease)")
            
        if oth_x.size > 0:
            ax.scatter(oth_x, oth_z, s = 1, label="No FOSD relation")   
    ax.set_title(f"{t_to}-period after change rate (compare t={t_from} → {t_to}), n={team_size}, supp(G)={grid_size}")


    



    ax.legend()
    st.pyplot(fig)

else:
    st.info("Please run simulations to see the results.")
