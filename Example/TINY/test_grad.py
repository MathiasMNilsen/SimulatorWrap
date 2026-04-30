import os
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from subsurface.multphaseflow.jutul_darcy import JutulDarcy
from misc.structures import PETDataFrame, PETStateArray 

datetimes = [
    datetime(2023, 2, 5),
    datetime(2024, 3, 11),
    datetime(2025, 4, 15),
    datetime(2026, 5, 20),
    datetime(2027, 6, 24),
    datetime(2028, 7, 28),
    datetime(2029, 9, 1),
    datetime(2030, 10, 6),
    datetime(2031, 11, 10),
    datetime(2032, 12, 14),
]

datatype = [
    'WOPR:PRO1', 'WOPR:PRO2', 'WOPR:PRO3', 
    'WWPR:PRO1', 'WWPR:PRO2', 'WWPR:PRO3', 
    'WWIR:INJ1'
]

kwargs = {
    'parallel': 5,
    'reporttype': 'dates',
    'reportpoint': datetimes,
    'runfile': 'RUNFILE.mako',
    'startdate': datetime(2022, 1, 1),
    'datatype': datatype,
    'adjoint_pbar': False,
    'adjoints': {'WOPR': {'steps': [datetime(2032, 12, 14)], 'wellID': 'PRO2', 'parameters': 'log_permx'}},
}

def test_finit_diff(calc_finite_diff=True, calc_adjoint=True):
    os.makedirs('TEST/FD', exist_ok=True)

    nc = 5
    eps = 1e-4

    log_permx = np.log(np.load('PERMX.npy'))

    # ----------------------------
    # FINITE-DIFFERENCE GRADIENT
    # ----------------------------
    if calc_finite_diff:
        kwargs_copy = dict(kwargs)  # Avoid mutating original kwargs
        kwargs_copy.pop('adjoints', None)  # Remove adjoint options for finite-difference run
        simulator = JutulDarcy(kwargs_copy)

        # Compute finite-difference gradient
        log_perm_minus = np.tile(log_permx[:, None], (1, nc))
        log_perm_plus  = np.tile(log_permx[:, None], (1, nc))
        
        for i in range(nc):
            log_perm_minus[i, i] -= eps
            log_perm_plus[i, i]  += eps

        
                # Minus ensemble
        inputs_minus = [{'log_permx': log_perm_minus[:, i]} for i in range(nc)]
        out_minus = simulator(inputs_minus)
        df_minus = PETDataFrame.merge_dataframes(out_minus)
        df_minus.to_pickle('TEST/FD/df_eps_minus.pkl')

        # Plus ensemble
        inputs_plus = [{'log_permx': log_perm_plus[:, i]} for i in range(nc)]
        out_plus = simulator(inputs_plus)
        df_plus = PETDataFrame.merge_dataframes(out_plus)
        df_plus.to_pickle('TEST/FD/df_eps_plus.pkl')
    

    # ------------------------
    # ADJOINT
    # ------------------------
    if calc_adjoint:
        simulator = JutulDarcy(kwargs)
        _, adjoint = simulator({'log_permx': log_permx})
        adj_df = PETDataFrame.from_pandas(adjoint)
        adj_df.to_pickle('TEST/FD/adj_df.pkl')

    
    col = f'WOPR:{kwargs["adjoints"]["WOPR"]["wellID"]}'
    idx = kwargs['adjoints']['WOPR']['steps'][0]

    df_minus = PETDataFrame.from_pickle('TEST/FD/df_eps_minus.pkl')
    df_plus = PETDataFrame.from_pickle('TEST/FD/df_eps_plus.pkl')
    df_minus.is_ensemble = True
    df_plus.is_ensemble = True

    grad_fd_df = (df_plus - df_minus)/(2*eps)
    grad_fd = grad_fd_df.loc[idx][col]
    grad_fd = np.asarray(grad_fd)[:nc]

    adj_df = PETDataFrame.from_pickle('TEST/FD/adj_df.pkl')
    grad_adj = np.asarray(adj_df.loc[idx][(col, 'log_permx')])[:nc]

    print("Finite-difference gradient:", grad_fd)
    print("Adjoint gradient:", grad_adj)
    print("Relative difference:", np.linalg.norm(grad_fd - grad_adj) / np.linalg.norm(grad_fd))


def test_sens_matrix(run=True):

    os.makedirs('TEST/SENS_MATRIX', exist_ok=True)

    ne = 10_000
    if run: 
        np.random.seed(29_01_1983)

        pinfo = {
            'nx': 10,
            'ny': 10,
            'nz': 2,
            'vario': ['sph', 'sph'],
            'mean': 200*[4.0],
            'variance': [1.0, 1.0],
            'corr_length': [10.0, 10.0],
            'aniso': [1.0, 1.0],
            'angle': [0.0, 0.0],
        }
        prior_log_permx_ensemble = PETStateArray.generate_from_prior_info(
            prior_info = {'log_permx': pinfo},
            ne=ne
        )
        np.save('TEST/SENS_MATRIX/prior_log_permx_ensemble.npy', prior_log_permx_ensemble.data)

        # Run simulator on ensemble
        simulator = JutulDarcy(kwargs)
        inputs = [{'log_permx': prior_log_permx_ensemble[:, i]} for i in range(ne)]
        results, adjoint = simulator(inputs)
        results = PETDataFrame.merge_dataframes(results)
        adjoint = PETDataFrame.merge_dataframes(adjoint)
        print(results)
        results.to_pickle('TEST/SENS_MATRIX/results.pkl')
        adjoint.to_pickle('TEST/SENS_MATRIX/adjoint.pkl')

    
    # Analyze results
    col = f'WOPR:{kwargs["adjoints"]["WOPR"]["wellID"]}'
    idx = kwargs['adjoints']['WOPR']['steps'][0]
    results = PETDataFrame.from_pickle('TEST/SENS_MATRIX/results.pkl').loc[idx, col]
    adjoint = PETDataFrame.from_pickle('TEST/SENS_MATRIX/adjoint.pkl').loc[idx, (col, 'log_permx')]

    nx = 200
    ny = 1
    enX = np.load('TEST/SENS_MATRIX/prior_log_permx_ensemble.npy')
    enY = results[np.newaxis, :]
    enG = adjoint[np.newaxis, :, :]

    assert enX.shape == (nx, ne)
    assert enY.shape == (ny, ne)
    assert enG.shape == (ny, nx, ne)

    # Compute sensitivity matrix using ensemble gradients
    P = (np.eye(ne) - np.ones((ne,ne))/ne)/np.sqrt(ne-1)
    A = enX @ P
    Y = enY @ P
    Gbar = np.mean(enG, axis=-1)

    Cyx = Y @ A.T
    GbarCxx = Gbar @ A @ A.T


    # Stein's lemma tells us that:
    # E[G]Cxx = Cyx   --->   Gbar @ A @ A.T ≈ Y @ A.T (for large ne)


    # Plotting
    err = []
    ens = np.logspace(1, np.log10(ne), num=15, dtype=int)
    Cxy_norm = []
    GbarCxx_norm = []
    for n in ens:
        P_n = (np.eye(n) - np.ones((n,n))/n)/np.sqrt(n-1)
        A_n = enX[:, :n] @ P_n
        Y_n = enY[:, :n] @ P_n
        Gbar_n = np.mean(enG[:, :, :n], axis=-1)
        
        GbarCxx_n = Gbar_n @ A_n @ A_n.T
        Cyx_n = Y_n @ A_n.T


        err.append(np.linalg.norm(GbarCxx_n-Cyx))
        Cxy_norm.append(np.linalg.norm(Cyx_n))
        GbarCxx_norm.append(np.linalg.norm(GbarCxx_n))
        

    # Plot norm of Cyx vs ensemble size
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)

    ax.plot(
        ens,
        Cxy_norm,
        color='#1f77b4',
        linewidth=2.5,
        marker='o',
        markersize=6,
        markerfacecolor='white',
        markeredgewidth=1.3,
        label=r'$\|C_{yx}\|$'
    )
    ax.plot(
        ens,
        GbarCxx_norm,
        color='#2ca02c',
        linewidth=2.5,
        marker='s',
        markersize=6,
        markerfacecolor='white',
        markeredgewidth=1.3,
        label=r'$\|\bar{G}C_{xx}\|$'
    )
    ax.plot(
        ens,
        err,
        color='#d62728',
        linewidth=2.5,
        linestyle='--',
        marker='^',
        markersize=6,
        markerfacecolor='white',
        markeredgewidth=1.3,
        label=r'$\|\bar{G}C_{xx} - C_{yx}\|$'
    )

    ax.set_xscale('log')
    ax.set_ylim(0, None)
    ax.set_xlabel('Ensemble size', fontsize=11)
    ax.set_ylabel(r'$L_2$-norm', fontsize=11)

    ax.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.45)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.6, alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.92)
    fig.tight_layout()
    plt.show()
        

    


if __name__ == "__main__":
    #test_finit_diff(calc_finite_diff=False, calc_adjoint=True)
    test_sens_matrix(run=False)