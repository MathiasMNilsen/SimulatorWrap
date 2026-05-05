import os
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from subsurface.multphaseflow.jutul_darcy import JutulDarcy
from misc.structures import PETDataFrame, PETStateArray 


def test_finite_difference_gradient(run_fda=True, run_adjoint=True, folder='TEST/FINITE_DIFF'):
    
    datapoint = 'WOPR:PRO2'
    date = datetime(2032, 12, 14)
    options = {
        'parallel': 50,
        'datatype': [datapoint],
        'reporttype': 'dates',
        'reportpoint': [date],
        'runfile': 'RUNFILE.mako',
        'startdate': datetime(2022, 1, 1),
        'adjoint_pbar': False,
        'adjoints': {'WOPR': 
            {'steps': [date], 'wellID': 'PRO2', 'parameters': 'permx'}
        },
    }

    os.makedirs(folder, exist_ok=True)

    # Load PERMX 
    permx = np.load('PERMX.npy')

    reps = 0.01 # Relevant perturbation size for finite difference approximation (1%)
    if run_fda: 
        # Calculate finite difference gradient
        kw = dict(options)
        kw.pop('adjoints')

        # Pertubed input
        permx_p = np.tile(permx[:, None], (1, permx.size))
        permx_m = np.tile(permx[:, None], (1, permx.size))
        for i in range(permx.size):
            permx_p[i, i] += reps*permx[i]
            permx_m[i, i] -= reps*permx[i]

        # Run simulator on perturbed inputs
        simulator = JutulDarcy(kw)
        inputs_p  = [{'log_permx': np.log(permx_p[:, i])} for i in range(permx.size)]
        inputs_m  = [{'log_permx': np.log(permx_m[:, i])} for i in range(permx.size)]
        results_p = simulator(inputs_p)
        results_m = simulator(inputs_m)

        # Convert results to PETDataFrame and save
        results_p = PETDataFrame.merge_dataframes(results_p)
        results_m = PETDataFrame.merge_dataframes(results_m)
        results_p.to_pickle(os.path.join(folder, 'results_p.pkl'))
        results_m.to_pickle(os.path.join(folder, 'results_m.pkl'))
    
    if run_adjoint:
        # Run simulator with adjoint to get gradient
        simulator = JutulDarcy(options)
        inputs = [{'log_permx': np.log(permx)}]
        results, adjoint = simulator(inputs)
        adjoint.to_pickle(os.path.join(folder, 'adjoint.pkl'))
    
    # Analyze results
    results_p = PETDataFrame.from_pickle(os.path.join(folder, 'results_p.pkl')).loc[date, datapoint]
    results_m = PETDataFrame.from_pickle(os.path.join(folder, 'results_m.pkl')).loc[date, datapoint]
    grad_fda = (results_p - results_m)/(2*reps*permx)
    grad_adj = PETDataFrame.from_pickle(os.path.join(folder, 'adjoint.pkl')).loc[date, (datapoint, 'permx')]

    relative_error = np.linalg.norm(grad_fda - grad_adj)/np.linalg.norm(grad_fda)
    print(f"Relative error: {relative_error:.4e}")
    print(f"Norm of FDA gradient: {np.linalg.norm(grad_fda):.4e}")
    print(f"Norm of adjoint gradient: {np.linalg.norm(grad_adj):.4e}")

    # Plot FDA vs adjoint gradient on grid
    nx = 10
    ny = 10
    nz = 2
    grad_fda_grid = grad_fda.reshape((nx, ny, nz), order='F')
    grad_adj_grid = grad_adj.reshape((nx, ny, nz), order='F')

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=140)
    cmap = 'YlGnBu_r'

    # im0: Finite difference gradient for layer 1
    im0 = axes[0, 0].imshow(grad_fda_grid[:, :, 0], cmap=cmap)
    axes[0, 0].set_title('FDA Gradient (Layer 1)', fontsize=11)
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # im1: Adjoint gradient for layer 1
    im1 = axes[0, 1].imshow(grad_adj_grid[:, :, 0], cmap=cmap)
    axes[0, 1].set_title('Adjoint Gradient (Layer 1)', fontsize=11)
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # im2: Finite difference gradient for layer 2
    im2 = axes[1, 0].imshow(grad_fda_grid[:, :, 1], cmap=cmap)
    axes[1, 0].set_title('FDA Gradient (Layer 2)', fontsize=11)
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # im3: Adjoint gradient for layer 2
    im3 = axes[1, 1].imshow(grad_adj_grid[:, :, 1], cmap=cmap)
    axes[1, 1].set_title('Adjoint Gradient (Layer 2)', fontsize=11)
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.tight_layout()
    fig.savefig(os.path.join(folder, 'fda_vs_adjoint_gradient.png'), dpi=300)
    #plt.show()


def test_sens_matrix_of_log_permx(run=True, folder='TEST'):

    datapoint = 'WOPR:PRO2'
    date = datetime(2032, 12, 14)
    options = {
        'parallel': 50,
        'datatype': [datapoint],
        'reporttype': 'dates',
        'reportpoint': [date],
        'runfile': 'RUNFILE.mako',
        'startdate': datetime(2022, 1, 1),
        'adjoint_pbar': False,
        'adjoints': {'WOPR': 
            {'steps': [date], 'wellID': 'PRO2', 'parameters': 'log_permx'}
        },
    }

    os.makedirs(folder, exist_ok=True)

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
            ne=ne,
            save=False
        )
        np.save(os.path.join(folder, 'prior_log_permx_ensemble.npy'), prior_log_permx_ensemble)

        # Run simulator on ensemble
        simulator = JutulDarcy(options)
        inputs = [{'log_permx': prior_log_permx_ensemble[:, i]} for i in range(ne)]
        results, adjoint = simulator(inputs)
        results = PETDataFrame.merge_dataframes(results)
        adjoint = PETDataFrame.merge_dataframes(adjoint)
        results.to_pickle(os.path.join(folder, 'results.pkl'))
        adjoint.to_pickle(os.path.join(folder, 'adjoint.pkl'))

    
    # Analyze results
    col = datapoint
    idx = date
    results = PETDataFrame.from_pickle(os.path.join(folder, 'results.pkl')).loc[idx, col]
    adjoint = PETDataFrame.from_pickle(os.path.join(folder, 'adjoint.pkl')).loc[idx, (col, 'log_permx')]

    nx = 200
    ny = 1
    enX = np.load(os.path.join(folder, 'prior_log_permx_ensemble.npy'))
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

    # -------------------------------------------------------------------
    # Stein's lemma tells us that:
    # E[G]Cxx = Cyx   --->   Gbar @ A @ A.T ≈ Y @ A.T (for large ne)
    # -------------------------------------------------------------------

    # Loop over different ensemble sizes and compute norms of Cyx, GbarCxx, and their difference
    Cxy_norm = []
    GbarCxx_norm = []
    err = []
    ens = np.logspace(1, np.log10(ne), num=10, dtype=int)
    for n in ens:
        P_n = (np.eye(n) - np.ones((n,n))/n)/np.sqrt(n-1)
        A_n = enX[:, :n] @ P_n
        Y_n = enY[:, :n] @ P_n
        Gbar_n = np.mean(enG[:, :, :n], axis=-1)
        
        GbarCxx_n = Gbar_n @ A_n @ A_n.T
        Cyx_n = Y_n @ A_n.T

        err.append(np.linalg.norm(GbarCxx_n-Cyx_n))
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
    fig.savefig(os.path.join(folder, 'sensitivity_matrix_convergence.png'), dpi=300)
    #plt.show()
    


if __name__ == "__main__":
    test_finite_difference_gradient(run_fda=False, run_adjoint=True, folder='TEST/TEMP')
    #test_sens_matrix_of_log_permx(run=True, folder='TEST/SENS_WOPR_LOG_PERMX_SUBSTATES')