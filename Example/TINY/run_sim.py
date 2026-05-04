import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from subsurface.multphaseflow.jutul_darcy import JutulDarcy

# Report steps
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

# Data types to report
datatype = [
    'WOPR:PRO1', 
    'WOPR:PRO2', 
    'WOPR:PRO3', 
    'WWPR:PRO1', 
    'WWPR:PRO2', 
    'WWPR:PRO3', 
]

# Simulator settings and adjoint configuration
kwargs = {
    'reporttype': 'dates',
    'reportpoint': datetimes,
    'runfile': 'RUNFILE.mako',
    'startdate': datetime(2022, 1, 1),
    'datatype': datatype,
    'adjoint_pbar': True,
    'adjoints': {'WOPR': 
        {'steps': [datetime(2032, 12, 14)], 'wellID': 'PRO2', 'parameters': ['log_permx', 'permx']},
    },
}

def simulate():
    log_permx = np.log(np.load('PERMX.npy'))
    simulator = JutulDarcy(kwargs)
    res, grad = simulator({'log_permx': log_permx})
    return res, grad

def plot(res):
    """Plot production rates with enhanced styling."""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    fig, ax = plt.subplots(ncols=3, figsize=(12, 3), sharey=True)
    fig.patch.set_facecolor('white')
    
    colors = {'WOPR': '#2E86AB', 'WWPR': '#A23B72'}
    linestyles = {'WOPR': '-', 'WWPR': '--'}
    linewidths = {'WOPR': 2.5, 'WWPR': 2.2}
    
    for col in range(3):
        producer_num = col + 1
        
        # Plot well production rates
        ax[col].plot(
            res.index, res[f'WOPR:PRO{producer_num}'], 
            label='Oil Production', color=colors['WOPR'],
            linestyle=linestyles['WOPR'], linewidth=linewidths['WOPR'],
            marker='o', markersize=5, alpha=0.85
        )
        ax[col].plot(
            res.index, res[f'WWPR:PRO{producer_num}'], 
            label='Water Production', color=colors['WWPR'],
            linestyle=linestyles['WWPR'], linewidth=linewidths['WWPR'],
            marker='s', markersize=4, alpha=0.85
        )
        
        # Enhanced styling
        ax[col].set_title(f'Producer {producer_num}', fontsize=14, fontweight='bold', pad=15)
        ax[col].set_ylabel('Rate (m³/day)', fontsize=11, fontweight='500')
        ax[col].legend(loc='best', framealpha=0.95, fontsize=10, edgecolor='gray')
        ax[col].grid(True, alpha=0.7, linewidth=0.8)
        ax[col].spines['top'].set_visible(False)
        ax[col].spines['right'].set_visible(False)
        
        # Rotate x-axis labels for better readability
        ax[col].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    fig.savefig('rates.png', dpi=300, bbox_inches='tight')


def plot_permx():
    """Plot the log permeability field as a 3D voxel visualization, including wells as small sticks above the grid."""
    nx, ny, nz = 10, 10, 2
    permx = np.load('PERMX.npy').reshape((nx, ny, nz), order='F')

    wells = {
        "INJ1": {"ij": (0, 0), "color": "deepskyblue"},
        "INJ2": {"ij": (4, 0), "color": "deepskyblue"},
        "INJ3": {"ij": (9, 0), "color": "deepskyblue"},
        "PRO1": {"ij": (0, 9), "color": "crimson"},
        "PRO2": {"ij": (4, 9), "color": "crimson"},
        "PRO3": {"ij": (9, 9), "color": "crimson"},
    }

    permx_norm = (permx - permx.min()) / (permx.max() - permx.min())
    filled = np.ones((nx, ny, nz), dtype=bool)

    cmap = plt.get_cmap('YlGnBu_r')
    facecolors = cmap(permx_norm)
    edgecolor_intensity = permx_norm * 0.5 + 0.5
    edgecolors = plt.cm.gray(edgecolor_intensity)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.computed_zorder = False  # allow manual zorder in 3D
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.12)

    x, y, z = np.indices(np.array(filled.shape) + 1).astype(float)
    x = x / nx
    y = y / ny
    z = z / nz

    ax.voxels(
        x, y, z, filled,
        facecolors=facecolors,
        edgecolors=edgecolors,
        linewidth=0.5,
        alpha=1.0,
        zsort='max'
    )

    # Very thin, taller well sticks: centered in each cell
    stick_extra_above = 0.75      # taller above z=1.0 (was 0.30)
    stick_size_x = 0.15 / nx      # thinner
    stick_size_y = 0.15 / ny      # thinner

    for name, w in wells.items():
        i, j = w["ij"]
        color = w.get("color", "black")

        # exact cell center in normalized coordinates
        cx = (i + 0.5) / nx
        cy = (j + 0.5) / ny

        # bar3d expects lower-left corner, so shift by half size to keep centered
        ax.bar3d(
            cx - 0.5 * stick_size_x, cy - 0.5 * stick_size_y, 1.0,
            stick_size_x, stick_size_y, 1.0 + stick_extra_above,
            color=color, edgecolor=None, linewidth=0.8, shade=True, alpha=0.7, zsort='max',
        )
        ax.text(cx, cy, 1.0 + stick_extra_above + 1.2, name, color=color, fontsize=9, ha='center', zorder=1)

    ax.set_zlim(0.0, 1.0 + stick_extra_above + 0.05)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=permx.min(), vmax=permx.max()))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.15, 0.2, 0.7, 0.03])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Permeability (mD)', fontsize=11, fontweight='bold')

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.view_init(elev=20, azim=45)
    ax.set_box_aspect([nx/10, ny/10, nz/10])
    fig.savefig('permx.png', dpi=300, bbox_inches='tight')


def plot_gradient(grad, savename='gradient_log_permx.png'):
    nx, ny, nz = 10, 10, 2
    grad = grad.reshape((nx, ny, nz), order='F')

    wells = {
        "INJ1": {"ij": (0, 0), "color": "deepskyblue"},
        "INJ2": {"ij": (4, 0), "color": "deepskyblue"},
        "INJ3": {"ij": (9, 0), "color": "deepskyblue"},
        "PRO1": {"ij": (0, 9), "color": "crimson"},
        "PRO2": {"ij": (4, 9), "color": "crimson"},
        "PRO3": {"ij": (9, 9), "color": "crimson"},
    }
    
    # Normalize gradient for color mapping (0-1)
    grad_norm = (grad - grad.min()) / (grad.max() - grad.min())
    
    # Create voxel array (all filled)
    filled = np.ones((nx, ny, nz), dtype=bool)
    
    # Create color map: low gradient (dark blue) to high gradient (bright yellow)
    cmap = plt.get_cmap('YlGnBu_r')
    rgba_colors = cmap(grad_norm)
    facecolors = rgba_colors
    
    # Edge colors based on intensity
    edgecolor_intensity = grad_norm * 0.5 + 0.5  # Scale to [0.5, 1]
    edgecolors = plt.cm.gray(edgecolor_intensity)
    
    # Create figure with 3D axes and space for colorbar
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.computed_zorder = False  # allow manual zorder in 3D
    
    # Minimize whitespace by adjusting margins
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.12)
    
    # Create coordinate arrays
    x, y, z = np.indices(np.array(filled.shape) + 1).astype(float)
    x = x / nx
    y = y / ny
    z = z / nz
    
    # Plot voxels
    ax.voxels(
        x, y, z, filled,
        facecolors=facecolors,
        edgecolors=edgecolors,
        linewidth=0.5,
        alpha=1.0,
        zsort='max'
    )

    # Very thin, taller well sticks: centered in each cell
    stick_extra_above = 0.75      # taller above z=1.0 (was 0.30)
    stick_size_x = 0.15 / nx      # thinner
    stick_size_y = 0.15 / ny      # thinner

    for name, w in wells.items():
        i, j = w["ij"]
        color = w.get("color", "black")

        # exact cell center in normalized coordinates
        cx = (i + 0.5) / nx
        cy = (j + 0.5) / ny

        # bar3d expects lower-left corner, so shift by half size to keep centered
        ax.bar3d(
            cx - 0.5 * stick_size_x, cy - 0.5 * stick_size_y, 1.0,
            stick_size_x, stick_size_y, 1.0 + stick_extra_above,
            color=color, edgecolor=None, linewidth=0.8, shade=True, alpha=0.7, zsort='max',
        )
        ax.text(cx, cy, 1.0 + stick_extra_above + 1.2, name, color=color, fontsize=9, ha='center', zorder=1)

    ax.set_zlim(0.0, 1.0 + stick_extra_above + 0.05)
    
    
    # Add horizontal colorbar below the plot
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=grad.min(), vmax=grad.max()))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.15, 0.2, 0.7, 0.03])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Gradient of log-PERMX (Sm3/day)', fontsize=11, fontweight='bold')
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # View angle
    ax.view_init(elev=20, azim=45)
    ax.set_box_aspect([nx/10, ny/10, nz/10])
    plt.show()
    fig.savefig(savename, dpi=300, bbox_inches='tight')


if __name__ == "__main__":

    # Visualize permeability field
    plot_permx()
    
    # Run simulation
    results, gradient = simulate()
    print(results)
    print(gradient)
    
    # Plot gradient field for report time step
    grad = gradient.loc[datetime(2032, 12, 14), ('WOPR:PRO2', 'log_permx')] 
    plot_gradient(grad, savename='gradient_log_permx.png')

    # Plot production rates
    plot(results)

    # Test that gradient with respect to log_permx is consistent with gradient with respect to permx
    permx = np.load('PERMX.npy')
    grad_permx = gradient.loc[datetime(2032, 12, 14), ('WOPR:PRO2', 'permx')]
    grad_log_permx = gradient.loc[datetime(2032, 12, 14), ('WOPR:PRO2', 'log_permx')]
    np.testing.assert_almost_equal(grad_log_permx, grad_permx * permx)
