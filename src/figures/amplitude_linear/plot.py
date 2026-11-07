import marimo
__generated_with="0.23.16"
app=marimo.App(width="full")

@app.cell
def _():
    from pathlib import Path
    import os
    HERE = Path(__file__).resolve().parent
    os.environ.setdefault('MPLBACKEND','Agg')
    os.environ.setdefault('MPLCONFIGDIR',str(HERE/'mpl_config'))
    import json
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import seaborn as sns
    import marimo as mo
    device = torch.device('cuda')
    sns.set_theme(style='whitegrid',font='cmr10',font_scale=1.9)
    plt.rcParams['axes.formatter.use_mathtext'] = True
    plt.rcParams.update({'mathtext.fontset':'cm','pdf.fonttype':42,'ps.fonttype':42,'pdf.use14corefonts':False})
    colors = sns.color_palette()
    return HERE, json, mo, np, os, plt, sns


@app.cell
def _(mo):
    mo.md("""# Figure 11
    All alpha runs use CPU initialization followed by CUDA training.
    The blue dashed line is the alpha=1 spectral approximation, using projections of the actual initial CPU weights onto Gamma modes and the configured learning rate.
    """)
    return


@app.cell
def _(HERE, json, np):
    config = json.loads((HERE/'linear/config.json').read_text(encoding='utf-8'))
    output_dict = np.load(HERE/'linear/data.npy',allow_pickle=True).item()
    with np.load(HERE/'linear/source_setup_diagnostics.npz',allow_pickle=False) as _setup:
        gamma = _setup['gamma'].astype(np.float64)
    s_out = len(config['frequency'])
    return config, gamma, output_dict, s_out


@app.cell
def _(config, gamma, np, output_dict):
    # The scalar solution assumes the Gamma modes evolve independently (alpha=1).
    # For random W0, initialize that approximation from each Gamma-mode projection,
    # rather than pairing full-space singular values of W0 with Gamma eigenvalues.
    _reference = output_dict['BarlowTwins']['1']
    _w0 = np.asarray(_reference['w'], dtype=np.float64)
    _gamma_values, _gamma_vectors = np.linalg.eigh((gamma + gamma.T) / 2)
    _order = np.argsort(_gamma_values)[::-1][:len(config['frequency'])]
    _gamma_values = _gamma_values[_order]
    _gamma_vectors = _gamma_vectors[:, _order]
    _mode_s0 = np.linalg.norm(_w0 @ _gamma_vectors, axis=0)
    _lambda0 = _gamma_values * _mode_s0**2
    _steps = np.arange(int(config['BarlowTwins']['epochs']) + 1, dtype=float)
    _eta = float(config['BarlowTwins']['lr'])
    _predicted_lambda = 1 / (
        1 + (1 / _lambda0[:, None] - 1)
        * np.exp(-8 * _eta * _gamma_values[:, None] * _steps[None, :])
    )
    theory = {
        'reference_alpha': '1', 'method': 'independent Gamma-mode spectral approximation',
        'steps': _steps, 'gamma': _gamma_values, 'mode_s0': _mode_s0,
        'lambda0': _lambda0, 'lr': _eta, 'eigenvalues': _predicted_lambda,
        'loss': ((_predicted_lambda - 1)**2).sum(axis=0),
        'feature_alignment': np.sqrt(_predicted_lambda),
        'old_full_space_s0': np.asarray(_reference['s0'], dtype=np.float64),
        'time_or_amplitude_fit': False,
    }
    return theory,


@app.cell
def _(np):
    def dt(data):
        output = []
        for i in range(1, len(data)):
            output.append(data[i] - data[i - 1])
        return output

    def smooth_data(data, window_size=5):
        data_array = np.array(data)
        smooth_data = np.copy(data_array)
        half_window = window_size // 2
        for i in range(len(data_array)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(data_array), i + half_window + 1)
            smooth_data[i] = np.mean(data_array[start_idx:end_idx])
        return smooth_data
    return dt, smooth_data


@app.cell
def _(config, dt, output_dict, plt, s_out, smooth_data, sns, theory, np):
    loss_types = "BarlowTwins"
    fig, axs = plt.subplots(1, 3, figsize=(24, 6.5))
    epochs = config[loss_types]["epochs"]
    color_continuas = sns.color_palette("rocket", n_colors=len(config[loss_types]["lambdas"]))
    for index, lambd in enumerate(config[loss_types]["lambdas"]):
        output_target = output_dict[loss_types][lambd]
        losses = output_target["loss"]
        eiglist = output_target["eiglist"]
        feature_alignment = output_target["feature_alignment"]

        dldt = smooth_data(dt(losses), 7)
        # plt.plot(dldt)
        # plt.show()


        axs[0].plot(losses, color=color_continuas[index], linewidth=6, alpha=0.7, label=(r"$\alpha = " + f"{lambd}" + "$"))
        axs[0].legend(ncols=2)
        critical_pos = []
        starting = 0
        for i in range(len(eiglist)):
            find = np.argmin(dldt[starting:])
            if dldt[starting + find] < -0.002:
                critical_pos.append(starting + find)
                starting += find + 100
            else:
                break

        axs[0].set_ylim(-0.05, s_out * 1.15)
        # for pos in critical_pos:
        #     axs[0].axvline(pos, color='black', linestyle='--')
        axs[0].set_xlabel("Step")
        axs[0].set_ylabel("Loss")

        for i in range(len(eiglist)):
            axs[1].plot(eiglist[i], linewidth=6, color=color_continuas[index], alpha=0.7,
                        label=(r"$\alpha_" + ("l$" if i == 0 else "s$")), linestyle='--' if i == 1 else '-')
        # for pos in critical_pos:
        #     axs[1].axvline(pos, color='black', linestyle='--')
        # axs[1].set_ylim(-0.05, 1.10)
        axs[1].set_xlabel("Step")
        axs[1].set_ylabel(r"C Eigenvalue $\lambda$")

        if config["network"] != "mlp":
            for i in range(len(feature_alignment)):
                k = feature_alignment[i]
                axs[2].plot(k, label=(r"${\text{FA}(e_l)}$" if i == 0 else r"${\text{FA}(e_s)}$"), alpha=0.7,
                            linewidth=6, color=color_continuas[index], linestyle='--' if i == 1 else '-')
            # for pos in critical_pos:
            #     axs[2].axvline(pos, color='black', linestyle='--')
            # axs[2].set_ylim(-0.05, 1.10)
            # axs[2].legend()
            axs[2].set_xlabel("Step")
            axs[2].set_ylabel(r"$\|We_i\|_2$")
        print(critical_pos)

    # A single alpha=1 theoretical reference; all empirical alpha curves stay intact.
    axs[0].plot(theory['steps'][:-1], theory['loss'][:-1], linestyle='--', color='blue', linewidth=2.5)
    for _mode in range(len(theory['gamma'])):
        axs[1].plot(theory['steps'], theory['eigenvalues'][_mode], linestyle='--', color='blue', linewidth=2.5)
        axs[2].plot(theory['steps'], theory['feature_alignment'][_mode], linestyle='--', color='blue', linewidth=2.5)

    plt.tight_layout()
    fig
    return fig,


@app.cell
def _(mo):
    save_pdf = mo.ui.run_button(label='PDF 저장')
    save_pdf
    return save_pdf,


@app.cell
def _(HERE, fig, os, save_pdf):
    if save_pdf.value or os.environ.get('FIG11_EXPORT')=='1':
        _path = HERE/'loss_lambda_all.pdf'
        _path.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(_path,format='pdf',bbox_inches='tight',pad_inches=.1)
        fig.savefig(_path.with_suffix('.png'),bbox_inches='tight',pad_inches=.1)
        fig.savefig(_path.with_suffix('.svg'),bbox_inches='tight',pad_inches=.1)
    return

if __name__=="__main__":
    app.run()
