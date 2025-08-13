# Necessary packages
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch


def tsne_visual(
    real_data,
    generated_data,
    class_label_data=None,
    save_root=None,
    min_viz_samples=1000,
):
    """TSNE visualization of generated time series and real time series.

    Args:
        real_data (ArrayLike): Real time series data, in shape of `[B, T, C]`.
        generated_data (ArrayLike): Generated time series data, in shape of `[B, T, C]`.
        class_label_data (ArrayLike, optional): Time series labels. If not `None`, in shape of `[B, ]` Defaults to None.
        save_root (str, optional): Save root path. The post fix should be `.png`/`.pdf`/etc. If `None`, don't save figure. Defaults to None.
        min_viz_samples (int, optional): The number of data samples put into visualization. Defaults to 1000.
    """

    # Analysis sample size (for faster computation)
    anal_sample_no = min([min_viz_samples, len(real_data)])
    idx = np.random.permutation(len(real_data))[:anal_sample_no]

    # Data preprocessing
    real_data = np.asarray(real_data)
    generated_data = np.asarray(generated_data)

    real_data = real_data[idx]
    generated_data = generated_data[idx]
    if class_label_data is not None:
        class_label_data = class_label_data[idx]
        unique_class = torch.unique(class_label_data)
        cls_idx = [class_label_data == i for i in unique_class]

    prep_data = real_data.reshape(len(real_data), -1)
    prep_data_hat = generated_data.reshape(len(generated_data), -1)

    # Do t-SNE Analysis together
    prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

    # TSNE anlaysis
    tsne = TSNE(n_components=2, max_iter=300, random_state=0, init="random")
    tsne_results = tsne.fit_transform(prep_data_final)
    real_tsne = tsne_results[:anal_sample_no]
    gen_tsne = tsne_results[anal_sample_no:]

    # Plotting
    f, ax = plt.subplots(1)

    if class_label_data is None:
        ax.scatter(
            real_tsne[:, 0],
            real_tsne[:, 1],
            c="C0",
            alpha=0.5,
            label="Real",
        )
        ax.scatter(
            gen_tsne[:, 0],
            gen_tsne[:, 1],
            c="C1",
            alpha=0.5,
            label="Syn",
        )
    else:
        for i in range(len(cls_idx)):
            ax.scatter(
                real_tsne[cls_idx[i], 0],
                real_tsne[cls_idx[i], 1],
                alpha=0.5,
                label=f"Real: class {i}",
            )
        for i in range(len(cls_idx)):
            ax.scatter(
                gen_tsne[cls_idx[i], 0],
                gen_tsne[cls_idx[i], 1],
                # c=colors[anal_sample_no:],
                alpha=0.5,
                label=f"Syn: class {i}",
            )

    ax.legend(ncols=2)
    ax.set_title("t-SNE plot")
    ax.set_xlabel("x_tsne")
    ax.set_ylabel("y_tsne")
    if save_root is not None:
        f.savefig(save_root)


def imputation_visual(
    real_data: torch.Tensor,
    gen_data: torch.Tensor,
    cond_data: torch.Tensor,
    data_mask: torch.BoolTensor,
    max_viz_n_channel=3,
    save_root=None,
):
    """Visualize time series imputation results, including 95% predict interval.

    Args:
        real_data (torch.Tensor): Ground truth time series, in shape of `[B, seq_len, C]`.
        gen_data (torch.Tensor): Predicted time series scenarios, in shape of `[B, seq_len, C, N]`, `N` is the number of scenarios.
        cond_data (torch.Tensor): Observed time series, in shape of `[B, seq_len, C]`, missing values should be set as `NaN`.
        data_mask (torch.BoolTensor): Ground truth time series data mask, in shape of `[B, seq_len, C]`.
        max_viz_n_channel (int, optional): The maximum number of channels to be visualized. Defaults to 3.
        save_root (str, optional): Save root path. The post fix should be `.png`/`.pdf`/etc. If `None`, don't save figure. Defaults to None.
    """
    real_data = real_data.detach().cpu()
    gen_data = gen_data.detach().cpu()
    cond_mask = ~torch.isnan(cond_data)
    data_mask = data_mask.detach().cpu()
    target_mask = data_mask.float() - cond_mask.float()
    target_mask = target_mask.bool()
    for sample_axis in range(gen_data.shape[-1]):
        gen_data[..., sample_axis][~target_mask] = real_data[~target_mask]

    q = torch.tensor([0.05, 0.95])
    gen_data_quantiles = torch.quantile(gen_data, q, dim=-1)
    n_channel = real_data.shape[-1]
    n_channel = min(max_viz_n_channel, n_channel)
    fig, axs = plt.subplots(1, n_channel, figsize=[12, 4])
    if n_channel > 1:
        axs = axs.flatten()
    t = range(real_data.shape[1])
    for i in range(n_channel):
        plt_ax = axs[i] if n_channel > 1 else axs
        sample_id = torch.randint(0, len(real_data), ())
        real_data_plt = real_data.masked_fill(~cond_mask, torch.nan)
        target_data_plt = real_data.masked_fill(~target_mask, torch.nan)

        obs_line = plt_ax.scatter(
            t, real_data_plt[sample_id, :, i], c="C0", marker="o", label="observed"
        )
        target_line = plt_ax.scatter(
            t, target_data_plt[sample_id, :, i], c="red", marker="x", label="target"
        )
        impute_interval = plt_ax.fill_between(
            t,
            gen_data_quantiles[0, sample_id, :, i],
            gen_data_quantiles[-1, sample_id, :, i],
            label="95%PI",
            alpha=0.25,
        )
        [impute_line] = plt_ax.plot(
            t, torch.mean(gen_data, dim=-1)[sample_id, :, i], label="avg_imputation"
        )
        # axs[i].legend()
        plt_ax.set_xlabel("time")
        plt_ax.set_ylabel("value")
        plt_ax.set_title(f"Channel {i + 1}")
    fig.legend(
        handles=[obs_line, target_line, impute_interval, impute_line],
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.1),
    )
    fig.tight_layout()
    if save_root is not None:
        fig.savefig(save_root, bbox_inches="tight")
    # fig.savefig(save_root, bbox_inches="tight")


def predict_visual(
    real_data: torch.Tensor,
    gen_data: torch.Tensor,
    data_mask: torch.BoolTensor,
    max_viz_n_channel: int = 3,
    save_root: str = None,
):
    """Visualize time series prediction results, including 95% predict interval.

    Args:
        real_data (torch.Tensor): Ground truth time series, in shape of `[B, obs_len + seq_len, C]`.
        gen_data (torch.Tensor): Predicted time series scenarios, in shape of `[B, seq_len, C, N]`, `N` is the number of scenarios.
        data_mask (torch.BoolTensor): Ground truth time series data mask, in shape of `[B, obs_len + seq_len, C]`.
        max_viz_n_channel (int, optional): The maximum number of channels to be visualized. Defaults to 3.
        save_root (str, optional): Save root path. The post fix should be `.png`/`.pdf`/etc. If `None`, don't save figure. Defaults to None.
    """
    real_data = real_data.detach().cpu()
    gen_data = gen_data.detach().cpu()
    data_mask = data_mask.detach().cpu()

    q = torch.tensor([0.05, 0.95])
    gen_data_quantiles = torch.quantile(gen_data, q, dim=-1)
    n_channel = real_data.shape[-1]
    n_channel = min(max_viz_n_channel, n_channel)
    fig, axs = plt.subplots(1, n_channel, figsize=[12, 4], layout="constrained")
    if n_channel > 1:
        axs = axs.flatten()
    t = range(real_data.shape[1])
    pred_t = t[gen_data_quantiles.shape[2] :]
    for i in range(n_channel):
        plt_ax = axs[i] if n_channel > 1 else axs
        sample_id = torch.randint(0, len(real_data), ())

        [obs_line] = plt_ax.plot(
            t, real_data[sample_id, :, i], c="C0", label="ground truth"
        )

        pred_interval = plt_ax.fill_between(
            pred_t,
            gen_data_quantiles[0, sample_id, :, i],
            gen_data_quantiles[-1, sample_id, :, i],
            label="95%PI",
            color="C1",
            alpha=0.25,
        )
        [pred_line] = plt_ax.plot(
            pred_t,
            torch.mean(gen_data, dim=-1)[sample_id, :, i],
            label="avg_pred",
            c="C1",
        )
        # axs[i].legend()
        plt_ax.set_xlabel("time")
        plt_ax.set_ylabel("value")
        plt_ax.set_title(f"Channel {i + 1}")
    fig.legend(
        handles=[obs_line, pred_interval, pred_line],
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.1),
    )
    if save_root is not None:
        fig.savefig(save_root, bbox_inches="tight")
