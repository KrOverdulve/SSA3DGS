# SSA-3DGS trainer: 3D Gaussian Splatting with a jointly optimized screen-space
# overlay (color map W + alpha matte A) for unsupervised artifact removal.
#
# Derived from the gsplat example trainer (nerfstudio-project/gsplat,
# Apache-2.0). The overlay model, its statistics-based initialization, the
# sparsity regularizer, and the FLIP metric are the SSA-3DGS additions.
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import torchvision

import flip_evaluator as flip
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from nerfview import CameraState, RenderTabState, apply_float_colormap
from pytorch_msssim import ssim
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never

from config.config import Config
from config.default_config import DefaultConfig
from config.mcmc_config import MCMCConfig
from datasets.colmap import Dataset, Parser, colmap_collate_fn
from gsplat import export_splats
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy

from gsplat_viewer import GsplatRenderTabState, GsplatViewer
from utils import knn, rgb_to_sh, set_random_seed


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    batch_size: int = 1,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    # color is SH coefficients.
    colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
    colors[:, 0, :] = rgb_to_sh(rgbs)
    params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
    params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)

    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizers = {
        name: torch.optim.Adam(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)
        self.watermark_dir = f"{cfg.result_dir}/watermark"
        os.makedirs(self.watermark_dir, exist_ok=True)

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            cached=True,
        )
        self.valset = Dataset(
            self.parser,
            split="val",
        )
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            batch_size=cfg.batch_size,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        if cfg.remove_watermark:
            # Overlay maps are stored in logit space; sigmoid is applied at every
            # use site so the effective color and alpha are bounded to [0, 1].
            ref_img = self.trainset[0]["image"]
            H, W, C = ref_img.shape

            if cfg.watermark_init == "constant":
                # Ablation baseline for the statistics-based initialization: no
                # training-set scan. Zero logits give a uniform gray color
                # (sigmoid(0) = 0.5) and 0.5 opacity everywhere; the optimizer
                # and the sparsity prior shape the overlay from scratch.
                color_logit = torch.zeros((H, W, C), device=self.device)
                alpha_logit = torch.zeros((H, W, 1), device=self.device)
                print("Watermark init: constant (zero logits, uniform gray / 0.5 alpha).")
            else:
                # Seed the overlay color to the per-pixel mean of the training
                # set, and the alpha to exp(-std / std_scale): low-variance
                # pixels are likely screen-space artifacts and start opaque,
                # high-variance pixels are likely scene content and start nearly
                # transparent so the 3D scene can explain them.
                pix_sum = torch.zeros((H, W, C), dtype=torch.float64, device=self.device)
                pix_sum_sq = torch.zeros((H, W, C), dtype=torch.float64, device=self.device)
                n_imgs = len(self.trainset)
                for i in tqdm.trange(n_imgs, desc="watermark init: scanning images"):
                    img = self.trainset[i]["image"].to(self.device).double() / 255.0
                    pix_sum += img
                    pix_sum_sq += img * img
                pix_mean = (pix_sum / n_imgs).float()
                pix_var = ((pix_sum_sq / n_imgs).float() - pix_mean * pix_mean).clamp_(min=0.0)
                # Aggregate across channels: a pixel is "stable" only if every
                # channel is stable, so take the max std over channels.
                pix_std = pix_var.sqrt().max(dim=-1, keepdim=True).values  # [H, W, 1]

                eps = 1e-4
                mean_clamped = pix_mean.clamp(eps, 1.0 - eps)
                color_logit = torch.log(mean_clamped / (1.0 - mean_clamped))

                # std_scale tunes the variance-to-alpha mapping. With 0.05, a pixel
                # whose intensity wobbles by ~0.05 across views gets alpha ~= 0.37;
                # near-static pixels (std << 0.05) start essentially opaque, pixels
                # with clear multi-view variation (std >> 0.05) start near-transparent.
                std_scale = 0.05
                alpha_target = torch.exp(-pix_std / std_scale).clamp(eps, 1.0 - eps)
                alpha_logit = torch.log(alpha_target / (1.0 - alpha_target))
                print(
                    f"Watermark init: statistics (mean alpha={alpha_target.mean().item():.3f}, "
                    f"std range [{pix_std.min().item():.3f}, {pix_std.max().item():.3f}])."
                )

            self.watermark = color_logit.clone().contiguous()
            self.watermark.requires_grad_(True)
            self.watermark_alpha = alpha_logit.clone().contiguous()
            self.watermark_alpha.requires_grad_(True)
            self.watermark_optimizer = torch.optim.Adam(
                [self.watermark, self.watermark_alpha], lr=cfg.watermark_lr
            )

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

    def rasterize_splats(
        self,
        splats: Dict[str, torch.Tensor],
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = splats["means"]  # [N, 3]
        # rasterization does quaternion normalization internally
        quats = splats["quats"]  # [N, 4]
        scales = torch.exp(splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(splats["opacities"])  # [N,]
        colors = torch.cat([splats["sh0"], splats["shN"]], 1)  # [N, K, 3]

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"

        camera_model = kwargs.pop("camera_model", self.cfg.camera_model)

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=camera_model,
            **kwargs,
        )
        return render_colors, render_alphas, info

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]

        trainloader_generator = torch.Generator()
        trainloader_generator.manual_seed(42 + self.local_rank)
        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            persistent_workers=False,
            pin_memory=True,
            collate_fn=colmap_collate_fn,
            generator=trainloader_generator,
        )
        trainloader_iter = iter(trainloader)

        # Sanity check parameters
        print("################################################################################")
        print("Config:")
        print(cfg)
        print("################################################################################")

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            gt = data["image"].to(device) / 255.0  # [1, H, W, 3]

            bkgd = (
                torch.rand(1, 3, device=device)
                if cfg.random_bkgd
                else torch.zeros(1, 3, device=device)
            )

            num_train_rays_per_step = gt.shape[0] * gt.shape[1] * gt.shape[2]
            height, width = gt.shape[1:3]

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            renders, alphas, info = self.rasterize_splats(
                splats=self.splats,
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB",
                backgrounds=bkgd,
            )
            colors = renders

            if cfg.remove_watermark and step >= cfg.watermark_from_iter:
                # Composite the shared screen-space overlay over the 3D render:
                # C_hat = A * W + (1 - A) * C_3D   (Eq. 2 in the paper)
                cl_wm = torch.sigmoid(self.watermark)
                cl_wm_alpha = torch.sigmoid(self.watermark_alpha)
                colors = cl_wm_alpha * cl_wm + (1.0 - cl_wm_alpha) * colors

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # loss
            l1loss = F.l1_loss(colors, gt)
            ssimloss = 1.0 - ssim(
                colors.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

            if cfg.remove_watermark and step >= cfg.watermark_from_iter:
                cl_wm = torch.sigmoid(self.watermark)
                cl_wm_alpha = torch.sigmoid(self.watermark_alpha)
                # Sparsity is on the alpha (support) only, so W is free to
                # take whatever color the data wants without being pushed dark.
                # TV still uses the overlay-over-black M = A*W.
                m = cl_wm_alpha * cl_wm
                watermark_l1_penalty = torch.norm(cl_wm_alpha, 1) / (
                    self.watermark.shape[0] * self.watermark.shape[1]
                )
                # Spatial L1 total variation on the masked overlay [H, W, 3]
                watermark_tv_penalty = (m[1:, :, :] - m[:-1, :, :]).abs().mean() + (
                    m[:, 1:, :] - m[:, :-1, :]
                ).abs().mean()
                watermark_penalty = (
                    cfg.watermark_l1_reg * watermark_l1_penalty
                    + cfg.watermark_tv_reg * watermark_tv_penalty
                )
                loss += watermark_penalty

            # regularizations (used by the MCMC configuration)
            if cfg.opacity_reg > 0.0:
                loss = (
                    loss
                    + cfg.opacity_reg
                    * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
                )

            if cfg.scale_reg > 0.0:
                loss = (
                    loss
                    + cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()
                )

            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.remove_watermark and step >= cfg.watermark_from_iter:
                desc += f"watermark_loss loss={watermark_penalty.item():.6f}| "
            pbar.set_description(desc)

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
                if cfg.remove_watermark:
                    with torch.no_grad():
                        wm = torch.sigmoid(self.watermark)
                        wm_a = torch.sigmoid(self.watermark_alpha)
                    self._save_watermark_pair(
                        wm, wm_a,
                        f"{self.watermark_dir}/watermark_step{step}_rank{self.world_rank}",
                    )

            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:
                export_splats(
                    means=self.splats["means"],
                    scales=self.splats["scales"],
                    quats=self.splats["quats"],
                    opacities=self.splats["opacities"],
                    sh0=self.splats["sh0"],
                    shN=self.splats["shN"],
                    format="ply",
                    save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                )

            # optimize
            self.make_optimization_step(cfg, schedulers)

            # Run post-backward steps after backward and optimizer
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # eval the full validation set
            if (step + 1) in cfg.eval_steps:
                print("Evaluating ... ", step)
                self.eval(step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    def _save_watermark_pair(self, wm: Tensor, wm_alpha: Tensor, path_prefix: str) -> None:
        """Save the overlay composed over BOTH black and white backgrounds.

        Black-bg shows the additive contribution M = A*W (matches the
        regularized quantity). White-bg is for visual inspection of dark
        / low-contrast artifacts like mud.
        """
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        for bg_name, bg_value in (("black", 0.0), ("white", 1.0)):
            composed = wm_alpha * wm + (1.0 - wm_alpha) * bg_value
            arr = (composed.clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
            imageio.imwrite(f"{path_prefix}_bg{bg_name}.png", arr)

    def make_optimization_step(self, cfg, schedulers):
        for optimizer in self.optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if cfg.remove_watermark:
            self.watermark_optimizer.step()
            self.watermark_optimizer.zero_grad(set_to_none=True)
        for scheduler in schedulers:
            scheduler.step()

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Evaluate on the held-out clean test views.

        Metrics are computed on the clean render C_3D (WITHOUT the overlay), so
        they measure the restoration quality of the underlying 3D scene.
        """
        if len(self.valset) == 0:
            print("No held-out clean_ views in this dataset; skipping evaluation.")
            return
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=colmap_collate_fn
        )
        ellipse_time = 0
        # Accumulate metrics over ALL validation views; the mean below is then a
        # true per-scene average.
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()

            renders, _, _ = self.rasterize_splats(
                splats=self.splats,
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            colors = renders[..., 0:3]
            depths = renders[..., 3:].repeat(1, 1, 1, 3)
            colors = torch.clamp(colors, 0.0, 1.0)
            depths = depths / depths.max()
            # Render strip layout: [ GT | render | depth | (overlay-on-white) ].
            canvas_list = [pixels, colors, depths]
            if cfg.remove_watermark:
                white_bg = torch.ones_like(self.watermark, dtype=torch.float32, device=device)
                cl_wm = torch.sigmoid(self.watermark)
                cl_wm_alpha = torch.sigmoid(self.watermark_alpha)
                white_bg_watermark = cl_wm_alpha * cl_wm + (1.0 - cl_wm_alpha) * white_bg
                canvas_list = canvas_list + [white_bg_watermark.unsqueeze(0)]

            if world_rank == 0:
                # write images
                canvas = torch.cat(canvas_list, dim=2).squeeze(0)
                torchvision.utils.save_image(
                    canvas.permute(2, 0, 1),
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                )

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))

                # FLIP (NVIDIA perceptual metric). Operates on HxWx3 float32
                # numpy arrays in [0, 1]; we squeeze the batch dim.
                gt_np = pixels[0].detach().cpu().numpy().astype(np.float32)
                rd_np = colors[0].detach().cpu().numpy().astype(np.float32)
                _, mean_flip, _ = flip.evaluate(gt_np, rd_np, "LDR")
                metrics["flip"].append(torch.tensor(float(mean_flip), device=device))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            print(
                f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, "
                f"LPIPS: {stats['lpips']:.3f}, FLIP: {stats['flip']:.4f} "
                f"Time: {stats['ellipse_time']:.3f}s/image "
                f"Number of GS: {stats['num_GS']}"
            )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = self.rasterize_splats(
            splats=self.splats,
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device) / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.eval(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training (ADC densification)
    CUDA_VISIBLE_DEVICES=0 python train.py default --data_dir <scene> --result_dir <out>

    # Single GPU training (MCMC densification)
    CUDA_VISIBLE_DEVICES=0 python train.py mcmc --strategy.cap-max 1000000 \
        --data_dir <scene> --result_dir <out>
    ```
    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            DefaultConfig(),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            MCMCConfig(),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    cli(main, cfg, verbose=True)
