# Deep-Generative-AI-Variational-Autoencoders-VAE-

---

## Overview

In this project I implemented and compared several members of the VAE family and trained them on image data:

* **VAE** (vanilla) on **MNIST**
* **GMVAE** (Mixture-of-Gaussians prior) on **MNIST**
* **IWAE** evaluation for tighter likelihood bounds
* **SSVAE** (Semi-Supervised VAE) classifier on MNIST
* **FSVAE** (class-conditional / “fully supervised” VAE) on **SVHN** to separate **content** (digit) from **style**

I’m focusing this repo on the code and the **real outputs** I generated.

---

## Results (files in the repo)

* `model=vae_z=10_run=0000.png` — MNIST samples from the **VAE**
* `model=gmvae_z=10_k=500_run=0000.png` — MNIST samples from the **GMVAE**
* `model=fsvae_run=0000.png` — **FSVAE** on SVHN: rows fix the digit **y** (content), columns vary latent **z** (style)

---

## What the numbers show & why

### 1) Likelihood bounds (VAE vs GMVAE)

After training I evaluated ELBO components and **IWAE-k** bounds (lower is better because these are **negative** log-likelihood bounds).

**VAE (MNIST)**

* NELBO ≈ **101.52**, KL ≈ **19.47**, Recon ≈ **82.04**
* Negative IWAE-k: **101.53** (k=1), **98.84** (k=10), **97.89** (k=100), **97.44** (k=1000)

**GMVAE (MNIST, z=10, k=500 mixture)**

* NELBO ≈ **99.03**, KL ≈ **17.89**, Recon ≈ **81.15**
* Negative IWAE-k: **98.99** (k=1), **96.75** (k=10), **95.85** (k=100), **95.43** (k=1000)

**Why this makes sense**

* **IWAE gets tighter with larger k.** As k↑ the bound drops (moves toward the true −log p(x)), which is exactly what I see.
* **GMVAE beats VAE by \~2 points** on IWAE-1000 (≈97.44 → ≈95.43). A mixture prior gives a more expressive latent space that captures multi-modal structure (different digit styles), so samples look a bit sharper and the bound improves.
* The **ELBO split** (Recon + KL) shows a reasonable trade-off: GMVAE achieves slightly better reconstruction with a slightly smaller KL than the plain VAE, consistent with a better-matched prior.

### 2) Semi-supervised classification (SSVAE)

I trained the SSVAE with different weights on the generative objective (gw) and classification objective (cw):

* **gw=0, cw=100** → test accuracy ≈ **75.7%**
  *Interpretation:* with **gw=0** the model is basically a small supervised classifier; it can’t leverage unlabeled data.
* **gw=0.001, cw=100** → test accuracy ≈ **93.9%**
  *Interpretation:* adding a **small generative weight** lets the model use unlabeled data via the ELBO, regularizing the classifier and smoothing decision boundaries → **big accuracy jump**.

### 3) Class-conditional generation (FSVAE on SVHN)

* In `model=fsvae_run=0000.png`, each **row** fixes the label **y** (digit class = content) while columns vary **z** (style).
* This shows **content/style disentanglement**: digits stay the same across a row, but color/lighting/stroke vary — exactly what a conditional VAE should learn.

---

## My takeaways

* **Expressive priors help.** GMVAE’s mixture prior improves the likelihood bound and sample quality vs. a standard Gaussian.
* **Better bounds matter.** IWAE confirms the trend across k and makes the VAE/GMVAE comparison clearer.
* **Unlabeled data is useful.** Even a tiny **gw** in SSVAE strongly boosts accuracy by leveraging the generative objective on unlabeled examples.
* **Conditioning disentangles.** FSVAE cleanly separates digit identity (y) from style (z), which you can see directly in the SVHN grid.

---

## How to navigate this repo

**Notebooks (Question 1–5)**
Each notebook contains the exact cells/commands I used per experiment:

* **Question 1 — VAE (train & eval)**
  `!python main.py --model vae --train --device gpu`
* **Question 2 — GMVAE (train), then eval**
  `!python main.py --model gmvae --train --device gpu`
  `!python main.py --model gmvae`
* **Question 3 — IWAE evaluation (VAE & GMVAE)**
  `!python main.py --model vae --iwae`
  `!python main.py --model gmvae --iwae`
* **Question 4 — SSVAE (semi-supervised classifier)**
  Baseline: `!python main.py --model ssvae --train --gw 0 --device gpu`
  Semi-supervised: `!python main.py --model ssvae --train --gw 0.001 --cw 100 --device gpu`
* **Question 5 — FSVAE (SVHN, class-conditional)**
  Train: `!python main.py --model fsvae --train --device gpu --iter_max 50000 --iter_save 10000`
  Render grid: `!python main.py --model fsvae --device gpu --iter_save 50000`
* `main.py` — entry point the notebooks call.

**`submission/` folder (code + artifacts)**

* `train.py` — entry point the notebooks call.
* `models/` — implementations: `vae.py`, `gmvae.py`, `ssvae.py`, `fsvae.py`

  * `models/nns/` — small MLP/CNN blocks used by encoders/decoders.
* `*_iwae_{k}.pkl` — saved IWAE evaluation results for different k (1, 10, 100, 1000).
* `{VAE,GMVAE,SSVAE}.pkl` — small run/config/summary files.
* Generated images:
  `model=vae_z=10_run=0000.png`, `model=gmvae_z=10_k=500_run=0000.png`, `model=fsvae_run=0000.png`.

---

## Note on data & weights

**Note:** The original training datasets and model weights are not included in this repository due to size and licensing constraints. All outputs in the repo were generated by models I trained.
