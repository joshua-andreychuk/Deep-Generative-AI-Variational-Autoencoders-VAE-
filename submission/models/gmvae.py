import numpy as np
import torch
import os
script_directory = os.path.dirname(os.path.abspath(__file__))

if 'solution' in script_directory:
    from solution import utils as ut
    from solution.models import nns
else:
    from submission import utils as ut
    from submission.models import nns

from torch import nn

class GMVAE(nn.Module):
    def __init__(self, nn='v1', z_dim=2, k=500, name='gmvae'):
        super().__init__()
        self.name = name
        self.k = k
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim) # from ./nns/v1.py
        self.dec = nn.Decoder(self.z_dim) # from ./nns/v1.py

        # Mixture of Gaussians prior
        self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k, self.z_dim)
                                        / np.sqrt(self.k * self.z_dim))
        # Uniform weighting
        self.pi = torch.nn.Parameter(torch.ones(k) / k, requires_grad=False)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # To help you start, we have computed the mixture of Gaussians prior
        # prior = (m_mixture, v_mixture) for you, where
        # m_mixture and v_mixture each have shape (1, self.k, self.z_dim)
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be tensor scalars
        #
        # Return:
        #   nelbo, kl, rec
        ################################################################################
        # We provide the learnable prior for you. Familiarize yourself with
        # this object by checking its shape.
        prior = ut.gaussian_parameters(self.z_pre, dim=1)
        ### START CODE HERE ###
        # unpack the learnable mixture prior (1, k, z_dim)
        prior_m, prior_v = prior

        # encode to get q(z|x)
        qm, qv = self.enc(x)

        # sample a single z ~ q(z|x)
        z = ut.sample_gaussian(qm, qv)

        # reconstruction term
        log_px_z = ut.log_bernoulli_with_logits(x, self.dec(z))
        rec = -log_px_z.mean()

        # Monte Carlo KL: E_q[ log q(z|x) - log p(z) ]
        log_qz = ut.log_normal(z, qm, qv)
        B = x.size(0)
        log_pz = ut.log_normal_mixture(
            z,
            prior_m.expand(B, -1, -1),
            prior_v.expand(B, -1, -1)
        )
        kl = (log_qz - log_pz).mean()

        # negative ELBO
        nelbo = kl + rec
        return nelbo, kl, rec
        ### END CODE HERE ###
        ################################################################################
        # End of code modification
        ################################################################################
        raise NotImplementedError

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be tensor scalars
        #
        # Return:
        #   niwae, kl, rec
        ################################################################################
        # We provide the learnable prior for you. Familiarize yourself with
        # this object by checking its shape.
        prior = ut.gaussian_parameters(self.z_pre, dim=1)
        ### START CODE HERE ###
        ### START CODE HERE ###
        B = x.size(0)

        # unpack mixture prior: each is (1, k, z_dim)
        prior_m, prior_v = prior

        # 1) encode q(z|x)
        qm, qv = self.enc(x)  # → (B, z_dim)

        # 2) duplicate for importance samples
        qm_rep = ut.duplicate(qm, iw)   # → (B*iw, z_dim)
        qv_rep = ut.duplicate(qv, iw)
        x_rep  = ut.duplicate(x, iw)    # → (B*iw, data_dim)

        # 3) sample z ~ q
        z_rep = ut.sample_gaussian(qm_rep, qv_rep)  # → (B*iw, z_dim)

        # 4) compute log-density terms
        log_px_z = ut.log_bernoulli_with_logits(x_rep, self.dec(z_rep))  # → (B*iw,)
        log_qz_x = ut.log_normal(z_rep, qm_rep, qv_rep)                  # → (B*iw,)

        # 5) compute log p(z) under mixture prior
        m_exp = prior_m.expand(B*iw, -1, -1)  # → (B*iw, k, z_dim)
        v_exp = prior_v.expand(B*iw, -1, -1)
        log_pz = ut.log_normal_mixture(z_rep, m_exp, v_exp)              # → (B*iw,)

        # 6) reshape into (B, iw)
        log_w = (log_px_z + log_pz - log_qz_x) \
                  .view(iw, B) \
                  .transpose(0, 1)  # now (B, iw)

        # 7) Negative IWAE bound
        log_mean_w = ut.log_mean_exp(log_w, dim=1)  # → (B,)
        niwae = -log_mean_w.mean()

        # 8) compute normalized weights for rec/kl
        log_sum_w = ut.log_sum_exp(log_w, dim=1)               # → (B,)
        w_norm    = torch.exp(log_w - log_sum_w.unsqueeze(1))  # → (B, iw)

        # 9) reshape per-sample matrices
        lp_mat = log_px_z.view(iw, B).transpose(0, 1)  # → (B, iw)
        lq_mat = log_qz_x.view(iw, B).transpose(0, 1)
        pp_mat = log_pz.view(iw, B).transpose(0, 1)

        # 10) reconstruction & KL terms
        rec = (-(w_norm * lp_mat).sum(1)).mean()
        kl  = ((w_norm * (lq_mat - pp_mat)).sum(1)).mean()

        return niwae, kl, rec
        ### END CODE HERE ###
        ### END CODE HERE ###
        ################################################################################
        # End of code modification
        ################################################################################
        raise NotImplementedError

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        m, v = ut.gaussian_parameters(self.z_pre.squeeze(0), dim=0)
        idx = torch.distributions.categorical.Categorical(self.pi).sample((batch,))
        m, v = m[idx], v[idx]
        return ut.sample_gaussian(m, v)

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
