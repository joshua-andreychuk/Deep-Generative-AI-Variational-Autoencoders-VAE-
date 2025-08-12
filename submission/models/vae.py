import torch
from torch import nn
import os
script_directory = os.path.dirname(os.path.abspath(__file__))

if 'solution' in script_directory:
    from solution import utils as ut
    from solution.models import nns
else:
    from submission import utils as ut
    from submission.models import nns

class VAE(nn.Module):
    def __init__(self, nn='v1', name='vae', z_dim=2):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim) # from ./nns/v1.py
        self.dec = nn.Decoder(self.z_dim) # from ./nns/v1.py

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

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
        # Note that nelbo = kl + rec
        #
        # Outputs should all be tensor scalars
        #
        # Return:
        #   nelbo, kl, rec
        ################################################################################
        ### START CODE HERE ###
        ### START CODE HERE ###
        # q(z|x) ← Encoder(x)
        qm, qv = self.enc(x)

        # sample z ∼ q(z|x)
        z = ut.sample_gaussian(qm, qv)

        # rec and KL as before …
        log_px_z = ut.log_bernoulli_with_logits(x, self.dec(z))
        rec = -log_px_z.mean()

        pm = self.z_prior_m.expand_as(qm)
        pv = self.z_prior_v.expand_as(qv)
        kl = ut.kl_normal(qm, qv, pm, pv).mean()

        nelbo = kl + rec
        return nelbo, kl, rec
        ### END CODE HERE ###
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
        #
        # HINT: The summation over m may seem to prevent us from 
        # splitting the ELBO into the KL and reconstruction terms, but instead consider 
        # calculating log_normal w.r.t prior and q
        ################################################################################
        ### START CODE HERE ###
        ### START CODE HERE ###
        B = x.size(0)

        # 1) encode q(z|x)
        qm, qv = self.enc(x)                             # (B, z_dim)

        # 2) duplicate for IW samples
        qm_rep = ut.duplicate(qm, iw)                    # (B*iw, z_dim)
        qv_rep = ut.duplicate(qv, iw)
        x_rep  = ut.duplicate(x, iw)                     # (B*iw, data_dim)

        # 3) sample z’s
        z_rep = ut.sample_gaussian(qm_rep, qv_rep)       # (B*iw, z_dim)

        # 4) compute log‐terms
        log_px_z = ut.log_bernoulli_with_logits(x_rep, self.dec(z_rep))  # (B*iw,)
        prior_m  = self.z_prior_m.expand(B*iw, self.z_dim)
        prior_v  = self.z_prior_v.expand(B*iw, self.z_dim)
        log_pz   = ut.log_normal(z_rep, prior_m, prior_v)              # (B*iw,)
        log_qz_x = ut.log_normal(z_rep, qm_rep, qv_rep)                # (B*iw,)

        # 5) reshape to (B, iw) **correctly**
        log_w = (log_px_z + log_pz - log_qz_x) \
                  .view(iw, B) \
                  .transpose(0, 1)                                  # (B, iw)

        # 6) Negative IWAE bound
        log_mean_w = ut.log_mean_exp(log_w, dim=1)                   # (B,)
        niwae = - log_mean_w.mean()

        # 7) ELBO‐decomposition via normalized weights
        log_sum_w = ut.log_sum_exp(log_w, dim=1)                     # (B,)
        w_norm    = torch.exp(log_w - log_sum_w.unsqueeze(1))        # (B, iw)

        # 8) reconstruct per-sample matrices and compute rec/kl
        lp_mat = log_px_z.view(iw, B).transpose(0, 1)                 # (B, iw)
        lq_mat = log_qz_x.view(iw, B).transpose(0, 1)
        pp_mat = log_pz.view(iw, B).transpose(0, 1)

        rec = (- (w_norm * lp_mat).sum(1)).mean()
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
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
