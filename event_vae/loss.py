import torch
import torch.nn as nn


class LossModule:
    def __init__(self, rec_loss, ae_type="vanilla", lamda=0):

        # Choice of reconstruction losses
        rec_loss_dict = {
            "mse": nn.MSELoss,
            "huber": nn.SmoothL1Loss,
            "bce": nn.BCELoss,
            "l1": nn.L1Loss,
        }

        self.criterion = rec_loss_dict[rec_loss]()
        self.ae_type = ae_type
        self.lamda = lamda

        self.loss1 = None
        self.loss2 = None

        self.scale_factor = 1.0

    def kld_loss(self, mu, log_var):
        batch_size = mu.size(0)
        KLD_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        return KLD / batch_size

    def kl_loss_scale(self, epoch, scaling="none"):
        """
        Implements an annealing strategy that slowly increases the weighting of KL Loss term
        to avoid collapse. Current strategy does not include KL loss for the first 1000 epochs
        and then linearly increases it till epoch 10000 to reach final weight, prescribed by
        lamda.
        """

        if scaling == "anneal":
            if epoch > 1000:
                self.scale_factor = (epoch / 10000) * self.lamda
            else:
                self.scale_factor = 0

        elif scaling == "beta":
            self.scale_factor = self.lamda

    def contractive_loss(self, z, x):
        """
        Implements contractive loss (Frobenius norm loss), which is computed as the
        weighted L2-norm of the Jacobian of the hidden units with respect to the inputs

        Requires the input to retain gradients prior to calling loss function as
        
        events.retain_grad()
        events.requires_grad_(True)
        """

        z.backward(torch.ones(z.size()).cuda(), retain_graph=True)
        frob_loss = torch.sqrt(torch.sum(torch.pow(x.grad, 2)))
        x.grad.data.zero_()

        return frob_loss

    def reconstruction_loss(self, recon, gt):
        return self.criterion(recon, gt)

    def compute_loss(self, epoch, x, gt, recon, z, logvar=None, scaling=None):
        """
        Stack different losses as required: VAE combines recon + KLD loss
        """

        loss = None

        # Vanilla autoencoder
        if self.ae_type == "vanilla":
            loss = self.reconstruction_loss(recon, gt)

        # Contractive autoencoder: Reconstruction loss + Frobenius norm loss
        elif self.ae_type == "contractive":
            self.loss1 = self.reconstruction_loss(recon, gt)
            self.loss2 = self.contractive_loss(z, x)

            loss = self.loss1 + (self.lamda * self.loss2)

        # Variational autoencoder: Reconstruction loss + KL divergence loss
        elif self.ae_type == "variational":
            self.loss1 = self.reconstruction_loss(recon, gt)
            self.loss2 = self.kld_loss(z, logvar)

            self.kl_loss_scale(epoch, scaling)

            loss = self.loss1 + self.scale_factor * self.loss2

        return loss
