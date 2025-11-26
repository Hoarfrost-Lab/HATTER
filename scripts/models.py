import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import math

from dal_toolbox.models.mc_dropout import MCDropoutModel
from dal_toolbox.models.utils.mcdropout import ConsistentMCDropout2d, MCDropoutModule
from dal_toolbox.models.deterministic.simplenet import SimpleNet as Net
from dal_toolbox.models.mc_dropout.simplenet import SimpleMCNet as MCNet

#---------------------------------------------------------------------------------------#
class CLEANLayerNormNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        super(CLEANLayerNormNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.drop_out = drop_out
        self.device = device
        self.dtype = dtype

        self.fc1 = nn.Linear(input_dim, hidden_dim, dtype=dtype, device=device)
        self.ln1 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim,
                             dtype=dtype, device=device)
        self.ln2 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.dropout(self.ln1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.ln2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        return x

    def get_logits(self, dataloader, device=None, return_labels=False, loss='triplet'):
        if device == None:
            device = self.device

        if loss == 'triplet':
            return self.get_logits_triplet(dataloader, device, return_labels=return_labels)
        #elif loss == 'supconh':
        #    return get_logits_triplet(self, dataloader, device, return_labels=return_labels)
        #elif loss == 'himulcone':
        #    return get_logits_triplet(self, dataloader, device, return_labels=return_labels)

    def get_logits_triplet(self, dataloader, device, return_labels=False):
        self.to(device)
        self.eval()
        
        all_anchor_logits = []
        all_pos_logits = []
        all_neg_logits = []
        
        for anchor, pos, neg in dataloader:
            anchor_logits = self(anchor.to(device))

            if return_labels:
                pos_logits = self(pos.to(device))
                neg_logits = self(neg.to(device))

            all_anchor_logits.append(anchor_logits)
            
            if return_labels:
                all_pos_logits.append(pos_logits)
                all_neg_logits.append(neg_logits)

        anchor_ret_logits = torch.cat(all_anchor_logits)
        
        if return_labels:
            pos_ret_logits = torch.cat(all_pos_logits)
            neg_ret_logits = torch.cat(all_neg_logits)

            return anchor_ret_logits, pos_ret_logits, neg_ret_logits

        return anchor_ret_logits

    @torch.inference_mode()
    def get_representations(self, dataloader, device, return_labels=False, loss='triplet'):
        if loss == 'triplet':
            return self.get_representations_triplet(dataloader, device, return_labels=return_labels)
        #elif loss == 'supconh':
        #    return get_representations_triplet(self, dataloader, device, return_labels=return_labels)
        #elif loss == 'himulcone':
        #    return get_representations_triplet(self, dataloader, device, return_labels=return_labels)

    @torch.inference_mode()
    def get_representations_triplet(self, dataloader, device, return_labels=False):
        all_anchors = []
        all_pos = []
        all_negs = []
        
        for batch in dataloader:
            anchor = batch[0]
            if return_labels:
                pos = batch[1]
                neg = batch[2]

            all_anchors.append(anchor.cpu())
            
            if return_labels:
                all_pos.append(pos.cpu())
                all_negs.append(neg.cpu())
        
        anchors = torch.cat(all_anchors)

        if return_labels:
            positives = torch.cat(all_pos)
            negatives = torch.cat(all_negs)
            return anchors, positives, negatives
        
        return anchors

    #for triplet need to modify later #FIXME
    @torch.inference_mode()
    def get_grad_representations(self, dataloader, device):
        self.eval()
        self.to(device)

        embedding = []
        for batch in dataloader:
            inputs = batch[0] #anchor
            embedding_batch = torch.empty([len(inputs), self.input_dim * self.out_dim])
            logits = self(inputs.to(device)).cpu()
            features = inputs.cpu()

            probas = logits.softmax(-1)
            max_indices = probas.argmax(-1)

            for n in range(len(inputs)):
                for c in range(self.out_dim):
                    if c == max_indices[n]:
                        embedding_batch[n, self.input_dim * c: self.input_dim * (c + 1)] = \
                            features[n] * (1 - probas[n, c])
                    else:
                        embedding_batch[n, self.input_dim * c: self.input_dim * (c + 1)] = \
                            features[n] * (-1 * probas[n, c])
            embedding.append(embedding_batch)
        # Concat all embeddings
        embedding = torch.cat(embedding)
        return embedding

    def get_logits_bayesian(self, dataloader, device=None, return_labels=False, loss='triplet'):
        if device == None:
            device = self.device

        if loss == 'triplet':
            return self.get_logits_triplet_bayesian(dataloader, device, return_labels=return_labels)
        #elif loss == 'supconh':
        #    return get_logits_triplet(self, dataloader, device, return_labels=return_labels)
        #elif loss == 'himulcone':
        #    return get_logits_triplet(self, dataloader, device, return_labels=return_labels)

    def get_logits_triplet_bayesian(self, dataloader, device, return_labels=False):
        self.to(device)
        self.eval()
        
        all_anchor_logits = []
        all_pos_logits = []
        all_neg_logits = []
        
        for anchor, pos, neg in dataloader:
            anchor_logits = self.mc_forward(anchor.to(device))

            if return_labels:
                pos_logits = self.mc_forward(pos.to(device))
                neg_logits = self.mc_forward(neg.to(device))

            all_anchor_logits.append(anchor_logits)
            
            if return_labels:
                all_pos_logits.append(pos_logits)
                all_neg_logits.append(neg_logits)

        anchor_ret_logits = torch.cat(all_anchor_logits)
        
        if return_labels:
            pos_ret_logits = torch.cat(all_pos_logits)
            neg_ret_logits = torch.cat(all_neg_logits)

            return anchor_ret_logits, pos_ret_logits, neg_ret_logits

        return anchor_ret_logits


#---------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------#
class CLEANBatchNormNet(CLEANLayerNormNet):
    def __init__(self, input_dim, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        super(CLEANBatchNormNet, self).__init__(input_dim, hidden_dim, out_dim, device, dtype, drop_out=drop_out)
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.drop_out = drop_out
        self.device = device
        self.dtype = dtype

        self.fc1 = nn.Linear(input_dim, hidden_dim, dtype=dtype, device=device)
        self.bn1 = nn.BatchNorm1d(hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim,
                             dtype=dtype, device=device)
        self.bn2 = nn.BatchNorm1d(hidden_dim, dtype=dtype, device=device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.dropout(self.bn1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.bn2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        return x
#---------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------#
class CLEANInstanceNormNet(CLEANLayerNormNet):
    def __init__(self, input_dim, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        super(CLEANInstanceNormNet, self).__init__(input_dim, hidden_dim, out_dim, device, dtype, drop_out=drop_out)
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.drop_out = drop_out
        self.device = device
        self.dtype = dtype

        self.fc1 = nn.Linear(input_dim, hidden_dim, dtype=dtype, device=device)
        self.in1 = nn.InstanceNorm1d(hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim,dtype=dtype, device=device)
        self.in2 = nn.InstanceNorm1d(hidden_dim, dtype=dtype, device=device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.dropout(self.in1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.in2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        return x
#---------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------#
#code credit: https://github.com/adityagoel4512/Spectral-normalized-Neural-Gaussian-Process-PyTorch/tree/master

from collections import namedtuple
NetResult = namedtuple('NetResult', ('mean', 'variance'))

class SNGPBackbone(CLEANLayerNormNet):
  def __init__(self,
               input_dim: int = 768,
               hidden_dim_layers: int = 5,
               hidden_dim: int = 512,
               out_dim: int = 128,
               drop_out: float = 0.1,
               device: str = 'cpu',
               dtype = torch.float32,
               norm_multiplier: float = 0.9):

    super(SNGPBackbone, self).__init__(input_dim, hidden_dim, out_dim, device, dtype, drop_out=drop_out)
    self.hidden_dim = hidden_dim
    self.drop_out = drop_out
    self.input_layer = nn.Linear(in_features=input_dim, out_features=hidden_dim)
    self.hidden_layers = nn.Sequential(*[nn.Linear(in_features=hidden_dim, out_features=hidden_dim) for _ in range(hidden_dim_layers)])

    if norm_multiplier is not None:
      self.norm_multiplier = norm_multiplier
      self.input_layer.register_full_backward_hook(self.spectral_norm_hook)
      for hidden_layer in self.hidden_layers:
        hidden_layer.register_full_backward_hook(self.spectral_norm_hook)

  def forward(self, _input):
    _input = self.input_layer(_input)
    for hidden_layer in self.hidden_layers:
      residual = _input
      _input = F.dropout(F.relu(hidden_layer(_input)), p=self.drop_out, training=self.training)
      _input += residual

    return _input

  def spectral_norm_hook(self, module, grad_input, grad_output):
    # applied to linear layer weights after gradient descent updates
    with torch.no_grad():
      norm = torch.linalg.matrix_norm(module.weight, 2)
      if self.norm_multiplier < norm:
        module.weight = nn.Parameter(self.norm_multiplier * module.weight / norm)

class CLEANSNGPNet(CLEANLayerNormNet):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            out_dim: int = 128, 
            device = 'cpu',
            dtype = torch.float32,
            drop_out=0.001,
            num_inducing: int = 1024,
            momentum: float = 0.98,
            ridge_penalty: float = 1e-6
    ):
        super(CLEANSNGPNet, self).__init__(input_dim, hidden_dim, out_dim, device, dtype, drop_out=drop_out)

        self.out_dim = out_dim
        self.num_inducing = num_inducing
        self.momentum = momentum
        self.ridge_penalty = ridge_penalty

        backbone = SNGPBackbone(input_dim=input_dim, hidden_dim=hidden_dim, out_dim=out_dim, device=device, dtype=dtype, drop_out=drop_out)
        
        # Random Fourier features (RFF) layer
        random_fourier_feature_layer = nn.Linear(backbone.hidden_dim, num_inducing)
        random_fourier_feature_layer.weight.requires_grad_(False)
        random_fourier_feature_layer.bias.requires_grad_(False)
        nn.init.normal_(random_fourier_feature_layer.weight, mean=0.0, std=1.0)
        nn.init.uniform_(random_fourier_feature_layer.bias, a=0.0, b=2 * math.pi)

        self.rff = nn.Sequential(backbone, random_fourier_feature_layer)

        # RFF approximation reduces the GP to a standard Bayesian linear model,
        # with beta being the parameters we wish to estimate by maximising
        # p(beta | D). To this end p(beta) (the prior) is gaussian so the loss
        # can be written as a standard MAP objective
        self.beta = nn.Linear(num_inducing, out_dim, bias=False)
        nn.init.normal_(self.beta.weight, mean=0.0, std=1.0)

        # RFF precision and covariance matrices
        self.is_fit = torch.tensor(False)
        self.dataset_passed = torch.tensor(False)

        self.covariance = Parameter(
            self.ridge_penalty * torch.eye(num_inducing),
            requires_grad=False,
        )

        self.precision_initial = self.ridge_penalty * torch.eye(
            num_inducing, requires_grad=False
        )
        self.precision = Parameter(
            self.precision_initial,
            requires_grad=False,
        )

    def forward(self, X, with_variance=False, update_precision=False):
        features = torch.cos(self.rff(X))

        if update_precision:
            self.update_precision_(features)

        logits = self.beta(features)

        if not with_variance:
            return logits
        else:
            if not self.is_fit:
                raise ValueError(
                    "`update_covariance` should be called before setting "
                    "`with_variance` to True"
                )
            with torch.no_grad():
                variances = torch.bmm(features[:, None, :], (features @ self.covariance)[:, :, None], ).reshape(-1)
                if not self.dataset_passed:
                    self.max_variance = torch.max(variances).unsqueeze(dim=0)
                    self.dataset_passed = torch.tensor(True)
                variances = variances / self.max_variance

            return NetResult(logits, variances)

    def reset_precision(self):
        self.precision = self.precision_initial.detach()

    def update_precision_(self, features):
        # This assumes that all classes share a precision matrix like in
        # https://www.tensorflow.org/tutorials/understanding/sngp

        # The original SNGP paper defines precision and covariance matrices on a
        # per class basis, however this can get expensive to compute with large
        # output spaces
        with torch.no_grad():
            if self.momentum < 0:
                # self.precision = identity => self.precision = identity + features.T @ features
                self.precision = Parameter(self.precision + features.T @ features)
            else:
                self.precision = Parameter(self.momentum * self.precision +
                                           (1 - self.momentum) * features.T @ features)

    def update_precision(self, X):
        with torch.no_grad():
            features = torch.cos(self.rff(X))
            self.update_precision_(features)

    def update_covariance(self):
        if not self.is_fit:
            # The precision matrix is positive definite and so we can use its cholesky decomposition to more
            # efficiently compute its inverse (when num_inducing is large)
            try:
                L = torch.linalg.cholesky(self.precision)
                self.covariance = Parameter(self.ridge_penalty * L.cholesky_inverse(), requires_grad=False)
                self.is_fit = torch.tensor(True)
            except:
                self.covariance = Parameter(self.ridge_penalty * self.precision.cholesky_inverse(), requires_grad=False)
                self.is_fit = torch.tensor(True)

    def reset_covariance(self):
        self.is_fit = torch.tensor(False)
        self.covariance.zero_()
#---------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------#
class CLEANConvNet(CLEANLayerNormNet):
    def __init__(self, input_dim, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        super(CLEANConvNet, self).__init__(input_dim, hidden_dim, out_dim, device, dtype, drop_out=drop_out)

        self.input_dim = input_dim
        self.relu = nn.ReLU()
        self.layer1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, padding='valid')
        self.pooling1 = nn.MaxPool1d(kernel_size=3)
        self.layer2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding='valid')
        self.pooling2 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

        n_size = self._get_conv_output((1, input_dim))

        self.layer3 = nn.Linear(n_size, hidden_dim)
        self.output = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(p=drop_out)

    # generate input sample and forward to get shape
    def _get_conv_output(self, shape):
        bs = 1
        _input = torch.autograd.Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(_input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.pooling1(self.relu(self.layer1(x)))
        x = self.pooling2(self.relu(self.layer2(x)))
        x = self.flatten(x)
        return x

    def forward(self, x):
        x = x.reshape(-1, self.input_dim, 1)
        x = x.permute(0, 2, 1)

        out = self._forward_features(x)
        out = self.relu(out)
        out = self.dropout(self.relu(self.layer3(out)))
        out = self.output(out)

        return out
#---------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------#
class CLEANStandardNet(CLEANLayerNormNet):
    def __init__(self, input_dim, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        super(CLEANStandardNet, self).__init__(input_dim, hidden_dim, out_dim, device, dtype, drop_out=drop_out)
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_out)
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = self.dropout(self.relu(out))
        out = self.layer2(out)
        return out
#---------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------#
#This is fixing the MC Dropout from DAL Toolbox for our needs -- not a model
class MyConsistentMCDropout2d(ConsistentMCDropout2d):
    def __init__(self, p=0.2):
        super().__init__()

    def _create_mask(self, input, k):
        if k is None:
            k = input.shape[0]
        mask_shape = [1, k] + list(self._get_sample_mask_shape(input.shape[1:]))
        mask = torch.empty(mask_shape, dtype=torch.bool, device=input.device).bernoulli_(self.p)
        return mask

    def forward(self, input: torch.Tensor):
        if self.p == 0.0:
            return input

        if self.training:
            # Create a new mask on each call and for each batch element.
            mask = self._create_mask(input, input.shape[0])
        else:
            if self.mask is None:
                # print('recreating mask', self)
                # Recreate mask.
                self.mask = self._create_mask(input, input.shape[0])

            mask = self.mask

        k = input.shape[0]
        if mask.shape[1] != k:
            mask = mask.narrow(1,0,k)

        mc_input = MCDropoutModule(n_passes=50).unflatten_tensor(input, k)
        mc_output = mc_input.masked_fill(mask.to(mc_input.device), 0) / (1 - self.p)

        # Flatten MCDI, batch into one dimension again.
        return MCDropoutModule(n_passes=50).flatten_tensor(mc_output)

#---------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------#
#overloads of dropout with monte carlo dropout for bayesian (see DAL-Toolbox docs)
class CLEANLayerNormNetMC(CLEANLayerNormNet):
    def __init__(self, input_dim, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        super(CLEANLayerNormNetMC, self).__init__(input_dim, hidden_dim, out_dim, device, dtype, drop_out=drop_out)
        self.dropout = MyConsistentMCDropout2d(p=drop_out)

    def mc_forward(self, input_B: torch.Tensor):
        mc_input_BK = MCDropoutModule.mc_tensor(input_B, MCDropoutModule.n_passes)
        mc_output_BK = self.mc_forward_impl(mc_input_BK)
        mc_output_B_K = MCDropoutModule.unflatten_tensor(mc_output_BK, MCDropoutModule.n_passes)
        return mc_output_B_K  # N x M x K

    def mc_forward_impl(self, *args, **kwargs):
        return self(*args, **kwargs)

class CLEANBatchNormNetMC(CLEANBatchNormNet):
    def __init__(self, input_dim, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        super(CLEANBatchNormNetMC, self).__init__(input_dim, hidden_dim, out_dim, device, dtype, drop_out=drop_out)
        self.dropout = MyConsistentMCDropout2d(p=drop_out)

    def mc_forward(self, input_B: torch.Tensor):
        mc_input_BK = MCDropoutModule.mc_tensor(input_B, MCDropoutModule.n_passes)
        mc_output_BK = self.mc_forward_impl(mc_input_BK)
        mc_output_B_K = MCDropoutModule.unflatten_tensor(mc_output_BK, MCDropoutModule.n_passes)
        return mc_output_B_K  # N x M x K

    def mc_forward_impl(self, *args, **kwargs):
        return self(*args, **kwargs)

class CLEANInstanceNormNetMC(CLEANInstanceNormNet):
    def __init__(self, input_dim, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        super(CLEANInstanceNormNetMC, self).__init__(input_dim, hidden_dim, out_dim, device, dtype, drop_out=drop_out)
        self.dropout = MyConsistentMCDropout2d(p=drop_out)

    def mc_forward(self, input_B: torch.Tensor):
        mc_input_BK = MCDropoutModule.mc_tensor(input_B, MCDropoutModule.n_passes)
        mc_output_BK = self.mc_forward_impl(mc_input_BK)
        mc_output_B_K = MCDropoutModule.unflatten_tensor(mc_output_BK, MCDropoutModule.n_passes)
        return mc_output_B_K  # N x M x K

    def mc_forward_impl(self, *args, **kwargs):
        return self(*args, **kwargs)

class CLEANSNGPNetMC(CLEANSNGPNet):
    def __init__(self, input_dim, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        super(CLEANSNGPNetMC, self).__init__(input_dim, hidden_dim, out_dim, device, dtype, drop_out=drop_out)
        self.dropout = MyConsistentMCDropout2d(p=drop_out)

    def mc_forward(self, input_B: torch.Tensor):
        mc_input_BK = MCDropoutModule.mc_tensor(input_B, MCDropoutModule.n_passes)
        mc_output_BK = self.mc_forward_impl(mc_input_BK)
        mc_output_B_K = MCDropoutModule.unflatten_tensor(mc_output_BK, MCDropoutModule.n_passes)
        return mc_output_B_K  # N x M x K

    def mc_forward_impl(self, *args, **kwargs):
        return self(*args, **kwargs)

class CLEANConvNetMC(CLEANConvNet):
    def __init__(self, input_dim, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        super(CLEANConvNetMC, self).__init__(input_dim, hidden_dim, out_dim, device, dtype, drop_out=drop_out)
        self.dropout = MyConsistentMCDropout2d(p=drop_out)

    def mc_forward(self, input_B: torch.Tensor):
        mc_input_BK = MCDropoutModule.mc_tensor(input_B, MCDropoutModule.n_passes)
        mc_output_BK = self.mc_forward_impl(mc_input_BK)
        mc_output_B_K = MCDropoutModule.unflatten_tensor(mc_output_BK, MCDropoutModule.n_passes)
        return mc_output_B_K  # N x M x K

    def mc_forward_impl(self, *args, **kwargs):
        return self(*args, **kwargs)

class CLEANStandardNetMC(CLEANStandardNet):
    def __init__(self, input_dim, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        super(CLEANStandardNetMC, self).__init__(input_dim, hidden_dim, out_dim, device, dtype, drop_out=drop_out)
        self.dropout = MyConsistentMCDropout2d(p=drop_out)
        
    def mc_forward(self, input_B: torch.Tensor):
        mc_input_BK = MCDropoutModule.mc_tensor(input_B, MCDropoutModule.n_passes)
        mc_output_BK = self.mc_forward_impl(mc_input_BK)
        mc_output_B_K = MCDropoutModule.unflatten_tensor(mc_output_BK, MCDropoutModule.n_passes)
        return mc_output_B_K  # N x M x K

    def mc_forward_impl(self, *args, **kwargs):
        return self(*args, **kwargs)

#---------------------------------------------------------------------------------------#

#ADDING ORIGINAL CODE - NO CLEAN
#---------------------------------------------------------------------------------------#
class TwoLayerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TwoLayerClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out

class TwoLayerMCClassifier(MCNet):
    def __init__(self,
                 num_classes: int,
                 dropout_rate: int = .2,
                 feature_dim: int = 128,
                 in_dimension: int = 2
                 ):

        super().__init__(num_classes=num_classes)

        self.first = torch.nn.Linear(in_dimension, feature_dim)
        self.first_dropout = ConsistentMCDropout(dropout_rate)
        self.hidden = torch.nn.Linear(feature_dim, feature_dim)
        self.hidden_dropout = ConsistentMCDropout(dropout_rate)
        self.last = torch.nn.Linear(feature_dim, num_classes)
        self.act = torch.nn.ReLU()

#---------------------------------------------------------------------------------------#


class BertClassifier(torch.nn.Module):
    def __init__(self, tokenizer, encoder, input_size, hidden_size, num_classes, dropout_rate=0.5, pretrained_weights=None, mode='static', bayesian=False):
        super(BertClassifier, self).__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.mode = mode
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.num_classes = num_classes
        self.bayesian = bayesian

        if self.mode == 'static':
            for param in self.encoder.parameters():
                param.required_grad = False


        if bayesian:
            model = TwoLayerMCClassifier(input_dim=input_size, hidden_dim=hidden_size, num_classes=self.num_classes, dropout_rate=dropout_rate)
        else:
            model = TwoLayerClassifier(input_dim=input_size, hidden_dim=hidden_size, num_classes=self.num_classes)


        if pretrained_weights is not None:
            model.load_state_dict(torch.load(pretrained_weights))

        self.head = model

    def forward(self, input_ids, attention_mask, return_cls=False):
        output = self.encoder(
            input_ids,
            attention_mask,
            return_dict=False
        )

        last_hidden_state = output[0]
        out_pooled = last_hidden_state[:, 0]
        out_logits = self.head(out_pooled)

        if return_cls:
            return (out_logits, out_pooled)
        else:
            return out_logits

    @torch.no_grad()
    def forward_logits(self, dataloader, device):
        self.to(device)
        all_logits = []

        for samples, _ in dataloader:
            logits = self(samples.to(device))
            all_logits.append(logits)

        return torch.cat(all_logits)


    @torch.inference_mode()
    def get_probas(self, dataloader, device):
        self.to(device)
        self.eval()
        all_logits = []

        for input_ids, attention_mask, labels, indexes in tqdm(dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = self(input_ids, attention_mask)
            all_logits.append(logits.to("cpu"))

        logits = torch.cat(all_logits)
        probas = logits.softmax(-1)

        return probas

    @torch.inference_mode()
    def get_logits(self, dataloader, device):
        self.to(device)
        self.eval()
        all_logits = []

        for input_ids, attention_mask, labels, indexes in tqdm(dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = self(input_ids, attention_mask)
            all_logits.append(logits)

        return torch.cat(all_logits)

    @torch.inference_mode()
    def get_representations(self, dataloader, device):
        self.to(device)
        self.eval()
        all_features = []

        for input_ids, attention_mask, labels, indexes in tqdm(dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            _, cls_state = self(input_ids, attention_mask, return_cls=True)
            all_features.append(cls_state.to("cpu"))

        features = torch.cat(all_features)
        return features

    @torch.inference_mode()
    def get_representations_and_probas(self, dataloader, device):
        self.to(device)
        self.eval()
        all_features = []
        all_logits = []

        for input_ids, attention_mask, labels in tqdm(dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits, cls_state = self(input_ids, attention_mask, return_cls=True)
            all_features.append(cls_state.to("cpu"))
            all_logits.append(logits.to('cpu'))

        logits = torch.cat(all_logits)
        probas = logits.softmax(-1)
        features = torch.cat(all_features)

        return features, probas

    @torch.inference_mode()
    def get_grad_representations(self, dataloader, device, feature_dim=768):
        self.eval()
        self.to(device)

        embedding = []
        for input_ids, attention_mask, labels, indexes in tqdm(dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            embedding_batch = torch.empty([len(input_ids), feature_dim * self.num_classes])

            logits, cls_state = self(input_ids, attention_mask, return_cls=True)
            logits = logits.cpu()
            features = cls_state.cpu()

            probas = logits.softmax(-1)
            max_indices = probas.argmax(-1)

            # TODO: optimize code
            # for each sample in a batch and for each class, compute the gradient wrt to weights
            for n in range(len(input_ids)):
                for c in range(self.num_classes):
                    if c == max_indices[n]:
                        embedding_batch[n, feature_dim * c: feature_dim * (c+1)] = features[n] * (1 - probas[n, c])
                    else:
                        embedding_batch[n, feature_dim * c: feature_dim * (c+1)] = features[n] * (-1 * probas[n, c])
            embedding.append(embedding_batch)

        # Concat all embeddings
        embedding = torch.cat(embedding)
        return embedding
