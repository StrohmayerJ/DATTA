import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, einsum

########################################################################
######################## WiFlexFormer with DAT  ########################
########################################################################

# WiFlexFormer paper: https://doi.org/10.48550/arXiv.2411.04224
# WiFlexFormer github: https://github.com/StrohmayerJ/WiFlexFormer
# DAT paper: https://dl.acm.org/doi/10.1145/3241539.3241548

class WiFlexFormerStem(nn.Module):
    def __init__(self,
                 input_dim: int,
                 feature_dim: int,
                 num_channels: int,
                 ):
        super(WiFlexFormerStem, self).__init__()

        # Single-channel stem (for amplitude features)
        self.stem1D = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
         )

        # Multi-channel stem (e.g., for DFS features)
        self.stem2D = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.GELU(),
            nn.Conv2d(in_channels=num_channels, out_channels=1, kernel_size=(1, 3), padding=(0, 1)),
            nn.GELU(),
        )

    def forward(self, x):
        if x.shape[1] == 1:  # single-channel input path
            x = einsum(x, 'b c f t -> b f t')
            x = self.stem1D(x)
            x = rearrange(x, 'b f t -> b t f')

        else:  # multi-channel input path
            x = self.stem2D(x)
            x = einsum(x, 'b c f t -> b f t')
            x = self.stem1D(x)
            x = rearrange(x, 'b f t -> b t f')

        return x

# Activity Recognizer
class ActivityRecognizer(nn.Module):
    def __init__(self,
                 feature_dim: int,
                 num_classes: int
                 ):
        super(ActivityRecognizer, self).__init__()

        self.activity_recognizer = nn.Sequential(
            nn.Linear(feature_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.activity_recognizer(x)

# Domain Discriminator
class DomainDiscriminator(nn.Module):
    def __init__(self,
                 feature_dim: int,
                 num_domains: int
                 ):
        super(DomainDiscriminator, self).__init__()

        self.domain_discriminator = nn.Sequential(
            nn.Linear(feature_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_domains)
        )

    def forward(self, x):
        return self.domain_discriminator(x)

# Gradient Reversal Layer
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

def GradientReversal(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)

# WiFlexFormer with Domain-Adversarial Training (DAT)
class WiFlexFormerDAT(nn.Module):
    def __init__(self, 
                 input_dim: int=30, # number of subcarriers
                 num_channels:int=1, # 1 channel for amplitude features
                 feature_dim: int=32, # feature dimension
                 num_heads: int=16, # number of attention heads
                 num_layers: int=4, # number of transformer encoder layers
                 dim_feedforward: int=64, # feedforward dimension
                 window_size: int=220, # number of wifi packets in feature window
                 K=10, # number of Gaussian kernels
                 num_classes: int=6, # number of classes
                 num_domains: int=7, # number of domains in training data
                 grl_lambda: float = 1.0 # GRL scaling factor
                 ):
        super(WiFlexFormerDAT, self).__init__()

        self.grl_lambda = grl_lambda

        # WiFlexFormer stem
        self.stem = WiFlexFormerStem(input_dim, feature_dim, num_channels)

        # Gaussian positional encoding
        self.pos_encoding = Gaussian_Position(feature_dim, window_size, K=K)

        # Class token embeddings
        self.class_token_embeddings = nn.Parameter(torch.randn(1, 1, feature_dim))

        # WiFlexFormer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=0.1, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        
        # Activity Recognizer and Domain Discriminator
        self.activity_recognizer = ActivityRecognizer(feature_dim, num_classes)
        self.domain_discriminator = DomainDiscriminator(feature_dim, num_domains)

    def forward(self, x):

        # Pass through WiFlexFormer stem
        x = self.stem(x)
  
        # Gaussian positional encoding
        x = self.pos_encoding(x) 
        x = rearrange(x, 'b t f -> t b f')   
        
        # Add class token to the input sequence
        class_tokens = self.class_token_embeddings.expand(-1, x.size(1), -1)
        x = torch.cat((class_tokens, x), dim=0)
        
        # Pass through WiFlexFormer encoder
        x = self.transformer_encoder(x)

        # Extract class token output
        cls_token_output = x[0, :, :]  # Shape: (batch_size, feature_dim)

        # Apply gradient reversal layer
        reversed_features = GradientReversal(cls_token_output, self.grl_lambda)

        # Pass through domain discriminator
        domain_logits = self.domain_discriminator(reversed_features)

        # Pass through activity recognizer
        activity_logits = self.activity_recognizer(cls_token_output)

        return {
            'activity_recognizer_logits': activity_logits,
            'domain_discriminator_logits': domain_logits,
            'cls_token_output': cls_token_output
        }

    # Domain-Adversarial Training (DAT) Loss
    def loss_dat(self, pred, target=None, s=0, c=0.2, a=0.0, b=0.3, e=0, M=1):
        
        # Compute and combine activity and domain losses
        loss_da_activity_supervised = self.loss_dat_supervised_activity(pred, target) 
        loss_da_activity_unsupervised = self.loss_dat_unsupervised_activity(pred) * a
        loss_da_domain = self.loss_dat_supervised_domain(pred, target) if target is not None else 0  
        loss_da = loss_da_activity_supervised + a * loss_da_activity_unsupervised + b * loss_da_domain  

        # Add constraints
        loss_total = (
            loss_da +
            s * self.loss_smoothing_constraint(pred['cls_token_output'], M) +
            c * self.loss_confidence_constraint(F.softmax(pred['activity_recognizer_logits'], dim=-1)) +
            e * self.loss_balance_constraint(F.softmax(pred['activity_recognizer_logits'], dim=-1), target['domain']) 
        )
        return loss_total


    def loss_dat_supervised_activity(self, pred, target):
        return nn.CrossEntropyLoss()(pred['activity_recognizer_logits'], target['activity'])
    
    def loss_dat_supervised_domain(self, pred, target):
        return nn.CrossEntropyLoss()(pred['domain_discriminator_logits'], target['domain'])
    
    def loss_dat_unsupervised_activity(self, pred):
        return self.loss_entropy(pred['activity_recognizer_logits']) 
    
    def loss_smoothing_constraint(self, z, M=1, eps=1):
        if M == 1:
            noise = torch.normal(mean=0, std=eps, size=z.shape, device=z.device)
            y_pred = F.softmax(self.activity_recognizer(z), dim=-1)
            y_pred_noisy = F.softmax(self.activity_recognizer(z + noise), dim=-1)
            return self.jensen_shannon_divergence(y_pred, y_pred_noisy)
        else:
            loss = 0
            for _ in range(M):
                noise = torch.normal(mean=0, std=eps, size=z.shape, device=z.device)
                y_pred = F.softmax(self.activity_recognizer(z), dim=-1)
                y_pred_noisy = F.softmax(self.activity_recognizer(z + noise), dim=-1)
                loss += self.jensen_shannon_divergence(y_pred, y_pred_noisy)
            return loss / M  # Mean over the M noisy samples
        
    
    def loss_confidence_constraint(self, logits):
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  
        self.loss_confidence = torch.mean(entropy)  
        return self.loss_confidence
    
    def loss_balance_constraint(self, y_pred, domain_labels):
        # TRAIN distribution [0.2123601012989135, 0.1470059635650682, 0.09170002450780165, 0.18290989298259946, 0.18246058328567927, 0.1835634343599379], alternatively, a uniform distribution can be used
        prior_distribution = torch.tensor([0.2123601012989135, 0.1470059635650682, 0.09170002450780165, 0.18290989298259946, 0.18246058328567927, 0.1835634343599379], device=y_pred.device)
        
        num_samples, num_classes = y_pred.shape
        q_auxiliary = torch.zeros_like(y_pred)
        
        for domain in torch.unique(domain_labels):
            domain_indices = (domain_labels == domain).nonzero(as_tuple=True)[0]
            y_pred_domain = y_pred[domain_indices] 
            
            if len(y_pred_domain) == 0:  
                continue
            
            class_sum = y_pred_domain.sum(dim=0)
            class_sum = class_sum + 1e-9
            
            for c in range(num_classes):
                q_auxiliary[domain_indices, c] = prior_distribution[c] * y_pred_domain[:, c] / class_sum[c]
            
            q_auxiliary[domain_indices] /= q_auxiliary[domain_indices].sum(dim=1, keepdim=True) 
        
        return self.jensen_shannon_divergence(y_pred, q_auxiliary)

    def loss_entropy(self, logits):
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return torch.mean(entropy)

    def jensen_shannon_divergence(self, p, q, epsilon=1e-6):
        m = 0.5 * (p + q)
        jsd = 0.5 * (torch.sum(p * (torch.log(p + epsilon) - torch.log(m + epsilon)), dim=-1) +
                    torch.sum(q * (torch.log(q + epsilon) - torch.log(m + epsilon)), dim=-1))
        return torch.mean(jsd)

# Gaussian Positional Encoding source code from: https://github.com/windofshadow/THAT/blob/main/TransCNN.py
class Gaussian_Position(nn.Module):
    def __init__(self, d_model, total_size, K=10):
        super(Gaussian_Position, self).__init__()
        self.embedding = nn.Parameter(torch.zeros([K, d_model], dtype=torch.float), requires_grad=True)
        nn.init.xavier_uniform_(self.embedding, gain=1)

        # Assume total_size corresponds to the sequence length in x
        self.total_size = total_size

        # Setup Gaussian distribution parameters
        positions = torch.arange(total_size).unsqueeze(1).repeat(1, K)
        self.register_buffer('positions', positions)
        
        s = 0.0
        interval = total_size / K
        mu = []
        for _ in range(K):
            mu.append(s)
            s += interval
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float).unsqueeze(0), requires_grad=True)
        self.sigma = nn.Parameter(torch.ones([1, K], dtype=torch.float) * 50.0, requires_grad=True)

    def forward(self, x):
        # Ensure input x has shape [batch_size, seq_length, d_model]
        batch_size, seq_length, d_model = x.shape

        # Check if total_size matches seq_length, if not adjust positions or error out
        assert self.total_size == seq_length, "total_size must match seq_length of input x"

        # Calculate Gaussian distribution values
        M = normal_pdf(self.positions, self.mu, self.sigma)  # Assuming this function is defined correctly
        
        # Positional encodings
        pos_enc = torch.matmul(M, self.embedding)  # [seq_length, d_model]

        # Expand pos_enc to match the batch size in x
        pos_enc = pos_enc.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, seq_length, d_model]

        # Add position encodings to input x
        return x + pos_enc
    
def normal_pdf(pos, mu, sigma):
    a = pos - mu
    log_p = -1*torch.mul(a, a)/(2*sigma) - torch.log(sigma)/2
    return F.softmax(log_p, dim=1)