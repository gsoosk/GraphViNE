
import torch
from torch.nn import Linear
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch_geometric.nn import ARGVA, VGAE
from sklearn.metrics import accuracy_score, silhouette_score, davies_bouldin_score
from torch_geometric.nn.conv import MessagePassing


class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W_2} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, normalize=False, bias=True,
                 **kwargs):
        super(SAGEConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.lin_rel = Linear(in_channels, out_channels, bias=bias)
        self.lin_root = Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        """"""

        if torch.is_tensor(x):
            x = (x, x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = self.lin_rel(out) + self.lin_root(x[1])

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        return out

    def message(self, x_j, edge_weight):
        # print(f'Message is {x_j} with weight {}')
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class Discriminator(torch.nn.Module):
    def __init__(self, in_channel, hidden, out_channel):
        super(Discriminator, self).__init__()
        self.dense1 = torch.nn.Linear(in_channel, hidden)
        self.dense2 = torch.nn.Linear(hidden, out_channel)
        self.dense3 = torch.nn.Linear(out_channel, out_channel)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv_mu = SAGEConv(hidden, out_channels)
        self.conv_logvar = SAGEConv(hidden, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        return self.conv_mu(x, edge_index, edge_attr), self.conv_logvar(x, edge_index, edge_attr)


def cluster_using_argva(data, verbose=True, max_epoch=150, pre_trained_model=None, gpu=True):
    data = data.clone()

    def train(data):
        model.train()
        model_optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index, data.edge_attr)

        for i in range(5):
            discriminator.train()
            discriminator_optimizer.zero_grad()
            discriminator_loss = model.discriminator_loss(z)
            discriminator_loss.backward()
            discriminator_optimizer.step()

        loss = model.recon_loss(z, data.edge_index)
        loss = loss + (1 / data.num_nodes) * model.kl_loss()
        loss.backward()
        model_optimizer.step()
        return loss

    def test(data):
        model.eval()
        kmeans_input = []
        with torch.no_grad():
            z = model.encode(data.x, data.edge_index, data.edge_attr)
            kmeans_input = z.cpu().data.numpy()
        kmeans = KMeans(n_clusters=4, random_state=0).fit(kmeans_input)
        pred_label = kmeans.predict(kmeans_input)
        X = data.x.cpu().data

        s = silhouette_score(X, pred_label)
        davies = davies_bouldin_score(X, pred_label)

        return s, davies

    encoder = Encoder(data.num_features, 16, 16)
    discriminator = Discriminator(16, 32, 16)
    model = None
    if pre_trained_model is not None:
        model = pre_trained_model
    else:
        model = ARGVA(encoder, discriminator)

    discriminator_optimizer = torch.optim.Adam(
        model.discriminator.parameters(), lr=0.001)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=5*0.001)

    device = torch.device(
        'cuda' if torch.cuda.is_available() and gpu else 'cpu')
    discriminator.to(device)
    model.to(device)
    data = data.to(device)

    for epoch in range(max_epoch):
        loss = train(data)
        if verbose or epoch == max_epoch-1:
            s, davies = test(data)
            print('Epoch: {:05d}, '
                  'Train Loss: {:.3f}, Silhoeute: {:.3f}, Davies: {:.3f}'.format(epoch, loss, s, davies))
    return model
