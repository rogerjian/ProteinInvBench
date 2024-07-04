from transformers import AutoTokenizer

from opencpd.modules.pifold_module import *
from opencpd.utils import _dihedrals, _get_rbf, _orientations_coarse_gl_tuple

pair_lst = ['N-N', 'C-C', 'O-O', 'Cb-Cb', 'Ca-N', 'Ca-C', 'Ca-O', 'Ca-Cb', 'N-C', 'N-O', 'N-Cb', 'Cb-C', 'Cb-O', 'O-C',
            'N-Ca', 'C-Ca', 'O-Ca', 'Cb-Ca', 'C-N', 'O-N', 'Cb-N', 'C-Cb', 'O-Cb', 'C-O']


class PiFold_Model(nn.Module):
    method = 'PiFold'
    hidden_dim = 128
    node_features = 128
    edge_features = 128
    k_neighbors = 30
    dropout = 0.1
    num_encoder_layers = 10
    Ca_Ca = 1
    Ca_C = 1
    Ca_N = 1
    Ca_O = 1
    C_C = 1
    C_N = 1
    C_O = 1
    N_N = 1
    N_O = 1
    O_O = 1
    updating_edges = 4
    node_dist = 1
    node_angle = 1
    node_direct = 1
    edge_dist = 1
    edge_angle = 1
    edge_direct = 1
    virtual_num = 3
    att_output_mlp = True
    node_output_mlp = True
    dihedral_type = 0
    NAT = 3
    node_net = 'AttMLP'
    edge_net = 'EdgeMLP'
    node_context = 1
    edge_context = 0
    autoregressive = 0
    AT_layer_num = 3
    use_gvp_feat = 0
    num_decoder_layers1 = 3
    kernel_size1 = 3
    act_type = 'silu'
    num_decoder_layers2 = 3
    kernel_size2 = 3
    glu = 0
    epoch = 20
    batch_size = 8
    lr = 0.001
    patience = 100
    augment_eps = 0.0
    num_rbf = 16
    num_positional_embeddings = 16

    def __init__(self):
        """ Graph labeling network """
        super(PiFold_Model, self).__init__()
        self.top_k = self.k_neighbors

        self.tokenizer = AutoTokenizer.from_pretrained("/Users/daolu/Downloads/transformers/esm2_t33_650M_UR50D")
        alphabet = [one for one in 'ACDEFGHIKLMNPQRSTVWYX']
        self.token_mask = torch.tensor([(one in alphabet) for one in self.tokenizer._token_to_id.keys()])

        # self.full_atom_dis = args.full_atom_dis

        # node_in = 12

        # node_in = node_in + 9 + 576 # node_in + 9 + 576
        prior_matrix = [
            [-0.58273431, 0.56802827, -0.54067466],
            [0.0, 0.83867057, -0.54463904],
            [0.01984028, -0.78380804, -0.54183614],
        ]

        # prior_matrix = torch.rand(self.virtual_num,3)
        self.virtual_atoms = nn.Parameter(torch.tensor(prior_matrix)[:self.virtual_num, :])
        # num_va = self.virtual_atoms.shape[0]
        # edge_in = (15 + 9 * num_va + (num_va - 1) * num_va) * 16 + 16 + 7 

        node_in = 0
        if self.node_dist:
            pair_num = 6
            if self.virtual_num > 0:
                pair_num += self.virtual_num * (self.virtual_num - 1)
            node_in += pair_num * self.num_rbf
        if self.node_angle:
            node_in += 12
        if self.node_direct:
            node_in += 9

        edge_in = 0
        if self.edge_dist:
            pair_num = 0
            if self.Ca_Ca:
                pair_num += 1
            if self.Ca_C:
                pair_num += 2
            if self.Ca_N:
                pair_num += 2
            if self.Ca_O:
                pair_num += 2
            if self.C_C:
                pair_num += 1
            if self.C_N:
                pair_num += 2
            if self.C_O:
                pair_num += 2
            if self.N_N:
                pair_num += 1
            if self.N_O:
                pair_num += 2
            if self.O_O:
                pair_num += 1

            if self.virtual_num > 0:
                pair_num += self.virtual_num
                pair_num += self.virtual_num * (self.virtual_num - 1)
            edge_in += pair_num * self.num_rbf
        if self.edge_angle:
            edge_in += 4
        if self.edge_direct:
            edge_in += 12

        if self.use_gvp_feat:
            node_in = 12
            edge_in = 48 - 16

        edge_in += 16 + 16  # position encoding, chain encoding

        self.node_embedding = nn.Linear(node_in, self.node_features, bias=True)
        self.edge_embedding = nn.Linear(edge_in, self.edge_features, bias=True)
        self.norm_nodes = nn.BatchNorm1d(self.node_features)
        self.norm_edges = nn.BatchNorm1d(self.edge_features)

        self.W_v = nn.Sequential(
            nn.Linear(self.node_features, self.hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        )

        self.W_e = nn.Linear(self.edge_features, self.hidden_dim, bias=True)
        self.W_f = nn.Linear(self.edge_features, self.hidden_dim, bias=True)

        self.encoder = StructureEncoder(self.hidden_dim, self.num_encoder_layers, self.dropout, self.updating_edges,
                                        self.att_output_mlp, self.node_output_mlp, self.node_net, self.edge_net,
                                        self.node_context, self.edge_context)

        # self.decoder = CNNDecoder(hidden_dim, hidden_dim, args.num_decoder_layers1, args.kernel_size1, args.act_type, args.glu)
        # self.decoder2 = CNNDecoder2(hidden_dim, hidden_dim, args.num_decoder_layers2, args.kernel_size2, args.act_type, args.glu)

        self.decoder = MLPDecoder(self.hidden_dim, self.hidden_dim, self.num_decoder_layers1, self.kernel_size1,
                                  self.act_type,
                                  self.glu, vocab=len(self.tokenizer._token_to_id))
        # self.chain_embed = nn.Embedding(2,16)
        self._init_params()

        # self.encode_t = 0
        # self.decode_t = 0

    def forward(self, h_V, h_P, P_idx, batch_id):
        # t1 = time.time()
        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_P = self.W_e(self.norm_edges(self.edge_embedding(h_P)))

        h_V, h_P = self.encoder(h_V, h_P, P_idx, batch_id)
        # t2 = time.time()

        log_probs, logits = self.decoder(h_V, batch_id, self.token_mask)

        # log_probs, logits = self.decoder2(h_V, logits, batch_id)
        # t3 = time.time()

        # self.encode_t += t2 - t1
        # self.decode_t += t3 - t2
        # return log_probs, log_probs0
        return log_probs, logits

    def _init_params(self):
        for name, p in self.named_parameters():
            if name == 'virtual_atoms':
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _full_dist(self, X, mask, top_k=30, eps=1E-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = (1. - mask_2D) * 10000 + mask_2D * torch.sqrt(torch.sum(dX ** 2, 3) + eps)

        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * (D_max + 1)
        D_neighbors, E_idx = torch.topk(D_adjust, min(top_k, D_adjust.shape[-1]), dim=-1, largest=False)
        return D_neighbors, E_idx

    def _get_features(self, S, score, X, mask, chain_mask, chain_encoding):
        # device = X.device
        mask_bool = (mask == 1)
        B, N, _, _ = X.shape
        X_ca = X[:, :, 1, :]
        D_neighbors, E_idx = self._full_dist(X_ca, mask, self.top_k)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = (mask.unsqueeze(-1) * mask_attend) == 1
        edge_mask_select = lambda x: torch.masked_select(x, mask_attend.unsqueeze(-1)).reshape(-1, x.shape[-1])
        node_mask_select = lambda x: torch.masked_select(x, mask_bool.unsqueeze(-1)).reshape(-1, x.shape[-1])

        # sequence
        S = torch.masked_select(S, mask_bool)
        if score is not None:
            score = torch.masked_select(score, mask_bool)
        chain_mask = torch.masked_select(chain_mask, mask_bool)
        chain_encoding = torch.masked_select(chain_encoding, mask_bool)

        # angle & direction
        V_angles = _dihedrals(X, self.dihedral_type)
        V_angles = node_mask_select(V_angles)

        V_direct, E_direct, E_angles = _orientations_coarse_gl_tuple(X, E_idx)
        V_direct = node_mask_select(V_direct)
        E_direct = edge_mask_select(E_direct)
        E_angles = edge_mask_select(E_angles)

        # distance
        atom_N = X[:, :, 0, :]
        atom_Ca = X[:, :, 1, :]
        atom_C = X[:, :, 2, :]
        atom_O = X[:, :, 3, :]
        b = atom_Ca - atom_N
        c = atom_C - atom_Ca
        a = torch.cross(b, c, dim=-1)

        if self.virtual_num > 0:
            virtual_atoms = self.virtual_atoms / torch.norm(self.virtual_atoms, dim=1, keepdim=True)
            for i in range(self.virtual_atoms.shape[0]):
                vars()['atom_v' + str(i)] = virtual_atoms[i][0] * a \
                                            + virtual_atoms[i][1] * b \
                                            + virtual_atoms[i][2] * c \
                                            + 1 * atom_Ca

        node_list = ['Ca-N', 'Ca-C', 'Ca-O', 'N-C', 'N-O', 'O-C']
        node_dist = []
        for pair in node_list:
            atom1, atom2 = pair.split('-')
            node_dist.append(node_mask_select(
                _get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], None, self.num_rbf).squeeze()))

        if self.virtual_num > 0:
            for i in range(self.virtual_atoms.shape[0]):
                for j in range(0, i):
                    node_dist.append(node_mask_select(
                        _get_rbf(vars()['atom_v' + str(i)], vars()['atom_v' + str(j)], None, self.num_rbf).squeeze()))
                    node_dist.append(node_mask_select(
                        _get_rbf(vars()['atom_v' + str(j)], vars()['atom_v' + str(i)], None, self.num_rbf).squeeze()))
        V_dist = torch.cat(tuple(node_dist), dim=-1).squeeze()

        pair_lst = []
        if self.Ca_Ca:
            pair_lst.append('Ca-Ca')
        if self.Ca_C:
            pair_lst.append('Ca-C')
            pair_lst.append('C-Ca')
        if self.Ca_N:
            pair_lst.append('Ca-N')
            pair_lst.append('N-Ca')
        if self.Ca_O:
            pair_lst.append('Ca-O')
            pair_lst.append('O-Ca')
        if self.C_C:
            pair_lst.append('C-C')
        if self.C_N:
            pair_lst.append('C-N')
            pair_lst.append('N-C')
        if self.C_O:
            pair_lst.append('C-O')
            pair_lst.append('O-C')
        if self.N_N:
            pair_lst.append('N-N')
        if self.N_O:
            pair_lst.append('N-O')
            pair_lst.append('O-N')
        if self.O_O:
            pair_lst.append('O-O')

        edge_dist = []  # Ca-Ca
        for pair in pair_lst:
            atom1, atom2 = pair.split('-')
            rbf = _get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], E_idx, self.num_rbf)
            edge_dist.append(edge_mask_select(rbf))

        if self.virtual_num > 0:
            for i in range(self.virtual_atoms.shape[0]):
                edge_dist.append(edge_mask_select(
                    _get_rbf(vars()['atom_v' + str(i)], vars()['atom_v' + str(i)], E_idx, self.num_rbf)))

                for j in range(0, i):
                    edge_dist.append(edge_mask_select(
                        _get_rbf(vars()['atom_v' + str(i)], vars()['atom_v' + str(j)], E_idx, self.num_rbf)))
                    edge_dist.append(edge_mask_select(
                        _get_rbf(vars()['atom_v' + str(j)], vars()['atom_v' + str(i)], E_idx, self.num_rbf)))

        E_dist = torch.cat(tuple(edge_dist), dim=-1)

        h_V = []
        if self.node_dist:
            h_V.append(V_dist)
        if self.node_angle:
            h_V.append(V_angles)
        if self.node_direct:
            h_V.append(V_direct)

        h_E = []
        if self.edge_dist:
            h_E.append(E_dist)
        if self.edge_angle:
            h_E.append(E_angles)
        if self.edge_direct:
            h_E.append(E_direct)

        _V = torch.cat(h_V, dim=-1)
        _E = torch.cat(h_E, dim=-1)

        # edge index
        shift = mask.sum(dim=1).cumsum(dim=0) - mask.sum(dim=1)
        src = shift.view(B, 1, 1) + E_idx

        src = torch.masked_select(src, mask_attend).view(1, -1)
        dst = shift.view(B, 1, 1) + torch.arange(0, N, device=src.device).view(1, -1, 1).expand_as(mask_attend)
        dst = torch.masked_select(dst, mask_attend).view(1, -1)
        E_idx = torch.cat((dst, src), dim=0).long()

        pos_embed = self._positional_embeddings(E_idx, 16)
        _E = torch.cat([_E, pos_embed], dim=-1)

        d_chains = ((chain_encoding[dst.long()] - chain_encoding[src.long()]) == 0).long().reshape(-1)
        chain_embed = self._idx_embeddings(d_chains)
        _E = torch.cat([_E, chain_embed], dim=-1)

        # 3D point
        sparse_idx = mask.nonzero()  # index of non-zero values
        X = X[sparse_idx[:, 0], sparse_idx[:, 1], :, :]
        batch_id = sparse_idx[:, 0]

        return X, S, score, _V, _E, E_idx, batch_id, chain_mask, chain_encoding

    def _positional_embeddings(self, E_idx,
                               num_embeddings=None):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = E_idx[0] - E_idx[1]

        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=E_idx.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d[:, None] * frequency[None, :]
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _idx_embeddings(self, d,
                        num_embeddings=None):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings

        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=d.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d[:, None] * frequency[None, :]
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E
