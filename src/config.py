import torch
import os

class Config(object):
    def __init__(self):
        DB = "PBMC"
        self.use_cuda = True
        self.threads = 1

        if not self.use_cuda:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0')
        
        if DB == '10x':
            # DB info
            self.number_of_class = 11
            self.input_size = 15463
            self.rna_paths = ['data_10x/exprs_10xPBMC_rna.npz']
            self.rna_labels = ['data_10x/cellType_10xPBMC_rna.txt']		
            self.atac_paths = ['data_10x/exprs_10xPBMC_atac.npz']
            self.atac_labels = []
            self.rna_protein_paths = []
            self.atac_protein_paths = []
            
            # Training config            
            self.batch_size = 256
            self.lr_stage1 = 0.01
            self.lr_stage3 = 0.01
            self.lr_decay_epoch = 20
            self.epochs_stage1 = 20
            self.epochs_stage3 = 20
            self.p = 0.8
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 1
            self.with_crossentorpy = True
            self.seed = 1
            self.alpha = 1
            self.lambda_t = 0.75
            self.reg = 0.5
            self.reg_m = 0.1
            self.checkpoint = ''
            
        elif DB == "PBMC":
            self.number_of_class = 7
            self.input_size = 17668
            self.rna_paths = ['PBMC/citeseq_control_rna.npz']
            self.rna_labels = ['PBMC/citeseq_control_cellTypes.txt']
            self.atac_paths = ['PBMC/asapseq_control_atac.npz']
            self.atac_labels = ['PBMC/asapseq_control_cellTypes.txt']
            self.rna_protein_paths = ['PBMC/citeseq_control_adt.npz']
            self.atac_protein_paths = ['PBMC/asapseq_control_adt.npz']
            self.label_map = 'PBMC/label_to_idx.txt'
            self.label_type_nums=7
            
            # Training config            
            self.batch_size = 256
            self.lr_stage1 = 0.001
            self.lr_stage3 = 0.001
            self.lr_decay_epoch = 20
            self.epochs_stage1 = 20
            self.epochs_stage3 = 20
            self.p = 0.8
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 4.5
            self.with_crossentorpy = True
            self.seed = 1
            self.alpha = 0.03
            self.lambda_t = 0.07
            self.reg = 0.01
            self.reg_m = 3
            self.checkpoint = ''