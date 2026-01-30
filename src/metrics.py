import numpy as np
import umap
import os
import seaborn as sns
import scanpy as sc
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, silhouette_score

from config import Config


def load_label_map(file_path):
    label_map = {}
    with open(file_path, 'r') as f:
        for line in f:
            label, idx = line.strip().rsplit(' ', 1)  # 从右侧分割一次
            # if idx == '-1':  # 处理 -1 的情况
            #     label_map[-1] = "Unknown"
            # else:
            label_map[int(idx)] = label
    return label_map


def calc_frac_idx(x1_mat, x2_mat):
    """
    Returns fraction closer than true match for each sample (as an array).
    :param x1_mat: numpy array of shape (nsamp, nfeat)
    :param x2_mat: numpy array of shape (nsamp, nfeat)
    :return: fracs (list of fractions), x (list of indices)
    """
    
    euc_dist = cdist(x1_mat, x2_mat, metric='euclidean')
    true_nbr = np.diag(euc_dist)
    fracs = np.mean(euc_dist < true_nbr[:, np.newaxis], axis=1)
    x = np.arange(1, x1_mat.shape[0] + 1)
    return fracs.tolist(), x.tolist()


def calc_domainAveraged_FOSCTTM(x1_mat, x2_mat):
    """
    Outputs average FOSCTTM measure (averaged over both domains).
    :param x1_mat: numpy array of shape (nsamp, nfeat)
    :param x2_mat: numpy array of shape (nsamp, nfeat)
    :return: fracs (list of averaged fractions)
    """
    
    fracs1, xs = calc_frac_idx(x1_mat, x2_mat)
    fracs2, xs = calc_frac_idx(x2_mat, x1_mat)
    fracs = [(f1 + f2) / 2 for f1, f2 in zip(fracs1, fracs2)]
    
    return fracs


def evaluate_atac_predictions(config):
    if len(config.atac_labels) == len(config.atac_paths):
        # read rna embeddings and predictions
        db_name = os.path.basename(config.rna_paths[0]).split('.')[0]
        rna_embeddings = np.loadtxt('./output/' + db_name + '_embeddings.txt')
        rna_predictions = np.loadtxt('./output/' + db_name + '_predictions.txt')
        rna_labels = np.loadtxt(config.rna_labels[0])    
        for i in range(1, len(config.rna_paths)):
            db_name = os.path.basename(config.rna_paths[i]).split('.')[0]
            rna_embeddings = np.concatenate((rna_embeddings, np.loadtxt('./output/' + db_name + '_embeddings.txt')), 0)
            rna_predictions = np.concatenate((rna_predictions, np.loadtxt('./output/' + db_name + '_predictions.txt')), 0)
            rna_labels = np.concatenate((rna_labels, np.loadtxt(config.rna_labels[i])), 0)
        
        # read atac embeddings and predictions
        db_names = []
        db_sizes = []
        db_name = os.path.basename(config.atac_paths[0]).split('.')[0]    
        atac_embeddings = np.loadtxt('./output/' + db_name + '_embeddings.txt')
        
        # Read predictions
        pred_data = np.loadtxt('./output/' + db_name + '_ot_predictions.txt')
        if pred_data.ndim == 1:
            atac_predictions = pred_data.astype(int)
        else:
            atac_predictions = pred_data[:, 0].astype(int)
        
        db_names.append(db_name)
        db_sizes.append(atac_embeddings.shape[0])
        
        for i in range(1, len(config.atac_paths)):
            db_name = os.path.basename(config.atac_paths[i]).split('.')[0]        
            em = np.loadtxt('./output/' + db_name + '_embeddings.txt')
            
            # Read predictions
            pred_data = np.loadtxt('./output/' + db_name + '_ot_predictions.txt')
            if pred_data.ndim == 1:
                pred = pred_data.astype(int)
            else:
                pred = pred_data[:, 0].astype(int)
            
            atac_embeddings = np.concatenate((atac_embeddings, em), 0)
            atac_predictions = np.concatenate((atac_predictions, pred), 0)        
            db_names.append(db_name)
            db_sizes.append(em.shape[0])

        # Read true ATAC labels
        true_labels = np.loadtxt(config.atac_labels[0], dtype=int)
        for i in range(1, len(config.atac_labels)):
            true_labels = np.concatenate((true_labels, np.loadtxt(config.atac_labels[i], dtype=int)), 0)

        valid_sample_cnt = 0
        valid_sample=[]
        atac_valid_predict=[]
        valid_indices = []
        correct = 0
        for i in range(atac_predictions.shape[0]):
            if true_labels[i] >= 0:
                valid_sample_cnt += 1
                valid_sample.append(true_labels[i])
                atac_valid_predict.append(atac_predictions[i])
                valid_indices.append(i)
                if true_labels[i] == atac_predictions[i]:
                    correct += 1

        try:
            accuracy = correct*1./valid_sample_cnt
            f1_weight = f1_score(valid_sample, atac_valid_predict, average='weighted')
            f1_macro = f1_score(valid_sample, atac_valid_predict, average='macro')
            kappa = cohen_kappa_score(valid_sample, atac_valid_predict)  
            print("Accuracy: %.3f, F1 weighted score: %.3f, F1 macro score: %.3f, Cohen's Kappa: %.3f" % (accuracy, f1_weight, f1_macro, kappa))

            # UMAP Visualization
            print("Running UMAP visualization...")
            label_map = load_label_map(config.label_map)

            combined_embeddings = np.concatenate((rna_embeddings, atac_embeddings), axis=0)
            combined_labels = np.concatenate((rna_labels, true_labels), axis=0)
            combined_labels = np.array([label_map[int(label)] for label in combined_labels])
            combined_modalities = np.array(['RNA-seq'] * len(rna_embeddings) + ['ATAC-seq'] * len(atac_embeddings))
            umap_reducer = umap.UMAP()
            umap_embeddings = umap_reducer.fit_transform(combined_embeddings)

            sc.set_figure_params(figsize=(7, 7), dpi=300)
            adata = sc.AnnData(combined_embeddings)
            adata.obs["CellType"] = [str(label) for label in combined_labels]
            adata.obs["Modality"] = combined_modalities

            sc.pp.neighbors(adata, use_rep="X")
            sc.tl.umap(adata)

            palette = sns.color_palette("husl", 8)
            sc.pl.umap(
                adata,
                color="CellType",
                palette=sns.color_palette("husl", np.unique(adata.obs["CellType"].values).size),
                title="scOT-LT",
                frameon=False,
                save=f'_{db_name}_CellType.png',
                save=False,
                show=False
            )
            
            sc.pl.umap(
                adata,
                color="Modality",
                palette=sns.color_palette("hls", 2),
                title="scOT-LT",
                frameon=False,
                save=f'_{db_name}_Modality.png',
                save=False,
                show=False
            )
            
            print("Calculating silhouette scores...")
            sil_type = silhouette_score(
                adata.obsm["X_umap"], adata.obs["CellType"]
            )
            sil_omic = silhouette_score(
                adata.obsm["X_umap"], adata.obs["Modality"]
            )
            sil_f1 = (
                2
                * (1 - (sil_omic + 1) / 2)
                * (sil_type + 1)
                / 2
                / (1 - (sil_omic + 1) / 2 + (sil_type + 1) / 2)
            )
            print("Silhouette Score Type: %.3f, Omics: %.3f, Harmonized: %.3f"% (sil_type, sil_omic, sil_f1))
            
            # FOSCTTM
            foscttm_val = -1
            if hasattr(config, 'focsttm') and config.focsttm:
                fracs = calc_domainAveraged_FOSCTTM(rna_embeddings, atac_embeddings)
                foscttm_val = float(np.mean(fracs))
                print("FOSCTTM: %.3f" % foscttm_val)
            
        except Exception as e:
            print(f"Metric calculation failed: {str(e)}")  

    else:
        print("ATAC labels not available for evaluation.")



if __name__ == "__main__":
    config = Config()
    evaluate_atac_predictions(config)