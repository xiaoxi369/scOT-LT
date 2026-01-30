import geomloss.utils
import ot
import torch
import geomloss
import numpy as np
import os



def OT(config, epsilon=0.5, geomloss_or_pot='pot', voting_method='max'):
    # Read RNA data
    print('[OT] Read RNA data')
    db_name = os.path.basename(config.rna_paths[0]).split('.')[0]
    rna_embeddings = np.loadtxt('./output/' + db_name + '_embeddings.txt')
    rna_predictions = np.loadtxt('./output/' + db_name + '_predictions.txt')
    rna_labels = np.loadtxt(config.rna_labels[0])
    
    # Concatenate RNA data from multiple files if available
    for i in range(1, len(config.rna_paths)):
        db_name = os.path.basename(config.rna_paths[i]).split('.')[0]
        rna_embeddings = np.concatenate((rna_embeddings, np.loadtxt('./output/' + db_name + '_embeddings.txt')), 0)
        rna_predictions = np.concatenate((rna_predictions, np.loadtxt('./output/' + db_name + '_predictions.txt')), 0)
        rna_labels = np.concatenate((rna_labels, np.loadtxt(config.rna_labels[i])), 0)
        
    
    # Read ATAC data
    print('[OT] Read ATAC data')
    db_names = []
    db_sizes = []
    db_name = os.path.basename(config.atac_paths[0]).split('.')[0]
    atac_embeddings = np.loadtxt('./output/' + db_name + '_embeddings.txt')
    atac_predictions = np.loadtxt('./output/' + db_name + '_predictions.txt')
    db_names.append(db_name)
    db_sizes.append(atac_embeddings.shape[0])
    
    # Concatenate ATAC data from multiple files if available
    for i in range(1, len(config.atac_paths)):
        db_name = os.path.basename(config.atac_paths[i]).split('.')[0]
        em = np.loadtxt('./output/' + db_name + '_embeddings.txt')
        pred = np.loadtxt('./output/' + db_name + '_predictions.txt')
        atac_embeddings = np.concatenate((atac_embeddings, em), 0)
        atac_predictions = np.concatenate((atac_predictions, pred), 0)
        db_names.append(db_name)
        db_sizes.append(em.shape[0])
    atac_embeddings = np.array(atac_embeddings)
    atac_predictions = np.array(atac_predictions)

    print('[OT] build cost matrix')
   
    if geomloss_or_pot=='pot':
        C = ot.dist(atac_embeddings, rna_embeddings, metric='euclidean')
        # Negative weights are interpreted as costs, scale them to be in [0, 1]
        C -= C.min()
        C += 1
        C = C / C.max()

        # Uniform distributions on cell profiles
        mod1_distr = np.ones(C.shape[0]) / C.shape[0]
        mod2_distr = np.ones(C.shape[1]) / C.shape[1]

        if epsilon > 0:
        # Entropy-regularized OT
            bipartite_matching_adjacency = ot.sinkhorn(
                mod1_distr, mod2_distr, reg=epsilon, M=C, numItermax=5000, verbose=True    # numItermax=5000
            )
        else:
            # Exact OT
            bipartite_matching_adjacency = ot.emd(
                mod1_distr, mod2_distr, M=C, numItermax=10000000
            )
        
    else:
        atac_embeddings=torch.tensor(atac_embeddings)
        rna_embeddings=torch.tensor(rna_embeddings)

        OTLoss = geomloss.SamplesLoss(
        loss='sinkhorn', p=2,
        cost=geomloss.utils.squared_distances,
        blur=0.5**0.5, backend='tensorized',
        potentials=True)

        u, v = OTLoss(atac_embeddings, rna_embeddings)
        M = geomloss.utils.distances(atac_embeddings, rna_embeddings)
        bipartite_matching_adjacency = np.array(torch.exp(1 / 0.5 * (u.t() + v - M)))

    print('[OT] ot')
    # ATAC predict
    atac_predict=[]
    atac_nums,rna_nums=C.shape[0],C.shape[1]


    def predict_atac_labels(bipartite_matching_adjacency, rna_labels, method='max', top_k=5):
        atac_predict = []
        num_classes = len(np.unique(rna_labels))
        
        if method == 'max':
            print("max method")
            for i in range(len(bipartite_matching_adjacency)):
                rna_idx = np.argmax(bipartite_matching_adjacency[i])
                predicted_label = int(rna_labels[rna_idx])
                atac_predict.append(predicted_label)
        
        elif method == 'weighted':
            for i in range(len(bipartite_matching_adjacency)):
                top_k_indices = np.argsort(bipartite_matching_adjacency[i])[-top_k:]
                top_k_weights = bipartite_matching_adjacency[i][top_k_indices]
                
                class_votes = np.zeros(num_classes)
                for idx, weight in zip(top_k_indices, top_k_weights):
                    label_idx = int(rna_labels[idx])
                    if 0 <= label_idx < num_classes: 
                        class_votes[label_idx] += weight
                
                predicted_label = np.argmax(class_votes)
                atac_predict.append(predicted_label)
        
        elif method == 'soft':
            for i in range(len(bipartite_matching_adjacency)):
                class_probs = np.zeros(num_classes)
                for j, weight in enumerate(bipartite_matching_adjacency[i]):
                    label_idx = int(rna_labels[j])
                    if 0 <= label_idx < num_classes: 
                        class_probs[label_idx] += weight
                
                class_probs = class_probs / np.sum(class_probs)
                predicted_label = np.argmax(class_probs)
                atac_predict.append(predicted_label)
        
        elif method == 'multiscale':
            scales = [1, 3, 5]
            for i in range(len(bipartite_matching_adjacency)):
                final_votes = np.zeros(num_classes)
                
                for scale in scales:
                    top_indices = np.argsort(bipartite_matching_adjacency[i])[-scale:]
                    top_weights = bipartite_matching_adjacency[i][top_indices]
                    top_weights = top_weights / np.sum(top_weights)
                    for idx, weight in zip(top_indices, top_weights):
                        label_idx = int(rna_labels[idx])
                        if 0 <= label_idx < num_classes:
                            final_votes[label_idx] += weight / len(scales)
                
                predicted_label = np.argmax(final_votes)
                atac_predict.append(predicted_label)

        return np.array(atac_predict)
    
    atac_predict = predict_atac_labels(bipartite_matching_adjacency, rna_labels, method=voting_method)

    # Save predictions
    cnt = 0
    for i, db_name in enumerate(db_names):
        start = cnt
        end = cnt + db_sizes[i]
        np.savetxt('./output/' + db_name + '_ot_predictions.txt', atac_predict[start:end], fmt='%d')
        cnt = end
