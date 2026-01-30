import torch
import torch.optim as optim
from torch.autograd import Variable
from itertools import cycle
from scipy.linalg import norm
from scipy.special import softmax
import ot
import torch.nn.functional as F

from dataloader import PrepareDataloaderStage1, PrepareDataloaderStage2
from model import Net_encoder, Net_cell
from loss import L1regularization, CellLoss, CenterLoss
from utils import *




def prepare_input(data_list, config):
    output = []
    for data in data_list:
        output.append(Variable(data.to(config.device)))
    return output


def def_cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


class TrainingProcessStage1():
    def __init__(self, config):
        self.config = config
        # load data
        self.train_rna_loaders, self.test_rna_loaders, self.train_atac_loaders, self.test_atac_loaders, self.training_iters = PrepareDataloaderStage1(config).getloader()
        self.training_iteration = 0
        for atac_loader in self.train_atac_loaders:
            self.training_iteration += len(atac_loader)
        
        # initialize dataset      
        if self.config.use_cuda:
            self.model_encoder = torch.nn.DataParallel(Net_encoder(config.input_size).to(self.config.device))
            self.model_cell = torch.nn.DataParallel(Net_cell(config.number_of_class).to(self.config.device))
        else:
            self.model_encoder = Net_encoder(config.input_size).to(self.config.device)
            self.model_cell = Net_cell(config.number_of_class).to(self.config.device)
                
        # initialize criterion (loss)
        self.criterion_cell = CellLoss()
        self.l1_regular = L1regularization()
        
        # initialize optimizer (sgd/momemtum/weight decay)
        self.optimizer_encoder = optim.SGD(self.model_encoder.parameters(), lr=self.config.lr_stage1, momentum=self.config.momentum,
                                           weight_decay=0)
        self.optimizer_cell = optim.SGD(self.model_cell.parameters(), lr=self.config.lr_stage1, momentum=self.config.momentum,
                                        weight_decay=0)


    def adjust_learning_rate(self, optimizer, epoch):
        lr = self.config.lr_stage1 * (0.1 ** ((epoch - 0) // self.config.lr_decay_epoch))
        if (epoch - 0) % self.config.lr_decay_epoch == 0:
            print('LR is set to {}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def load_checkpoint(self, args):
        if self.config.checkpoint is not None:
            if os.path.isfile(self.config.checkpoint):
                print("=> loading checkpoint '{}'".format(self.config.checkpoint))
                checkpoint = torch.load(self.config.checkpoint)                
                self.model_encoder.load_state_dict(checkpoint['model_encoding_state_dict'])
                self.model_cell.load_state_dict(checkpoint['model_cell_state_dict'])
            else:
                print("=> no resume checkpoint found at '{}'".format(self.config.checkpoint))


    def train(self, epoch):
        self.model_encoder.train()
        self.model_cell.train()
        total_cell_loss, total_transfer_loss = 0., 0.
        self.adjust_learning_rate(self.optimizer_encoder, epoch)
        self.adjust_learning_rate(self.optimizer_cell, epoch)

        # initialize iterator
        iter_rna_loaders = []
        iter_atac_loaders = []
        for rna_loader in self.train_rna_loaders:
            iter_rna_loaders.append(def_cycle(rna_loader))
        for atac_loader in self.train_atac_loaders:
            iter_atac_loaders.append(def_cycle(atac_loader))
                
        for batch_idx in range(self.training_iters):
            # rna forward
            rna_embeddings = []
            rna_cell_predictions = []
            rna_labels = []
            for iter_rna_loader in iter_rna_loaders:
                rna_data, rna_label = next(iter_rna_loader)    
                # prepare data
                rna_data, rna_label = prepare_input([rna_data, rna_label], self.config)
                # model forward
                rna_embedding = self.model_encoder(rna_data)
                rna_cell_prediction = self.model_cell(rna_embedding)
                rna_embeddings.append(rna_embedding)
                rna_cell_predictions.append(rna_cell_prediction)
                rna_labels.append(rna_label)
                
            # atac forward
            atac_embeddings = []
            atac_cell_predictions = []
            for iter_atac_loader in iter_atac_loaders:
                atac_data = next(iter_atac_loader)    
                # prepare data
                atac_data = prepare_input([atac_data], self.config)[0]
                # model forward
                atac_embedding = self.model_encoder(atac_data)
                atac_cell_prediction = self.model_cell(atac_embedding)

                atac_embeddings.append(atac_embedding)
                atac_cell_predictions.append(atac_cell_prediction)
            
            
            # caculate loss  
            cell_loss = self.criterion_cell(rna_cell_predictions[0], rna_labels[0])
            for i in range(1, len(rna_cell_predictions)):
                cell_loss += self.criterion_cell(rna_cell_predictions[i], rna_labels[i])
            cell_loss = cell_loss/len(rna_cell_predictions)
            
            regularization_loss_encoder = self.l1_regular(self.model_encoder)   
            regularization_loss_cell = self.l1_regular(self.model_cell)         


            # transfer loss
            total_transfer_loss = 0
            batch_count = min(len(rna_embeddings), len(atac_embeddings))
            
            def cosine_distance(x, y):
                x_norm = F.normalize(x, p=2, dim=1)
                y_norm = F.normalize(y, p=2, dim=1)
                cosine_sim = torch.mm(x_norm, y_norm.t())
                cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)
                return 1 - cosine_sim

            for i in range(batch_count):
                M_embed = torch.cdist(rna_embeddings[i], atac_embeddings[i]) ** 2
                
                rna_labels_onehot = F.one_hot(rna_labels[i].long(), num_classes=self.config.number_of_class).float()
                atac_pred_softmax = F.softmax(atac_cell_predictions[i], dim=1)
                M_sce = -torch.mm(rna_labels_onehot, torch.transpose(torch.log(atac_pred_softmax), 0, 1))
                
                M = self.config.alpha * M_embed + self.config.lambda_t * M_sce
                
                a, b = ot.unif(len(rna_embeddings[i])), ot.unif(len(atac_embeddings[i]))
                pi = ot.unbalanced.sinkhorn_knopp_unbalanced(
                    a, b, M.detach().cpu().numpy(), 
                    reg=self.config.reg, reg_m=self.config.reg_m, reg_type='entropy'
                )
                pi = torch.from_numpy(pi).float().to(self.config.device)
                
                batch_transfer_loss = torch.sum(pi * M)
                total_transfer_loss += batch_transfer_loss
            
            transfer_loss = total_transfer_loss / batch_count

            
            total_loss = (cell_loss + transfer_loss + regularization_loss_encoder + regularization_loss_cell)
            
            self.optimizer_encoder.zero_grad()
            self.optimizer_cell.zero_grad()
            total_loss.backward()
            self.optimizer_encoder.step()
            self.optimizer_cell.step()

            # print log
            total_cell_loss += cell_loss.data.item()
            total_transfer_loss += transfer_loss.item()

            progress_bar(batch_idx, self.training_iters,
                         'rna_loss: %.3f , transfer_loss: %.3f' % (
                             total_cell_loss / (batch_idx + 1),
                             total_transfer_loss / (batch_idx + 1)
                             ))
        # save checkpoint
        save_checkpoint({
            'epoch': epoch,
            'model_cell_state_dict': self.model_cell.state_dict(),
            'model_encoding_state_dict': self.model_encoder.state_dict(),
            'optimizer': self.optimizer_cell.state_dict()            
        })
        
        
    def write_embeddings(self):
        self.model_encoder.eval()
        self.model_cell.eval()
        if not os.path.exists("output/"):
            os.makedirs("output/")
        
        # rna db
        for i, rna_loader in enumerate(self.test_rna_loaders):
            db_name = os.path.basename(self.config.rna_paths[i]).split('.')[0]
            fp_em = open('./output/' + db_name + '_embeddings.txt', 'w')
            fp_pre = open('./output/' + db_name + '_predictions.txt', 'w')
            for batch_idx, (rna_data, rna_label) in enumerate(rna_loader):    
                # prepare data
                rna_data, rna_label = prepare_input([rna_data, rna_label], self.config)
                    
                # model forward
                rna_embedding = self.model_encoder(rna_data)
                rna_cell_prediction = self.model_cell(rna_embedding)
                            
                rna_embedding = rna_embedding.data.cpu().numpy()
                rna_cell_prediction = rna_cell_prediction.data.cpu().numpy()
                
                # normalization & softmax
                rna_embedding = rna_embedding / norm(rna_embedding, axis=1, keepdims=True)
                rna_cell_prediction = softmax(rna_cell_prediction, axis=1)
                                
                # write embeddings
                test_num, embedding_size = rna_embedding.shape
                for print_i in range(test_num):
                    fp_em.write(str(rna_embedding[print_i][0]))
                    for print_j in range(1, embedding_size):
                        fp_em.write(' ' + str(rna_embedding[print_i][print_j]))
                    fp_em.write('\n')
                    
                # write predictions
                test_num, prediction_size = rna_cell_prediction.shape
                for print_i in range(test_num):
                    fp_pre.write(str(rna_cell_prediction[print_i][0]))
                    for print_j in range(1, prediction_size):
                        fp_pre.write(' ' + str(rna_cell_prediction[print_i][print_j]))
                    fp_pre.write('\n')
                
                progress_bar(batch_idx, len(rna_loader),
                         'write embeddings and predictions for db:' + db_name)                    
            fp_em.close()
            fp_pre.close()
        
        
        # atac db
        for i, atac_loader in enumerate(self.test_atac_loaders):
            db_name = os.path.basename(self.config.atac_paths[i]).split('.')[0]
            fp_em = open('./output/' + db_name + '_embeddings.txt', 'w')
            fp_pre = open('./output/' + db_name + '_predictions.txt', 'w')
            for batch_idx, (atac_data) in enumerate(atac_loader):    
                # prepare data
                atac_data = prepare_input([atac_data], self.config)[0]
                
                # model forward
                atac_embedding = self.model_encoder(atac_data)
                atac_cell_prediction = self.model_cell(atac_embedding)
                                
                                
                atac_embedding = atac_embedding.data.cpu().numpy()
                atac_cell_prediction = atac_cell_prediction.data.cpu().numpy()
                
                # normalization & softmax
                atac_embedding = atac_embedding / norm(atac_embedding, axis=1, keepdims=True)
                atac_cell_prediction = softmax(atac_cell_prediction, axis=1)
                
                # write embeddings
                test_num, embedding_size = atac_embedding.shape
                for print_i in range(test_num):
                    fp_em.write(str(atac_embedding[print_i][0]))
                    for print_j in range(1, embedding_size):
                        fp_em.write(' ' + str(atac_embedding[print_i][print_j]))
                    fp_em.write('\n')
                    
                # write predictions
                test_num, prediction_size = atac_cell_prediction.shape
                for print_i in range(test_num):
                    fp_pre.write(str(atac_cell_prediction[print_i][0]))
                    for print_j in range(1, prediction_size):
                        fp_pre.write(' ' + str(atac_cell_prediction[print_i][print_j]))
                    fp_pre.write('\n')
                
                progress_bar(batch_idx, len(atac_loader),
                         'write embeddings and predictions for db:' + db_name)                    
            fp_em.close()
            fp_pre.close()



class TrainingProcessStage2():
    def __init__(self, config):
        self.config = config
        # load data
        self.train_rna_loaders, self.test_rna_loaders, self.train_atac_loaders, self.test_atac_loaders, self.training_iters = PrepareDataloaderStage2(config).getloader()
        self.training_iteration = 0
        for atac_loader in self.train_atac_loaders:
            self.training_iteration += len(atac_loader)
        
        # initialize dataset       
        if self.config.use_cuda:  
            self.model_encoder = torch.nn.DataParallel(Net_encoder(config.input_size).to(self.config.device))
            self.model_cell = torch.nn.DataParallel(Net_cell(config.number_of_class).to(self.config.device))
        else:
            self.model_encoder = Net_encoder(config.input_size).to(self.config.device)
            self.model_cell = Net_cell(config.number_of_class).to(self.config.device)
                
        # initialize criterion (loss)
        self.criterion_cell = CellLoss()
        self.criterion_center = CenterLoss(self.config.number_of_class, use_gpu = self.config.use_cuda)
        self.l1_regular = L1regularization()
        
        # initialize optimizer (sgd/momemtum/weight decay)
        self.optimizer_encoder = optim.SGD(self.model_encoder.parameters(), lr=self.config.lr_stage3, momentum=self.config.momentum,
                                           weight_decay=0)
        self.optimizer_cell = optim.SGD(self.model_cell.parameters(), lr=self.config.lr_stage3, momentum=self.config.momentum,
                                        weight_decay=0)


    def adjust_learning_rate(self, optimizer, epoch):
        lr = self.config.lr_stage3 * (0.1 ** ((epoch - 0) // self.config.lr_decay_epoch))
        if (epoch - 0) % self.config.lr_decay_epoch == 0:
            print('LR is set to {}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def load_checkpoint(self, args):
        if self.config.checkpoint is not None:
            if os.path.isfile(self.config.checkpoint):
                print("=> loading checkpoint '{}'".format(self.config.checkpoint))
                checkpoint = torch.load(self.config.checkpoint)                
                self.model_encoder.load_state_dict(checkpoint['model_encoding_state_dict'])
                self.model_cell.load_state_dict(checkpoint['model_cell_state_dict'])
            else:
                print("=> no resume checkpoint found at '{}'".format(self.config.checkpoint))


    def train(self, epoch):
        self.model_encoder.train()
        self.model_cell.train()
        total_cell_loss, total_center_loss, total_transfer_loss = 0., 0., 0.
        self.adjust_learning_rate(self.optimizer_encoder, epoch)
        self.adjust_learning_rate(self.optimizer_cell, epoch)

        # initialize iterator
        iter_rna_loaders = []
        iter_atac_loaders = []
        for rna_loader in self.train_rna_loaders:
            iter_rna_loaders.append(def_cycle(rna_loader))
        for atac_loader in self.train_atac_loaders:
            iter_atac_loaders.append(def_cycle(atac_loader))
                
        for batch_idx in range(self.training_iters):
            # rna forward
            rna_embeddings = []
            rna_cell_predictions = []
            rna_labels = []
            for iter_rna_loader in iter_rna_loaders:
                rna_data, rna_label = next(iter_rna_loader)    
                # prepare data
                rna_data, rna_label = prepare_input([rna_data, rna_label], self.config)
                # model forward
                rna_embedding = self.model_encoder(rna_data)
                rna_cell_prediction = self.model_cell(rna_embedding)
                rna_embeddings.append(rna_embedding)
                rna_cell_predictions.append(rna_cell_prediction)
                rna_labels.append(rna_label)
                
            # atac forward
            atac_embeddings = []
            atac_cell_predictions = []
            atac_labels = []
            for iter_atac_loader in iter_atac_loaders:
                atac_data, atac_label = next(iter_atac_loader)    
                # prepare data
                atac_data, atac_label = prepare_input([atac_data, atac_label], self.config)
                # model forward
                atac_embedding = self.model_encoder(atac_data)
                atac_cell_prediction = self.model_cell(atac_embedding)

                atac_embeddings.append(atac_embedding)
                atac_cell_predictions.append(atac_cell_prediction)
                atac_labels.append(atac_label)
            
            # caculate loss  
            if self.config.with_crossentorpy == True:
                cell_loss = self.criterion_cell(rna_cell_predictions[0], rna_labels[0])
                for i in range(1, len(rna_cell_predictions)):
                    cell_loss += self.criterion_cell(rna_cell_predictions[i], rna_labels[i])

                cell_loss = cell_loss/len(rna_cell_predictions)    

                atac_cell_loss = self.criterion_cell(atac_cell_predictions[0], atac_labels[0])
                for i in range(1, len(atac_cell_predictions)):
                    atac_cell_loss += self.criterion_cell(atac_cell_predictions[i], atac_labels[i])
                
                cell_loss += atac_cell_loss/len(atac_cell_predictions)
            else: 
                cell_loss = 0
                
            
            center_loss = self.config.center_weight*(self.criterion_center(atac_embeddings, atac_labels) + self.criterion_center(rna_embeddings, rna_labels))
            regularization_loss_encoder = self.l1_regular(self.model_encoder)   
            regularization_loss_cell = self.l1_regular(self.model_cell)        


            # transfer loss
            total_transfer_loss = 0
            batch_count = min(len(rna_embeddings), len(atac_embeddings))
            
            for i in range(batch_count):
                M_embed = torch.cdist(rna_embeddings[i], atac_embeddings[i]) ** 2
                rna_labels_onehot = F.one_hot(rna_labels[i].long(), num_classes=self.config.number_of_class).float()
                atac_pred_softmax = F.softmax(atac_cell_predictions[i], dim=1)
                M_sce = -torch.mm(rna_labels_onehot, torch.transpose(torch.log(atac_pred_softmax), 0, 1))
                M = self.config.alpha * M_embed + self.config.lambda_t * M_sce
                a, b = ot.unif(len(rna_embeddings[i])), ot.unif(len(atac_embeddings[i]))
                pi = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M.detach().cpu().numpy(), reg=self.config.reg, reg_m=self.config.reg_m, reg_type='entropy')
                pi = torch.from_numpy(pi).float().to(self.config.device)
                batch_transfer_loss = torch.sum(pi * M)
                total_transfer_loss += batch_transfer_loss
            transfer_loss = total_transfer_loss / batch_count


            total_loss = (cell_loss + transfer_loss + center_loss + 
                            regularization_loss_encoder + regularization_loss_cell)
            self.optimizer_encoder.zero_grad()
            self.optimizer_cell.zero_grad()
            total_loss.backward()
            self.optimizer_encoder.step()
            self.optimizer_cell.step()

            # print log
            if self.config.with_crossentorpy == True:
                total_cell_loss += cell_loss.data.item()
            else: 
                total_cell_loss += 0
            total_center_loss += center_loss.data.item()
            total_transfer_loss += transfer_loss.item()

            progress_bar(batch_idx, self.training_iters,
                         'rna_loss: %.3f, center_loss: %.3f, transfer_loss: %.3f' % (
                             total_cell_loss / (batch_idx + 1), 
                             total_center_loss / (batch_idx + 1),
                             total_transfer_loss / (batch_idx + 1)  
                             ))
                             
                             
        # save checkpoint
        save_checkpoint({
            'epoch': epoch,
            'model_cell_state_dict': self.model_cell.state_dict(),
            'model_encoding_state_dict': self.model_encoder.state_dict(),
            'optimizer': self.optimizer_cell.state_dict()            
        })
        
        
    def write_embeddings(self):
        self.model_encoder.eval()
        self.model_cell.eval()
        if not os.path.exists("output/"):
            os.makedirs("output/")
        
        # rna db
        for i, rna_loader in enumerate(self.test_rna_loaders):
            db_name = os.path.basename(self.config.rna_paths[i]).split('.')[0]
            fp_em = open('./output/' + db_name + '_embeddings.txt', 'w')
            for batch_idx, (rna_data, rna_label) in enumerate(rna_loader):    
                # prepare data
                rna_data, rna_label = prepare_input([rna_data, rna_label], self.config)
                    
                # model forward
                rna_embedding = self.model_encoder(rna_data)
                rna_cell_prediction = self.model_cell(rna_embedding)
                            
                rna_embedding = rna_embedding.data.cpu().numpy()
                rna_cell_prediction = rna_cell_prediction.data.cpu().numpy()
                
                # normalization & softmax
                rna_embedding = rna_embedding / norm(rna_embedding, axis=1, keepdims=True)
                rna_cell_prediction = softmax(rna_cell_prediction, axis=1)
                                
                # write embeddings
                test_num, embedding_size = rna_embedding.shape
                for print_i in range(test_num):
                    fp_em.write(str(rna_embedding[print_i][0]))
                    for print_j in range(1, embedding_size):
                        fp_em.write(' ' + str(rna_embedding[print_i][print_j]))
                    fp_em.write('\n')
                    
                                
                progress_bar(batch_idx, len(rna_loader),
                         'write embeddings for db:' + db_name)                    
            fp_em.close()
        
        
        # atac db
        for i, atac_loader in enumerate(self.test_atac_loaders):
            db_name = os.path.basename(self.config.atac_paths[i]).split('.')[0]
            fp_em = open('./output/' + db_name + '_embeddings.txt', 'w')
            fp_pre = open('./output/' + db_name + '_predictions.txt', 'w')
            for batch_idx, (atac_data, atac_label) in enumerate(atac_loader):    
                # prepare data
                atac_data, atac_label = prepare_input([atac_data, atac_label], self.config)
                
                # model forward
                atac_embedding = self.model_encoder(atac_data)
                atac_cell_prediction = self.model_cell(atac_embedding)
                                
                                
                atac_embedding = atac_embedding.data.cpu().numpy()
                atac_cell_prediction = atac_cell_prediction.data.cpu().numpy()
                
                # normalization & softmax
                atac_embedding = atac_embedding / norm(atac_embedding, axis=1, keepdims=True)
                atac_cell_prediction = softmax(atac_cell_prediction, axis=1)
                
                # write embeddings
                test_num, embedding_size = atac_embedding.shape
                for print_i in range(test_num):
                    fp_em.write(str(atac_embedding[print_i][0]))
                    for print_j in range(1, embedding_size):
                        fp_em.write(' ' + str(atac_embedding[print_i][print_j]))
                    fp_em.write('\n')
                    
                # write predictions
                test_num, prediction_size = atac_cell_prediction.shape
                for print_i in range(test_num):
                    fp_pre.write(str(atac_cell_prediction[print_i][0]))
                    for print_j in range(1, prediction_size):
                        fp_pre.write(' ' + str(atac_cell_prediction[print_i][print_j]))
                    fp_pre.write('\n')

                progress_bar(batch_idx, len(atac_loader),
                         'write embeddings and predictions for db:' + db_name)                    
            fp_em.close()
            fp_pre.close()

