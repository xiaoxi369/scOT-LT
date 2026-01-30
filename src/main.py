import torch
from datetime import datetime

from config import Config
from util.training import TrainingProcessStage1, TrainingProcessStage2
from util.ot import OT
from util.metrics import evaluate_atac_predictions


def main():
    config = Config()
    torch.manual_seed(config.seed)
    print('Start time: ', datetime.now().strftime('%H:%M:%S'))
    
    # stage1 training
    print('Training start [Stage1]')
    model_stage1= TrainingProcessStage1(config)    
    for epoch in range(config.epochs_stage1):
        print(f'Stage1 Epoch: {epoch}')
        model_stage1.train(epoch)
    
    print('Write embeddings')
    model_stage1.write_embeddings()
    print('Stage 1 finished: ', datetime.now().strftime('%H:%M:%S'))

    print('OT [Stage1]')
    OT(config, epsilon=0.5, geomloss_or_pot='pot', voting_method='max')
    print('OT stage1 finished: ', datetime.now().strftime('%H:%M:%S'))
    
    
    # stage2 training
    print('Training start [Stage2]')
    model_stage2 = TrainingProcessStage2(config)    
    for epoch in range(config.epochs_stage2):
       print(f'Stage2 Epoch: {epoch}')
       model_stage2.train(epoch)
        
    print('Write embeddings [Stage2]')
    model_stage2.write_embeddings()
    print('Stage 2 finished: ', datetime.now().strftime('%H:%M:%S'))
    
    print('OT [Stage2]')
    OT(config, epsilon=0.5, geomloss_or_pot='pot', voting_method='max')
    print('OT stage2 finished: ', datetime.now().strftime('%H:%M:%S'))
    evaluate_atac_predictions(config)

    
if __name__ == "__main__":
    main()