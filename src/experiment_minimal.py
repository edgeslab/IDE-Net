
import torch
import pickle as pkl
import torch.nn as nn
import torch.optim as optim
import utils as utils
import numpy as np
import os


class ExperimentMinimal():

    def __init__(self,args,model,trainA, trainX, trainT, POTrain):
        super(ExperimentMinimal, self).__init__()

        self.args = args
        self.model = model
        if self.args['model']=="NetEstimator":
            self.optimizerD = optim.Adam([{'params':self.model.discriminator.parameters()}],lr=self.args['lrD'], weight_decay=self.args['weight_decay'])
            self.optimizerD_z = optim.Adam([{'params':self.model.discriminator_z.parameters()}],lr=self.args['lrD_z'], weight_decay=self.args['weight_decay'])
            self.optimizerP = optim.Adam([{'params':self.model.encoder.parameters()},{'params':self.model.predictor.parameters()}],lr=self.args['lr'], weight_decay=self.args['weight_decay'])
            
        elif self.args['model'] == 'TNet':
            self.optimizerT = optim.Adam(self.model.parameter_base(), lr=self.args['lr_1step'], weight_decay=self.args['weight_decay_tr'])
            self.optimizer2step = optim.Adam(self.model.parameters(), lr=self.args['lr_2step'], weight_decay=self.args['weight_decay_tr'])

        else:
            self.optimizerB = optim.Adam(self.model.parameters(),lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        if self.args['cuda']:
            self.model = self.model.cuda()
        # print ("================================Model================================")
        # print(self.model)

        self.Tensor = torch.cuda.FloatTensor if self.args['cuda'] else torch.FloatTensor
        self.trainA = trainA 
        self.trainX = trainX
        self.trainT = trainT
        self.trainZ = self.compute_z(self.trainT,self.trainA)
        self.POTrain = POTrain


        """PO normalization if any"""
        self.YFTrain = utils.PO_normalize_minimal(self.args['normy'],self.POTrain,self.POTrain)

        self.loss = nn.MSELoss()
        self.d_zLoss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.peheLoss = nn.MSELoss()

        
        self.alpha = self.Tensor([self.args['alpha']])
        self.gamma = self.Tensor([self.args['gamma']])
        self.alpha_base = self.Tensor([self.args['alpha_base']])
        if self.args['cuda']:
            torch.cuda.manual_seed(self.args['seed'])
            self.loss = self.loss.cuda()
            self.bce_loss = self.bce_loss.cuda()
            self.peheLoss = self.peheLoss.cuda()

        self.lossTrain = []

        self.dissTrain = []
        self.dissTrainHalf = []
        self.diss_zTrain = []
        self.diss_zTrainHalf = []

        self.labelTrain = []

        self.predT = []
        self.labelT = []


    def get_peheLoss(self,y1pred,y0pred,y1gt,y0gt):
        pred = y1pred - y0pred
        gt = y1gt - y0gt
        return torch.sqrt(self.peheLoss(pred,gt))
    
    def compute_z(self,T,A):
        # print ("A has identity?: {}".format(not (A[0][0]==0 and A[24][24]==0 and A[8][8]==0)))
        neighbors = torch.sum(A, 1)
        neighborAverageT = torch.div(torch.matmul(A, T.reshape(-1)), neighbors)
        return neighborAverageT
        
        

    def train_one_step_discriminator(self,A,X,T):

        self.model.train()
        self.optimizerD.zero_grad()
        pred_treatmentTrain,_, _, _,_ = self.model(A,X,T)
        discLoss = self.bce_loss(pred_treatmentTrain.reshape(-1),T)
        num = pred_treatmentTrain.shape[0]
        target05 = [0.5 for _ in range(num)]
        discLosshalf = self.loss(pred_treatmentTrain.reshape(-1), self.Tensor(target05))
        discLoss.backward()
        self.optimizerD.step()

        return discLoss,discLosshalf


    def eval_one_step_discriminator(self,A,X,T):

        self.model.eval()
        pred_treatment,_,_,_,_ = self.model(A,X,T)
        discLossWatch = self.bce_loss(pred_treatment.reshape(-1), T)
        num = pred_treatment.shape[0]
        target05 = [0.5 for _ in range(num)]
        discLosshalf = self.loss(pred_treatment.reshape(-1), self.Tensor(target05))

        return discLossWatch,discLosshalf,pred_treatment,T
        

    def train_discriminator(self,epoch):
        
        for ds in range(self.args['dstep']):
            discLoss,discLossTrainhalf = self.train_one_step_discriminator(self.trainA, self.trainX, self.trainT)
            if ds == self.args['dstep']-1:
                if self.args['printDisc']:
                    print('d_Epoch: {:04d}'.format(epoch + 1),
                        'dLoss:{:05f}'.format(discLoss),
                        'dLoss0.5:{:05f}'.format(discLossTrainhalf),
                        )
                self.dissTrain.append(discLoss.detach().cpu().numpy())
                self.dissTrainHalf.append(discLossTrainhalf.detach().cpu().numpy())


    def train_one_step_discriminator_z(self,A,X,T):

        self.model.train()
        self.optimizerD_z.zero_grad()
        _,pred_zTrain,_, _,labelZ = self.model(A,X,T)
        discLoss_z = self.d_zLoss(pred_zTrain.reshape(-1),labelZ)
        num = pred_zTrain.shape[0]
        target = self.Tensor(np.random.uniform(low=0.0, high=1.0, size=num))
        discLosshalf_z = self.loss(pred_zTrain.reshape(-1), self.Tensor(target))
        discLoss_z.backward()
        self.optimizerD_z.step()

        return discLoss_z,discLosshalf_z


    def eval_one_step_discriminator_z(self,A,X,T):

        self.model.eval()
        _,pred_z,_,_,labelZ = self.model(A,X,T)
        discLossWatch = self.d_zLoss(pred_z.reshape(-1), labelZ)
        num = pred_z.shape[0]
        target = self.Tensor(np.random.uniform(low=0.0, high=1.0, size=num))
        discLosshalf = self.loss(pred_z.reshape(-1), self.Tensor(target))

        return discLossWatch,discLosshalf,pred_z,labelZ


    def train_discriminator_z(self,epoch):
        
        for dzs in range(self.args['d_zstep']):
            discLoss_z,discLoss_zTrainRandom = self.train_one_step_discriminator_z(self.trainA, self.trainX, self.trainT)

            if dzs == self.args['d_zstep']-1:
                if self.args['printDisc_z']:
                    print('d_Epoch: {:04d}'.format(epoch + 1),
                        'd_zLoss:{:05f}'.format(discLoss_z),
                        'd_zLRanTrain:{:05f}'.format(discLoss_zTrainRandom),
                        )
                self.diss_zTrain.append(discLoss_z.detach().cpu().numpy())
                self.diss_zTrainHalf.append(discLoss_zTrainRandom.detach().cpu().numpy())


    def train_one_step_encoder_predictor(self,A,X,T,Y):
        
        if self.args['model'] == "NetEstimator":
            self.model.zero_grad()
            self.model.train()
            self.optimizerP.zero_grad()
            pred_treatmentTrain, pred_zTrain,pred_outcomeTrain,_,_ = self.model(A,X,T)
            pLoss = self.loss(pred_outcomeTrain.reshape(-1), Y)
            num = pred_treatmentTrain.shape[0]
            target05 = [0.5 for _ in range(num)]
            dLoss = self.loss(pred_treatmentTrain.reshape(-1), self.Tensor(target05))
            num = pred_zTrain.shape[0]
            target = self.Tensor(np.random.uniform(low=0.0, high=1.0, size=num))
            d_zLoss = self.d_zLoss(pred_zTrain.reshape(-1), target)
            loss_train = pLoss + dLoss*self.alpha + d_zLoss*self.gamma
            loss_train.backward()
            self.optimizerP.step()
            
        elif self.args['model'] == 'TNet':
            # self.model.zero_grad()
            self.model.train()
            self.optimizerT.zero_grad()
            g_T_hat, g_Z_hat, Q_hat, epsilon, embeddings, neighborAverageT = self.model(A, X, T)
            Q_Loss = self.loss(Q_hat.reshape(-1), Y)
            g_T_Loss = self.bce_loss(g_T_hat.reshape(-1), T)
            g_Z_Loss = - torch.log(g_Z_hat + 1e-6).mean()
            Loss_base = Q_Loss + self.args['alpha_tr'] * g_T_Loss + self.args['gamma_tr'] * g_Z_Loss
            Loss_TR = self.loss(
                Q_hat.reshape(-1) +
                epsilon.reshape(-1) * (1 / (g_T_hat.detach().reshape(-1) * g_Z_hat.detach().reshape(-1) + 1e-6) )
                        #  - (1 - T) / (  (1 - g_T_hat.reshape(-1)) * g_Z_hat.reshape(-1) + 1e-6))
                ,Y)

            loss_train = Loss_base + self.args['beta_tr'] * Loss_TR
            loss_train.backward()
            self.optimizerT.step()
            pLoss, dLoss, d_zLoss = Q_Loss, g_T_Loss, g_Z_Loss

        else:
            self.model.zero_grad()
            self.model.train()
            self.optimizerB.zero_grad()
            _, _,pred_outcomeTrain,rep,_ = self.model(A,X,T, self.args.get('motifs', None))
            pLoss = self.loss(pred_outcomeTrain.reshape(-1), Y)
            if self.args['model'] in set(["TARNet","TARNet_INTERFERENCE", "TARNet_MOTIFS"]):
                loss_train = pLoss
                dLoss = self.Tensor([0])
            else:
                rep_t1, rep_t0 = rep[(T > 0).nonzero()], rep[(T < 1).nonzero()]
                dLoss,_= utils.wasserstein(rep_t1, rep_t0, cuda=self.args['cuda'])
                loss_train = pLoss+self.alpha_base*dLoss
            d_zLoss = self.Tensor([-1])
            loss_train.backward()
            self.optimizerB.step()


        return loss_train,pLoss,dLoss,d_zLoss
        
        
    def eval_one_step_encoder_predictor(self,A,X,T,Y):

        self.model.eval()
        if self.args['model'] == "NetEstimator":
            pred_treatment, pred_z,pred_outcome,_,_ = self.model(A,X,T)
            pLossV = self.loss(pred_outcome.reshape(-1), Y)
            num = pred_treatment.shape[0]
            target05 = [0.5 for _ in range(num)]
            dLossV = self.loss(pred_treatment.reshape(-1), self.Tensor(target05))
            num = pred_z.shape[0]
            target = self.Tensor(np.random.uniform(low=0.0, high=1.0, size=num))
            d_zLossV = self.d_zLoss(pred_z.reshape(-1), target)
            loss_val = pLossV+dLossV*self.alpha+d_zLossV*self.gamma

        else:
            _, _,pred_outcome,rep,_ = self.model(A,X,T, self.args.get('motifs', None))
            pLossV = self.loss(pred_outcome.reshape(-1), Y)
            if self.args['model'] in set(["TARNet","TARNet_INTERFERENCE", "TARNet_MOTIFS"]):
                loss_val = pLossV
                dLossV = self.Tensor([0])
            else:
                rep_t1, rep_t0 = rep[(T > 0).nonzero()], rep[(T < 1).nonzero()]
                dLossV,_ = utils.wasserstein(rep_t1, rep_t0, cuda=self.args['cuda'])
                loss_val = self.args['alpha_base']*dLossV
                loss_val = pLossV+self.args['alpha_base']*dLossV
            d_zLossV = self.Tensor([-1])

        return loss_val,pLossV,dLossV,d_zLossV


    
    def compute_effect_pehe(self,A,X,gt_t1z1,gt_t1z0,gt_t0z7,gt_t0z2,gt_t0z0):
            
        num = X.shape[0]
        z_1s = self.Tensor(np.ones(num))
        z_0s = self.Tensor(np.zeros(num))
        z_07s = self.Tensor(np.zeros(num)+self.z_1)
        z_02s = self.Tensor(np.zeros(num)+self.z_2)
        t_1s = self.Tensor(np.ones(num))
        t_0s = self.Tensor(np.zeros(num))

        _, _,pred_outcome_t1z1,_,_ = self.model(A,X,t_1s,z_1s)
        _, _,pred_outcome_t1z0,_,_ = self.model(A,X,t_1s,z_0s)
        _, _,pred_outcome_t0z0,_,_ = self.model(A,X,t_0s,z_0s)
        _, _,pred_outcome_t0z7,_,_ = self.model(A,X,t_0s,z_07s)
        _, _,pred_outcome_t0z2,_,_ = self.model(A,X,t_0s,z_02s)

        pred_outcome_t1z1 = utils.PO_normalize_recover(self.args['normy'],self.POTrain,pred_outcome_t1z1)
        pred_outcome_t1z0 = utils.PO_normalize_recover(self.args['normy'],self.POTrain,pred_outcome_t1z0)
        pred_outcome_t0z0 = utils.PO_normalize_recover(self.args['normy'],self.POTrain,pred_outcome_t0z0)
        pred_outcome_t0z7 = utils.PO_normalize_recover(self.args['normy'],self.POTrain,pred_outcome_t0z7)
        pred_outcome_t0z2 = utils.PO_normalize_recover(self.args['normy'],self.POTrain,pred_outcome_t0z2)

        individual_effect = self.get_peheLoss(pred_outcome_t1z0,pred_outcome_t0z0,gt_t1z0,gt_t0z0)
        peer_effect = self.get_peheLoss(pred_outcome_t0z7,pred_outcome_t0z2,gt_t0z7,gt_t0z2)
        total_effect = self.get_peheLoss(pred_outcome_t1z1,pred_outcome_t0z0,gt_t1z1,gt_t0z0)

        return individual_effect,peer_effect,total_effect
    
    def compute_individual_effect(self,A,X,T):
            
        num = X.shape[0]
        print(self.args['model'])
        if self.args['isolated']:
            z_0s = self.Tensor(np.zeros(num))
        else:
            z_0s = self.args.get('motifs', None)
            if z_0s is None: # Use observed or assigned neighbor treatments
                deg = torch.sum(A, 1)
                z_0s = torch.div(torch.matmul(A, T.reshape(-1)), deg).view(-1,1)
        t_1s = self.Tensor(np.ones(num))
        t_0s = self.Tensor(np.zeros(num))
        with torch.no_grad():
            self.model.eval()
            if self.args['model'] == 'TNet':
                z_0s = z_0s.reshape(-1)
                pred_outcome_t1z0 = self.model.infer_potential_outcome(A,X,t_1s,z_0s)
                pred_outcome_t0z0 = self.model.infer_potential_outcome(A,X,t_0s,z_0s)                
                # g_T_hat1, g_Z_hat1, Q_hat1, epsilon1, _, _ = self.model(A, X, t_1s,z_0s)
                # g_T_hat0, g_Z_hat0, Q_hat0, epsilon0, _, _ = self.model(A, X, t_0s,z_0s)
                # pred_outcome_t1z0 = (Q_hat1.reshape(-1) + epsilon1.reshape(-1) * (1 / (g_T_hat1.reshape(-1) * g_Z_hat1.reshape(-1) + 1e-6)))
                # pred_outcome_t0z0 = (Q_hat0.reshape(-1) + epsilon0.reshape(-1) * (1 / (g_T_hat0.reshape(-1) * g_Z_hat0.reshape(-1) + 1e-6)))
            else:
                _, _,pred_outcome_t1z0,_,_ = self.model(A,X,t_1s,z_0s)
                _, _,pred_outcome_t0z0,_,_ = self.model(A,X,t_0s,z_0s)

        pred_outcome_t1z0 = utils.PO_normalize_recover(self.args['normy'],self.POTrain,pred_outcome_t1z0)
        pred_outcome_t0z0 = utils.PO_normalize_recover(self.args['normy'],self.POTrain,pred_outcome_t0z0)

        return pred_outcome_t1z0, pred_outcome_t0z0


    
    def train_encoder_predictor(self,epoch):
        
        for _ in range(self.args['pstep']):
           loss_train,pLoss_train,dLoss_train,d_zLoss_train = self.train_one_step_encoder_predictor(self.trainA, self.trainX, self.trainT,self.YFTrain)

           self.lossTrain.append(loss_train.cpu().detach().numpy())
           
           # individual_effect_train,peer_effect_train,total_effect_train = self.compute_effect_pehe(self.trainA, self.trainX,self.train_t1z1,self.train_t1z0,self.train_t0z7,self.train_t0z2,self.train_t0z0)
        

        if self.args['printPred']:
            print('p_Epoch: {:04d}'.format(epoch + 1),
                'pLossTrain:{:.4f}'.format(pLoss_train.item()),
                'dLossTrain:{:.4f}'.format(dLoss_train.item()),
                'd_zLossTrain:{:.4f}'.format(d_zLoss_train.item()),

                # 'iE_train:{:.4f}'.format(individual_effect_train.item()),
                # 'PE_train:{:.4f}'.format(peer_effect_train.item()),
                # 'TE_train:{:.4f}'.format(total_effect_train.item()),
                )
            
    def train_fluctuation_param(self, epoch):
            for ds in range(self.args['iter_2step_tr']):
                # self.model.zero_grad()
                self.model.train()
                self.optimizer2step.zero_grad()
                A, X, T, Y = self.trainA, self.trainX, self.trainT, self.YFTrain

                g_T_hat, g_Z_hat, Q_hat, epsilon, embeddings, neighborAverageT = self.model(A, X, T)
                Loss_TR = self.loss(
                    Q_hat.reshape(-1) +
                    epsilon.reshape(-1) * (1 / (g_T_hat.reshape(-1).detach() * g_Z_hat.reshape(-1).detach() + 1e-6))
                    #  - (1 - T) / (  (1 - g_T_hat.reshape(-1)) * g_Z_hat.reshape(-1) + 1e-6))
                    , Y)
                loss_train = self.args['beta_tr'] * Loss_TR

                if self.args['loss_2step_with_ly'] == 1:
                    Q_Loss = self.loss(Q_hat.reshape(-1), Y)
                    loss_train = loss_train + Q_Loss

                if self.args['loss_2step_with_ltz'] == 1:
                    g_T_Loss = self.bce_loss(g_T_hat.reshape(-1), T)
                    g_Z_Loss = - torch.log(g_Z_hat + 1e-6).mean()
                    loss_train = loss_train + self.args['alpha_tr'] * g_T_Loss + self.args['gamma_tr'] * g_Z_Loss

                loss_train.backward()
                self.optimizer2step.step()

#                 g_T_hat_val, g_Z_hat_val, Q_hat_val, epsilon_val, _, _ = self.model(self.valA, self.valX,
#                  self.valT)

#                 pLoss_val = self.loss(
#                     Q_hat_val.reshape(-1) +
#                     epsilon_val.reshape(-1) * (1 / (g_T_hat_val.reshape(-1).detach() * g_Z_hat_val.reshape(-1).detach() + 1e-6))
#                     #  - (1 - T) / (  (1 - g_T_hat.reshape(-1)) * g_Z_hat.reshape(-1) + 1e-6))
#                     , self.YFVal
#                 ) * self.args.beta



    def train(self):
        # print ("================================Training Start================================")

        if self.args['model'] == "NetEstimator":
            print ("******************NetEstimator******************")
            for epoch in range(self.args['epochs']):
                self.train_discriminator(epoch)
                self.train_discriminator_z(epoch)
                self.train_encoder_predictor(epoch)
                
        elif self.args['model'] == "TNet":
            print("******************" + str(self.args['model']) + "******************")
            print("******************" + "train 1 step for specified epochs" + "******************")
            for epoch in range(self.args['pre_train_step_tr']):
                # 1 step
                self.train_encoder_predictor(epoch)

            print("******************" + "train iteratively for specified epochs" + "******************")
            for epoch in range(self.args['epochs']):
                # iterative training step
                self.train_encoder_predictor(epoch)
                # if self.args.beta != 0:
                #     self.train_fluctuation_param(epoch)
                self.train_fluctuation_param(epoch)

        else:
            print ("******************Baselines******************")
            for epoch in range(self.args['epochs']):
                self.train_encoder_predictor(epoch)
    

    def one_step_predict(self,A,X,T,Y):
        self.model.eval()
        pred_treatment, _,pred_outcome,_,_ = self.model(A,X,T)
        pred_outcome = utils.PO_normalize_recover(self.args['normy'],self.POTrain,pred_outcome)
        Y = utils.PO_normalize_recover(self.args['normy'],self.POTrain,Y)
        pLoss = self.loss(pred_outcome.reshape(-1), Y)
        
        return pLoss,pred_outcome,Y
        

    def predict(self):
        print ("================================Predicting================================")
        factualLossTrain,pred_train,YFTrainO = self.one_step_predict(self.trainA,self.trainX,self.trainT,self.YFTrain)


        # individual_effect_train,peer_effect_train,total_effect_train = self.compute_effect_pehe(self.trainA, self.trainX,self.train_t1z1,self.train_t1z0,self.train_t0z7,self.train_t0z2,self.train_t0z0)        

        print('F_train:{:.4f}'.format(factualLossTrain.item()),

              # 'iE_train:{:.4f}'.format(individual_effect_train.item()),
              # 'PE_train:{:.4f}'.format(peer_effect_train.item()),
              # 'TE_train:{:.4f}'.format(total_effect_train.item()),
              )

        data = {
            "pred_train_factual":pred_train,
            "PO_train_factual":YFTrainO,

            "factualLossTrain":factualLossTrain.detach().cpu().numpy(),
            # "individual_effect_train":individual_effect_train.detach().cpu().numpy(),
            # "peer_effect_train":peer_effect_train.detach().cpu().numpy(),
            # "total_effect_train":total_effect_train.detach().cpu().numpy(),
        }

        
#         if self.args['model'] == "NetEstimator":
#             print ("================================Save prediction...================================")
#             folder = "../results/"+self.args['dataset']+"/perf/"
#             os.makedirs(folder, exist_ok=True)
#             file = folder+self.args['dataset']+"_prediction_expID_"+str(self.args['expID'])+"_alpha_"+str(self.args['alpha'])+"_gamma_"+str(self.args['gamma'])+"_flipRate_"+str(self.args['flipRate'])+".pkl"
#         else:
#             print ("================================Save Bseline prediction...================================")
#             folder = "../results/baselines/"+self.args['model']+"/"+self.args['dataset']+"/perf/"
#             os.makedirs(folder, exist_ok=True)
#             file = folder + "prediction_expID_"+str(self.args['expID'])+"_alpha_"+str(self.args['alpha'])+"_gamma_"+str(self.args['gamma'])+"_flipRate_"+str(self.args['flipRate'])+".pkl"
        
#         with open(file,"wb") as f:
#             pkl.dump(data,f)
#         print ("================================Save prediction done!================================")

    def save_curve(self):
        print ("================================Save curve...================================")
        data = {"dissTrain":self.dissTrain,
                "dissTrainHalf":self.dissTrainHalf,
                "diss_zTrain":self.diss_zTrain,
                "diss_zTrainHalf":self.diss_zTrainHalf,
               }

        with open("../results/"+str(self.args['dataset'])+"/curve/"+"curve_expID_"+str(self.args['expID'])+"_alpha_"+str(self.args['alpha'])+"_gamma_"+str(self.args['gamma'])+"_flipRate_"+str(self.args['flipRate'])+".pkl", "wb") as f:
            pkl.dump(data, f)
        print ("================================Save curve done!================================")


    def save_embedding(self):
        print ("================================Save embedding...================================")
        _, _,_, embedsTrain,_ = self.model(self.trainA, self.trainX, self.trainT)
        # _, _,_, embedsTest,_ = self.model(self.testA, self.testX, self.testT)
        data = {"embedsTrain": embedsTrain.cpu().detach().numpy(), 
                # "embedsTest": embedsTest.cpu().detach().numpy(),
                "trainT": self.trainT.cpu().detach().numpy(),
                # "testT": self.testT.cpu().detach().numpy(),
                "trainZ": self.trainZ.cpu().detach().numpy(),
                # "testZ": self.testZ.cpu().detach().numpy()
               }
        with open("../results/"+str(self.args['dataset'])+"/embedding/"+"embeddings_expID_"+str(self.args['expID'])+"_alpha_"+str(self.args['alpha'])+"_gamma_"+str(self.args['gamma'])+"_flipRate_"+str(self.args['flipRate'])+".pkl", "wb") as f:
            pkl.dump(data, f)
        print ("================================Save embedding done!================================")


