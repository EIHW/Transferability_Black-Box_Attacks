import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import logging
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_generator import DataGenerator, TestDataGenerator
from utilities import (create_folder, get_filename, create_logging,
                       calculate_confusion_matrix, calculate_accuracy, 
                       plot_confusion_matrix, print_accuracy, 
                       write_leaderboard_submission, write_evaluation_submission)
from models_pytorch import move_data_to_gpu, DecisionLevelMaxPooling, CnnPooling_Max, ResNet, Vggish, AlexNet, CnnAtrous, EWC
import config
from torch.autograd import Variable

Model_attacker = CnnAtrous
#Model_target_list = [ResNet, Vggish]
#Model_target_list = [DecisionLevelMaxPooling, Vggish]
Model_target_list = [DecisionLevelMaxPooling, Vggish, ResNet]
Model_attacker_folder = 'CnnAtrous' 
#Model_target_folder_list = ['models-resnet50', 'models-vgg16']
#Model_target_folder_list = ['models-basecnn', 'models-vgg16']
Model_target_folder_list = ['models-basecnn', 'models-vgg16', 'models-resnet50']
Model_num = 3

batch_size = 16


def evaluate(model_attacker, model_target, generator, data_type, devices, max_iteration, cuda):
    """Evaluate
    
    Args:
      model: object.
      generator: object.
      data_type: 'train' | 'validate'.
      devices: list of devices, e.g. ['a'] | ['a', 'b', 'c']
      max_iteration: int, maximum iteration for validation
      cuda: bool.
      
    Returns:
      accuracy: float
    """
    
    # Generate function
    generate_func = generator.generate_validate(data_type=data_type, 
                                                devices=devices, 
                                                shuffle=True, 
                                                max_iteration=max_iteration)
            
    # Forward
    dict = forward(model_attacker=model_attacker,
		   model_target=model_target,
                   generate_func=generate_func, 
                   cuda=cuda, 
                   return_target=True)

    outputs = dict['output']    # (audios_num, classes_num)
    outputs_adv = dict['output_adv']    # (audios_num, classes_num)
    targets = dict['target']    # (audios_num, classes_num)
    
    predictions = np.argmax(outputs, axis=-1)   # (audios_num,)
    predictions_adv = np.argmax(outputs_adv, axis=-1)   # (audios_num,)

    # Evaluate
    classes_num = outputs.shape[-1]

    loss = F.nll_loss(Variable(torch.Tensor(outputs)), Variable(torch.LongTensor(targets))).data.numpy()
    loss = float(loss)

    loss_adv = F.nll_loss(Variable(torch.Tensor(outputs_adv)), Variable(torch.LongTensor(targets))).data.numpy()
    loss_adv = float(loss_adv)
 
    accuracy = calculate_accuracy(targets, predictions, classes_num, 
                                  average='macro')

    accuracy_adv = calculate_accuracy(targets, predictions_adv, classes_num, 
                                  average='macro')

    return accuracy, loss, accuracy_adv, loss_adv

# forward: model_pytorch---        Return_heatmap = False
# forward_heatmap: model_pytorch---        Return_heatmap = True
def forward(model_attacker, model_target, generate_func, cuda, return_target):
    """Forward data to a model.
    
    Args:
      generate_func: generate function
      cuda: bool
      return_target: bool
      
    Returns:
      dict, keys: 'audio_name', 'output'; optional keys: 'target'
    """
    
    outputs = []
    audio_names = []
    outputs_adv = []
    
    if return_target:
        targets = []
    
    # Evaluate on mini-batch
    for data in generate_func:
            
        if return_target:
            (batch_x, batch_y, batch_audio_names) = data
            
        else:
            (batch_x, batch_audio_names) = data
            
        batch_x = move_data_to_gpu(batch_x, cuda)

        # Predict
        model_attacker.eval()
	model_target.eval()
	batch_output = model_target(batch_x)	

	# advesarial predict
	batch_x_adv = batch_x + model_attacker(batch_x)
	batch_output_adv = model_target(batch_x_adv)	

        # Append data
        outputs.append(batch_output.data.cpu().numpy())
        audio_names.append(batch_audio_names)
        outputs_adv.append(batch_output_adv.data.cpu().numpy())        

        if return_target:
            targets.append(batch_y)

    dict = {}

    outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs
    
    audio_names = np.concatenate(audio_names, axis=0)
    dict['audio_name'] = audio_names

    outputs_adv = np.concatenate(outputs_adv, axis=0)
    dict['output_adv'] = outputs_adv
    
    if return_target:
        targets = np.concatenate(targets, axis=0)
        dict['target'] = targets
        
    return dict


def forward_test(model_attacker, model_target, generate_func, cuda, return_target):
    """Forward data to a model.
    
    Args:
      generate_func: generate function
      cuda: bool
      return_target: bool
      
    Returns:
      dict, keys: 'audio_name', 'output'; optional keys: 'target'
    """
    
    outputs = []
    audio_names = []
    outputs_adv = []
    mseloss = 0
    outputs_heatmap = [] ###############################
    inputs_heatmap = [] ###############################
    noise_heatmap = []
    
    if return_target:
        targets = []
    
    # Evaluate on mini-batch
    for data in generate_func:
            
        if return_target:
            (batch_x, batch_y, batch_audio_names) = data
            
        else:
            (batch_x, batch_audio_names) = data
            
        batch_x = move_data_to_gpu(batch_x, cuda)

        # Predict
        model_attacker.eval()
	model_target.eval()
        batch_output = model_target(batch_x)
	
	# advesarial predict
        batch_noise = model_attacker(batch_x)
	batch_x_adv = batch_x + batch_noise
        batch_output_adv = model_target(batch_x_adv)

	mseloss = mseloss + F.mse_loss(batch_x_adv.data.cpu(), batch_x.data.cpu()).data.numpy()
        mseloss = float(mseloss)

        # Append data
        outputs.append(batch_output.data.cpu().numpy())
        audio_names.append(batch_audio_names)
        outputs_adv.append(batch_output_adv.data.cpu().numpy())
  
        noise_heatmap.append(batch_noise.data.cpu().numpy())  
        inputs_heatmap.append(batch_x.data.cpu().numpy())####################################### 
        outputs_heatmap.append(batch_x_adv.data.cpu().numpy())#######################################

        if return_target:
            targets.append(batch_y)

    dict = {}

    outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs
    
    audio_names = np.concatenate(audio_names, axis=0)
    dict['audio_name'] = audio_names

    outputs_adv = np.concatenate(outputs_adv, axis=0)
    dict['output_adv'] = outputs_adv

    if return_target:
        targets = np.concatenate(targets, axis=0)
        dict['target'] = targets

    inputs_heatmap = np.concatenate(inputs_heatmap, axis=0)###########################################
    dict['inputs_heatmap'] = inputs_heatmap
    outputs_heatmap = np.concatenate(outputs_heatmap, axis=0)###########################################
    dict['outputs_heatmap'] = outputs_heatmap
    noise_heatmap = np.concatenate(noise_heatmap, axis=0)###########################################
    dict['noise_heatmap'] = noise_heatmap

    dict['mseloss'] = mseloss/len(audio_names)

    return dict


def train(args):

    # Arugments & parameters
    dataset_dir = args.dataset_dir
    subdir = args.subdir
    workspace = args.workspace
    feature_type = args.feature_type
    filename = args.filename
    validation = args.validation
    holdout_fold = args.holdout_fold
    mini_data = args.mini_data
    cuda = args.cuda

    labels = config.labels

    if 'mobile' in subdir:
        devices = ['a', 'b', 'c']
    else:
        devices = ['a']

    classes_num = len(labels)

    # Paths
    if mini_data:
        hdf5_path = os.path.join(workspace, 'features', feature_type, 'mini_development.h5')
    else:
        hdf5_path = os.path.join(workspace, 'features', feature_type, 'development.h5')

    if validation:
        
        dev_train_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                     'fold{}_train.txt'.format(holdout_fold))
                                    
        dev_validate_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                    'fold{}_devel.txt'.format(holdout_fold))
                              
        models_attacker_dir = os.path.join(workspace, 'models', Model_attacker_folder, subdir, filename,
                                  'holdout_fold={}'.format(holdout_fold))

        models_target_path = [os.path.join(workspace, 'models', Model_target_folder_list[0], subdir, filename,
                                'holdout_fold={}'.format(holdout_fold), 'md_10000_iters.tar'),
			      os.path.join(workspace, 'models', Model_target_folder_list[1], subdir, filename,
                                'holdout_fold={}'.format(holdout_fold), 'md_10000_iters.tar'),
			      os.path.join(workspace, 'models', Model_target_folder_list[2], subdir, filename,
                                'holdout_fold={}'.format(holdout_fold), 'md_10000_iters.tar')]

                                        
    else:
        dev_train_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                     'fold{}_traindevel.txt'.format(holdout_fold))

        dev_validate_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                        'fold{}_test.txt'.format(holdout_fold))
        
        models_attacker_dir = os.path.join(workspace, 'models', Model_attacker_folder, subdir, filename,
                                  'full_train')

        models_target_path = [os.path.join(workspace, 'models', Model_target_folder_list[0], subdir, filename,
                                'full_train', 'md_10000_iters.tar'),
			      os.path.join(workspace, 'models', Model_target_folder_list[1], subdir, filename,
                                'full_train', 'md_10000_iters.tar'),
			      os.path.join(workspace, 'models', Model_target_folder_list[2], subdir, filename,
                                'full_train', 'md_10000_iters.tar')]


    create_folder(models_attacker_dir)

    # Model
    model_attacker = Model_attacker()
    model_target = []
    for i in range(0, Model_num):
        model = Model_target_list[i](classes_num)
        checkpoint = torch.load(models_target_path[i])
        model.load_state_dict(checkpoint['state_dict'])
	model_target.append(model)
        del checkpoint
        del model
		
    if cuda:
        model_attacker.cuda()
	model_target[0].cuda()
	model_target[1].cuda()
	model_target[2].cuda()
	
    # Data generator
    generator = DataGenerator(hdf5_path=hdf5_path,
                                  batch_size=batch_size,
                                  dev_train_csv=dev_train_csv,
                                  dev_validate_csv=dev_validate_csv)

        # Optimizer
    lr = 1e-3
    optimizer_attacker = optim.Adam(model_attacker.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.)

    train_bgn_time = time.time()

    # Train on mini batches
    for (iteration, (batch_x, batch_y)) in enumerate(generator.generate_train()):

        # Evaluate
        if iteration % 100 == 0:

            train_fin_time = time.time()

            (tr_acc, tr_loss, tr_acc_adv, tr_loss_adv) = evaluate(model_attacker=model_attacker,
					     model_target=model_target[1],
                                             generator=generator,
                                             data_type='train',
                                             devices=devices,
                                             max_iteration=None,
                                             cuda=cuda)

            logging.info('tr_acc: {:.3f}, tr_loss: {:.3f}, tr_acc_adv: {:.3f}, tr_loss_adv: {:.3f}'.format(
                     tr_acc, tr_loss, tr_acc_adv, tr_loss_adv))

            (va_acc, va_loss, va_acc_adv, va_loss_adv) = evaluate(model_attacker=model_attacker,
					    model_target=model_target[1],
                                            generator=generator,
                                            data_type='validate',
                                            devices=devices,
                                            max_iteration=None,
                                            cuda=cuda)

            logging.info('va_acc: {:.3f}, va_loss: {:.3f}, va_acc_adv: {:.3f}, va_loss_adv: {:.3f}'.format(
                        va_acc, va_loss, va_acc_adv, va_loss_adv))

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info('iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'
                    ''.format(iteration, train_time, validate_time))

            logging.info('------------------------------------')

	train_bgn_time = time.time()

        if iteration % 10000 == 0 and iteration > 0:
            save_out_dict = {'iteration': iteration,
                                 'state_dict': model_attacker.state_dict(),
                                 'optimizer': optimizer_attacker.state_dict()
                                 }
            save_out_path = os.path.join(models_attacker_dir, 'md_{}_iters.tar'.format(10000))
            torch.save(save_out_dict, save_out_path)
            logging.info('Model saved to {}'.format(save_out_path))
            
        # Reduce learning rate
        if iteration % 1000 == 0 > 0:
            for param_group in optimizer_attacker.param_groups:
                param_group['lr'] *= 0.9

        # Train
        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)

	# attacker -- generator
        optimizer_attacker.zero_grad()
	model_attacker.train()
	perturbation = model_attacker(batch_x)
	batch_x_adv = batch_x + perturbation  
        loss_a_rec = F.mse_loss(batch_x_adv, batch_x) 

	loss_a_cla = 0
	for i in range(0, Model_num):
  	    model_target[i].eval()
	    batch_output_adv = model_target[i](batch_x_adv)

	    # C&W loss
            onehot_labels = torch.eye(classes_num)
	    onehot_labels = move_data_to_gpu(onehot_labels, cuda)
	    onehot_labels = onehot_labels[batch_y]

            prob_real = torch.sum(onehot_labels * batch_output_adv, dim=1)
            prob_other, _ = torch.max((1 - onehot_labels) * batch_output_adv - onehot_labels * 10000, dim=1)
            zeros = torch.zeros_like(prob_other)
            loss_a_cla_i = torch.max(prob_real - prob_other, zeros)
            loss_a_cla_i = torch.sum(loss_a_cla_i)
	        
	    loss_a_cla = loss_a_cla + loss_a_cla_i

	loss_a = 0.02/3.0 *loss_a_cla + 0.98*loss_a_rec
	if iteration % 100 == 0 and iteration > 0:
	    print(str(loss_a_cla) + '\t' + str(loss_a_rec))
	

        # Backward
        loss_a.backward()
        optimizer_attacker.step()

        # Stop learning
	if iteration == 10000:
	    break


def inference_validation_data(args):
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    subdir = args.subdir
    workspace = args.workspace
    feature_type = args.feature_type
    holdout_fold = args.holdout_fold
    iteration = args.iteration
    filename = args.filename
    cuda = args.cuda
    validation = args.validation

    labels = config.labels

    if 'mobile' in subdir:
        devices = ['a', 'b', 'c']
    else:
        devices = ['a']

    classes_num = len(labels)

    # Paths
    #hdf5_path = os.path.join(workspace, 'features', subdir, '{}.h5'.format(feature_type))
    hdf5_path = os.path.join(workspace, 'features', feature_type, 'development.h5')

    Model_target_list = [DecisionLevelMaxPooling, Vggish, ResNet]
    Model_target_folder_list = ['models-basecnn', 'models-vgg16', 'models-resnet50']
    Model_num = 3

    for target_num in range(0, Model_num):
        Model_target = Model_target_list[target_num]
        Model_target_folder = Model_target_folder_list[target_num]
        logging.info('target_num: {}'.format(target_num))

        if validation:

            dev_train_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup', 'fold{}_train.txt'.format(holdout_fold))
                                 
            dev_validate_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup', 'fold{}_devel.txt'.format(holdout_fold))

            model_target_path = os.path.join(workspace, 'models', Model_target_folder, subdir, filename,
                                'holdout_fold={}'.format(holdout_fold), 'md_10000_iters.tar')

            model_attacker_path = os.path.join(workspace, 'models', Model_attacker_folder, subdir, filename,
                                'holdout_fold={}'.format(holdout_fold), 
                                'md_{}_iters.tar'.format(iteration))

        else:

            dev_train_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup', 'fold{}_traindevel.txt'.format(holdout_fold))

            dev_validate_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup', 'fold{}_test.txt'.format(holdout_fold))

            model_target_path = os.path.join(workspace, 'models', Model_target_folder, subdir, filename,
                                'full_train', 'md_10000_iters.tar')

            model_attacker_path = os.path.join(workspace, 'models', Model_attacker_folder, subdir, filename,
                                  'full_train', 
                                  'md_{}_iters.tar'.format(iteration))

        # Load model
        model_attacker = Model_attacker()
        model_target = Model_target(classes_num)

    	checkpoint = torch.load(model_attacker_path)
        model_attacker.load_state_dict(checkpoint['state_dict'])
        del checkpoint
        checkpoint = torch.load(model_target_path)
        model_target.load_state_dict(checkpoint['state_dict'])
        del checkpoint

        if cuda:
	    model_attacker.cuda()
            model_target.cuda()

        # Predict & evaluate
        for device in devices:

            print('Device: {}'.format(device))

            # Data generator
            generator = DataGenerator(hdf5_path=hdf5_path,
                                  batch_size=batch_size,
                                  dev_train_csv=dev_train_csv,
                                  dev_validate_csv=dev_validate_csv)

            generate_func = generator.generate_validate(data_type='validate', 
                                                     devices=device, 
                                                     shuffle=False)

            # Inference
            dict = forward_test(model_attacker=model_attacker,
		       model_target=model_target,
                       generate_func=generate_func, 
                       cuda=cuda, 
                       return_target=True)

            outputs = dict['output']    # (audios_num, classes_num)
            targets = dict['target']    # (audios_num, classes_num)
  	    outputs_adv = dict['output_adv']    # (audios_num, classes_num)
	    inputs_heatmap = dict['inputs_heatmap']
  	    outputs_heatmap = dict['outputs_heatmap']
	    audio_names = dict['audio_name']

            predictions = np.argmax(outputs, axis=-1)
            predictions_adv = np.argmax(outputs_adv, axis=-1)

            classes_num = outputs.shape[-1]      

            # Evaluate
            confusion_matrix = calculate_confusion_matrix(targets, predictions, classes_num)
            
            class_wise_accuracy = calculate_accuracy(targets, predictions, classes_num)

            confusion_matrix_adv = calculate_confusion_matrix(targets, predictions_adv, classes_num)
            
            class_wise_accuracy_adv = calculate_accuracy(targets, predictions_adv, classes_num)

            # Print
            print_accuracy(class_wise_accuracy, labels)
            print('confusion_matrix: \n', confusion_matrix)

            print_accuracy(class_wise_accuracy_adv, labels)
            print('confusion_matrix: \n', confusion_matrix_adv)

	    mseloss = dict['mseloss']
            print('mseloss: ', mseloss)

	##########################################################################
            heatmaps_input = []
	    heatmaps_output = []
            classes = []
            audio_name_tosave = []
	    probs = []
            for i in range(0, len(predictions)):
            	pred_num = predictions[i]
            	pred_num_adv = predictions_adv[i]
            	if not pred_num == pred_num_adv:
            	    classes.append([pred_num, pred_num_adv])
	    	    #probs.append(outputs[i][pred_num])
            	    heatmaps_input.append(inputs_heatmap[i])
            	    heatmaps_output.append(outputs_heatmap[i])
            	    audio_name_tosave.append(audio_names[i])

	    #print 'final heatmaps number: ' + str(len(heatmaps))  

            # save
	    if validation:
	    	file_name = 'devel'
	    else:
	    	file_name = 'test'	

	    folder_name = 'heatmap' #####output path folder#####
	    if not os.path.exists(os.path.join(workspace, 'models', folder_name)):
	    	create_folder(os.path.join(workspace, 'models', folder_name))

            np.save(os.path.join(workspace, 'models', folder_name, "target_num"+str(target_num)+"-"+file_name+"-heatmap.npy"),heatmaps_input)##############
            np.save(os.path.join(workspace, 'models', folder_name, "target_num"+str(target_num)+"-"+file_name+"-heatmap_adv.npy"),heatmaps_output)##############
            np.save(os.path.join(workspace, 'models', folder_name, "target_num"+str(target_num)+"-"+file_name+"-audioName.npy"), audio_name_tosave)#########                 
	###########################################################################    

        	# Plot confusion matrix
#        	plot_confusion_matrix(
#            	confusion_matrix,
#            	title='Device {}'.format(device.upper()), 
#           	labels=labels,
#           	values=class_wise_accuracy,
#           	path=os.path.join(workspace, 'logs', 'main_pytorch', 'fig-confmat-device-'+device+'.pdf'))



if __name__ == '__main__':
    '''
    ######################################################################
    DATASET_DIR = "/home/zhao/NAS/data_work/Zhao/wav_DEMoS/zhao_code"
    WORKSPACE = "/home/zhao/NAS/data_work/Zhao/wav_DEMoS/zhao_code/pub_demos_cnn"
    DEV_SUBTASK_A_DIR = "demos_data"

    parser_train = argparse.ArgumentParser(description='Example of parser. ')

    parser_train.add_argument('--mode', type=str, default='train')
    parser_train.add_argument('--dataset_dir', type=str, default=DATASET_DIR)
    parser_train.add_argument('--subdir', type=str, default=DEV_SUBTASK_A_DIR)
    parser_train.add_argument('--workspace', type=str, default=WORKSPACE)
    parser_train.add_argument('--feature_type', type=str, default='logmel')
#    parser_train.add_argument('--iteration', type=str, default=2800)
    parser_train.add_argument('--holdout_fold', type=str, default=1)
    parser_train.add_argument('--validation', action='store_true', default=True)
    parser_train.add_argument('--cuda', action='store_true', default=True)
    parser_train.add_argument('--mini_data', action='store_true', default=False) # what is this?

    args = parser_train.parse_args()

    args.filename = get_filename(__file__)

    # Create log
    logs_dir = os.path.join(args.workspace, 'logs', args.filename)
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference_validation':
        inference_validation(args)
    #######################################################################

    '''
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataset_dir', type=str, required=True)
    parser_train.add_argument('--subdir', type=str, required=True)
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--feature_type', type=str, default='logmel')
    parser_train.add_argument('--validation', action='store_true', default=False)
    parser_train.add_argument('--holdout_fold', type=int)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--mini_data', action='store_true', default=False)

    
    parser_inference_validation_data = subparsers.add_parser('inference_validation_data')
    parser_inference_validation_data.add_argument('--dataset_dir', type=str, required=True)
    parser_inference_validation_data.add_argument('--subdir', type=str, required=True)
    parser_inference_validation_data.add_argument('--workspace', type=str, required=True)
    parser_inference_validation_data.add_argument('--feature_type', type=str, default='logmel')
    parser_inference_validation_data.add_argument('--validation', action='store_true', default=False)
    parser_inference_validation_data.add_argument('--holdout_fold', type=int, required=True)
    parser_inference_validation_data.add_argument('--iteration', type=int, required=True)
    parser_inference_validation_data.add_argument('--cuda', action='store_true', default=False)


    parser_inference_validation_heatmap = subparsers.add_parser('inference_validation_heatmap')
    parser_inference_validation_heatmap.add_argument('--dataset_dir', type=str, required=True)
    parser_inference_validation_heatmap.add_argument('--subdir', type=str, required=True)
    parser_inference_validation_heatmap.add_argument('--workspace', type=str, required=True)
    parser_inference_validation_heatmap.add_argument('--feature_type', type=str, default='logmel')
    parser_inference_validation_heatmap.add_argument('--validation', action='store_true', default=False)
    parser_inference_validation_heatmap.add_argument('--holdout_fold', type=int, required=True)
    parser_inference_validation_heatmap.add_argument('--iteration', type=int, required=True)
    parser_inference_validation_heatmap.add_argument('--cuda', action='store_true', default=False)


    args = parser.parse_args()

    args.filename = get_filename(__file__)

    # Create log
    logs_dir = os.path.join(args.workspace, 'logs', args.filename)
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    if args.mode == 'train':
        train(args)

    elif args.mode == 'inference_validation_data':
        inference_validation_data(args)

    elif args.mode == 'inference_validation_heatmap':
        inference_validation_heatmap(args)

    else:
        raise Exception('Error argument!')

