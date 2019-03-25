
from argparse import ArgumentParser


# start_near_true_label = False ## global 

def get_args():
  parser = ArgumentParser(description='Structure Prediction Energy Network, designed for mimic3 data')

  parser.add_argument('--do_gcnn', action='store_true',
            help='run graph cnn') 
  parser.add_argument('--init_label_emb', action='store_true',
            help='use some random init for label emb')
  parser.add_argument('--do_label_label', action='store_true', 
            help='do_label_label energy')
  parser.add_argument('--do_children_pd', action='store_true',
            help='use children of terms')
  parser.add_argument('--lr_optim_y', type=float, default=0.0001,
            help='learning rate')
  parser.add_argument('--lr_predict_y', type=float, default=0.0001,
            help='learning rate')
  parser.add_argument('--lr', type=float, default=0.0001,
            help='learning rate') 
  parser.add_argument('--lr_lagran', type=float, default=0.1,
            help='learning rate')
  parser.add_argument('--result_folder', type=str, default=None,
						help='where to save result')
  parser.add_argument('--data_type', type=str, default=None,
						help='what data to use (yeast / BP / CC / MF)')
  parser.add_argument('--batch_size', type=int, default=16,
            help='batch size for entailment model')
  parser.add_argument('--gcnn_dim', type=int, default=300,
            help='emb dim for the gcnn vec') 
  parser.add_argument('--epoch', type=int, default=200,
            help='how many epoch in total')
  parser.add_argument('--load_gcnn_emb', action='store_true',
            help='load pre-trained gcnn emb')
  parser.add_argument('--epoch_y', type=int, default=25,
            help='we have to optim y, how many iter ??') 
  parser.add_argument('--epoch_predict_y', type=int, default=25,
            help='we have to predict y, how many iter ??') 
  parser.add_argument('--model_load', type=str, default=None,
            help='path to a model to load')
  parser.add_argument('--model_cnn', type=str, default=None,
            help='path to a gcnn or cnn model to load')
  parser.add_argument('--gcnn_label_path', type=str, default=None,
            help='folder where the gcnn labels are')
  parser.add_argument('--label_layer_dim', type=int, default=50,
            help='label_layer_dim because we do not directly measure label-label energy based on 1-hot')
  parser.add_argument('--add_name', type=str, default=None,
            help='are we using subset like _5icd ?') 
  parser.add_argument('--w2v_emb', type=str, default=None,
            help='path to a w2v emb') 
  parser.add_argument('--not_train_w2v_emb', action='store_true',
            help='do not train word emb') 
  parser.add_argument('--filter_type', type=str, default=None,
            help='how do we handel word indexing ?? remove unseen word ?? ') 
  parser.add_argument('--test_path', type=str, default=None,
            help='where do get the test data, this can be different from train path ?? ')
  parser.add_argument('--label_encoded_dim', type=int, default=300,
            help='what is the dim for the ave. of word emb in labels ') 
  parser.add_argument('--num_of_filter', type=int, default=300,
            help='what dim do we project the seq into ?? ')
  parser.add_argument('--top_k', type=int, default=5,
            help='what dim do we project the seq into ?? ')
  parser.add_argument('--L2_scale', type=float, default=1,
            help='scale down L2 loss for label-label ?? ')
  parser.add_argument('--do_gcnn_only', action='store_true',
            help='do_gcnn_only no prediction') 
  parser.add_argument('--not_update_cnn_att', action='store_true',
            help='not_update_cnn_att params') 
  parser.add_argument('--do_test', action='store_true',
            help='run on test set') 
  parser.add_argument('--near_cnn', type=float, default=0,
            help='during optim for new sample, run best cnn 1st, then keep new prediction close') 
  parser.add_argument('--warm_start', action='store_true',
            help='use prediction of cnn as starting point when we do any optim') 
  parser.add_argument('--epoch_params', type=int, default=1, 
            help='what dim do we project the seq into ?? ') 
  parser.add_argument('--weight_term', type=float, default=1, 
            help='weight_term to scale the label-label energy ?? ')
  parser.add_argument('--do_constraint', action='store_true',
            help='run constraint optim')
  parser.add_argument('--brute_force', action='store_true',
            help='run brute_force optim as well only works for some set of num of labels')  
  parser.add_argument('--start_last_best', action='store_true',
            help='start the next guess using the last best guess, this may be diff from cnn-best guess') 
  parser.add_argument('--pair_wise_only', action='store_true',
            help='do only pair-wise in format y^t A y') 
  parser.add_argument('--keep_feasible', action='store_true',
            help='replace infeasible solution with something we know, like true labels or cnn_best, is this ideal ??')

  
  args = parser.parse_args()
  return args

