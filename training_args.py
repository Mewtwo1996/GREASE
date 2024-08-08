import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Model
    parser.add_argument('--n_hop', type=int, default=2, 
                        help='n_hops')
    parser.add_argument('--path', type=str, default="../data/lastfm", 
                        help="path")
    parser.add_argument('--file', type=str, default="lgn-lastfm-3-64.pth.tar", 
                        help="file")
    
    parser.add_argument('--device', type=str, default="cuda", 
                        help="device")
    parser.add_argument('--epoch', type=int, default=360, 
                        help="epoch")
    parser.add_argument('--epoch', type=int, default=360, 
                        help="epoch")
    
    parser.add_argument('--alpha', type=float, default=0.3, 
                        help='initial learning rate')
    parser.add_argument('--beta', type=float, default=0.03, 
                        help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.01, 
                        help='initial learning rate')
    parser.add_argument('--lamb', type=float, default=0.008, 
                        help='initial learning rate')
    

    return args
