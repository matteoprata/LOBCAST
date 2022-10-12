import numpy as np
import pandas as pd 
import torch



def compute_mask(lob_orders, pertubation, i_col, debug=False):
    # make perturbation as a mask
    pertubation = torch.sign(pertubation)
    pertubation[pertubation < 0] = 0
    # print original shape
    if debug:
        print(pertubation.shape)
    # add a new dimension to comply with best-bid / best-ask
    mask = torch.zeros((lob_orders.shape[0], lob_orders.shape[1], 10), device=lob_orders.device)
    mask[:, :, i_col] = pertubation[:, :, i_col-1]
    if debug:
        print(mask.shape)
    return mask 

def lob_prices_and_volumes(lob_orders, sell=True):
    # sell 
    start_price = 0 if sell else 2
    start_volume = start_price + 1

    lob_prices = lob_orders[:, :, start_price::4]  # first dimension batch, time, columns (price x volume)
    lob_volumes = lob_orders[:, :, start_volume::4]
    return lob_prices, lob_volumes
    
def lob_prices_and_volumes_setup(lob_orders, lob_prices, lob_volumes, sell=True):
    # sell 
    start_price = 0 if sell else 2
    start_volume = start_price + 1

    lob_orders[:, :, start_price::4] = lob_prices  # first dimension batch, time, columns (price x volume)
    lob_orders[:, :, start_volume::4] = lob_volumes
    return lob_orders

def fill_and_neg_mask(mask, i_col):
    fill_mask = mask.clone()
    # TODO: check 
    fill_mask[:, :, :i_col] = mask[:, :, i_col:i_col+1]  # fill
    fill_mask = fill_mask + (torch.ones(fill_mask.shape, device=mask.device) * (1 - mask[:,:,i_col:i_col+1]))
    neg_fill_mask = -fill_mask + 1
    
    return fill_mask, neg_fill_mask


def add_perturbation(lob_orders, perturbation, nlevels=10, min_volume=1):
    lob_orders_tmp = torch.clone(lob_orders.detach())
    perturbation = perturbation.detach()
    # TODO: check nlevels +1 
    for i_col in range(1, nlevels):
        lob_orders_tmp = perturbate_column(lob_orders_tmp, perturbation, i_col, min_volume)
    return lob_orders_tmp


def perturbate_column(lob_orders, perturbation, i_col, min_volume=1):
    #MASK
    mask = compute_mask(lob_orders, perturbation, i_col)
    #print("mask", mask)
    not_mask = -mask + 1
    fill_mask, neg_fill_mask = fill_and_neg_mask(mask, i_col)
    #print("filled mask", fill_mask, "Neg fill", neg_fill_mask)
    # get prices and volumes 
    lob_prices, lob_volumes = lob_prices_and_volumes(lob_orders)
    
    #print("before", lob_prices)
    # work only on prices
    # apply the shifting to each row 
    lob_prices_shifted = torch.roll(lob_prices, shifts=1, dims=2)
    avg_prices = (lob_prices + lob_prices_shifted) / 2
    lob_prices = lob_prices * not_mask * fill_mask + lob_prices_shifted * neg_fill_mask
    # put fake prices
    lob_prices = lob_prices * not_mask + avg_prices * mask
    
    # work only on volumes
    lob_volumes_shifted = torch.roll(lob_volumes, shifts=1, dims=2)
    lob_volumes = lob_volumes * not_mask * fill_mask + lob_volumes_shifted * neg_fill_mask
    # put fake volumes
    lob_volumes[lob_volumes <= 0] = min_volume
    
    # set back values inside the lob_orders
    lob_orders = lob_prices_and_volumes_setup(lob_orders, lob_prices, lob_volumes)
 
    return lob_orders
    