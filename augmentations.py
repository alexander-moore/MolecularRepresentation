# Augmentation holder
# Augmentations are transformers to data (X,y) returning data (X,y)
# Keep an open mind about making them as flexible as possible

# Check out ConvNext, styleGAn and other modern augmentation approaches
import torch
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Linear transforms
def signal_squisher(x,y):
    # Squishes signals


    x = torch.stack([z*random.uniform(0.85, 0.95) for z in x]).to(device)
    
    return x,y

def signal_stretcher(x,y):
    # Stretches signals bigger
    x = torch.stack([z*random.uniform(1.05, 1.15) for z in x]).to(device)

    return x,y

# Nonlinear transforms
# def signal_sqrt(x,y):
#   # this one is probably too nasty. maybe just signal clamper instead
#   x = torch.clamp(x,0,torch.max(x))
#   x = torch.sqrt(x)
#   return x,y

# def signal_clamp_low(x,y):
#     # clamps out the negative values
#     x = torch.clamp(x,0,torch.max(x))
#     x = torch.sqrt(x)
#     return x,y

# def signal_clamp_high(x,y):
#     # clamps out over 90% of max values
#     x = torch.clamp(x,0,torch.max(x)*.95)
#     x = torch.sqrt(x)
#     return x,y

# MixUp https://keras.io/examples/vision/mixup/
#def mixup(x,y):
    # Combine random samples of x and their labels
    # take a random weight A (0,1) and corresponding weight B (1-A)
    # Then take the sum of the images or sensors
    # Label becomes sum

    #return x,y

# Noising transforms
def small_signal_noiser(x,y):
    # Adds a little noise


    x = torch.stack([z + 0.1*torch.randn_like(z, device = z.device) for z in x]).to(device)

    return x,y

def big_signal_noiser(x,y):
    # Adds a lot of noise

    #x = x + 1.5*torch.randn_like(x, device = x.device)
    x = torch.stack([z + 0.3*torch.randn_like(z, device = z.device) for z in x]).to(device)

    return x,y

# def signal_translator(x,y):
#     # Translate entire signal up by 1-5%

#     x = torch.stack([z + random.uniform(-0.01, 0.01)*torch.max(z) for z in x]).to(device)

#     return x,y

def signal_early(x,y):
    # Make the entire signal late or early
    # since we will later take signal[:,:,175:275]
    newx = []
    for samp in x:
        size = random.randint(1,5)
        noise = torch.randn((x.shape[1],size), device = device)

        samp = samp[:,size:]
        samp = torch.cat((samp, noise), dim=1)
        newx.append(samp)

    x = torch.stack(newx).to(device)

    return x, y

def signal_late(x,y):
        # signal late:
        # append noise to beggin
    newx = []


    for samp in x:
        size = random.randint(1,5)
        noise = torch.randn((x.shape[1],size), device = device)

        samp = samp[:,:-size]
        #print(noise.shape, samp.shape)
        samp = torch.cat((noise, samp), dim=1)
        #print(samp.shape)
        # take off the end of samp to compensate

        newx.append(samp)

    x = torch.stack(newx).to(device)

    return x,y

# def sensor_translator(x,y):
#   # Translate one sensor 

#   return x,y

# def signal_masker(x,y):
#     # Mask a random component of the signal
#     newx = []
#     for samp in x:
#         rand_sens = torch.randint(0,x.shape[1],size=(1,))
#         samp[rand_sens,:] = 0

#         newx.append(samp)

#     x = torch.stack(newx).to(device)
#     return x,y

# def label_smoothing(x,y):
#     # Smooth labels y - given binary labels? - won't work with Scikit
#     y = torch.clamp(y, 0.1, 0.9)

#     return x,y