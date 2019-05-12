import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random
from tqdm import tqdm
import argparse

Conv_W = 3
CC, LL, WW = 32, 21, 21
EMB_DIM = 32
PAD = 2 * (int((Conv_W -1) / 2), )

def to_torch(x, dtype="float", req=False):
    tor_type = torch.cuda.LongTensor if dtype == "int" else torch.cuda.FloatTensor
    x = Variable(torch.from_numpy(x).type(tor_type), requires_grad=req)
    return x


class CNN(nn.Module):
    def __init__(self, n_chan):
        super(CNN, self).__init__()
        # 1 channel input to 2 channel output of first time print and written
        self.conv1 = nn.Conv2d(n_chan, 8, Conv_W, padding = PAD)
        self.conv2 = nn.Conv2d(8, 16, Conv_W, padding = PAD)
        self.conv3 = nn.Conv2d(16, 32, Conv_W, padding = PAD)

        self.dense_enc = nn.Linear(CC * LL * WW, 100)

        # variational bits
        self.fc_mu = nn.Linear(100, EMB_DIM)
        self.fc_logvar = nn.Linear(100, EMB_DIM)

        self.dense_dec = nn.Linear(EMB_DIM, CC * LL * WW)

        self.deconv3 = torch.nn.ConvTranspose2d(32, 16, Conv_W, padding = PAD)
        self.deconv2 = torch.nn.ConvTranspose2d(16, 8, Conv_W, padding = PAD)
        self.deconv1 = torch.nn.ConvTranspose2d(8, n_chan, Conv_W, padding = PAD)

        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)

        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        # conv1
        x = F.relu(self.conv1(x))
        size1 = x.size()
        # x, idx1 = self.pool(x)

        # conv2
        x = F.relu(self.conv2(x))
        size2 = x.size()
        x, idx2 = self.pool(x)
        # print('size2=',size2)

        # conv3
        x = F.relu(self.conv3(x))
        size3 = x.size()
        x, idx3 = self.pool(x)

        # =================================================
        # reached the middle layer, some dense
        x = x.view(-1, CC * LL * WW)
        # x = x.view(-1,8*60*60)
        x = torch.relu(self.dense_enc(x))

        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        x = self.reparameterize(mu, logvar)

        x = F.relu(self.dense_dec(x))
        x = x.view(-1, CC, LL, WW)
        # =================================================

        # deconv3
        x = self.unpool(x, idx3, size3)
        x = F.relu(self.deconv3(x))

        # deconv2
        x = self.unpool(x, idx2, size2)
        x = F.relu(self.deconv2(x))

        # deconv1
        # x = self.unpool(x, idx1, size1)
        x = self.deconv1(x)
        # x = torch.sigmoid(self.deconv1(x))
        return x, mu, logvar

    # def decode(self, x):

    def embed(self, x):
        _, mu, _ = self(to_torch(x))
        return mu

    # VAE MAGIC =================

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def learn_once(self, imgs):
        img_rec, mu, logvar = self(imgs)

        self.opt.zero_grad()

        L2_LOSS = ((img_rec - imgs) ** 2).mean()
        KLD_LOSS = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # print (L2_LOSS, KLD_LOSS)
        loss = L2_LOSS + KLD_LOSS * 0.01
        loss.backward()

        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)

        self.opt.step()
        return loss

    def save(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load(self, loc):
        self.load_state_dict(torch.load(loc))
    def draw(self, img,filename):
        from PIL import Image
        im = Image.fromarray(img)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(filename)

    def learn(self, X, learn_iter=1000, batch_size=40):
        losses = []
        # for i in range(99999999999):
        for i in tqdm(range(learn_iter)):
            # load in the datas
            indices = sorted(random.sample(range(len(X)), batch_size))
            # indices = list(range(40))
            X_sub = X[indices]
            # convert to proper torch forms
            X_sub = to_torch(X_sub)

            # optimize
            losses.append(self.learn_once(X_sub).data.cpu().numpy())

            if i % 1000 == 0:
                print(i, losses[len(losses) - 1])
                img_orig = X_sub[30].detach().cpu().numpy()
                img_rec = self(X_sub)[0][30].detach().cpu().numpy()
                self.draw(img_orig[0] * 256 ,'drawings/orig_img0.png')
                self.draw(img_orig[1] * 256 ,'drawings/orig_img1.png')
                self.draw(img_orig[2] * 256 ,'drawings/orig_img2.png')
                self.draw(img_orig[3] * 256 ,'drawings/orig_img3.png')
                self.draw(img_rec[0]  * 256 , 'drawings/rec_img0.png')
                self.draw(img_rec[1]  * 256 , 'drawings/rec_img1.png')
                self.draw(img_rec[2]  * 256 , 'drawings/rec_img2.png')
                self.draw(img_rec[3]  * 256 , 'drawings/rec_img3.png')
                img_orig = X_sub[35].detach().cpu().numpy()
                img_rec = self(X_sub)[0][35].detach().cpu().numpy()
                self.draw(img_orig[0] * 256 ,'drawings/orig_img10.png')
                self.draw(img_orig[1] * 256 ,'drawings/orig_img11.png')
                self.draw(img_orig[2] * 256 ,'drawings/orig_img12.png')
                self.draw(img_orig[3] * 256 ,'drawings/orig_img13.png')
                self.draw(img_rec[0]  * 256 , 'drawings/rec_img10.png')
                self.draw(img_rec[1]  * 256 , 'drawings/rec_img11.png')
                self.draw(img_rec[2]  * 256 , 'drawings/rec_img12.png')
                self.draw(img_rec[3]  * 256 , 'drawings/rec_img13.png')

def flatten(X):
    s = X.shape
    s_new = (s[0], s[1] * s[2] * s[3])
    return np.reshape(X, s_new)

def get_X_Y(filename):
    with open(filename, 'rb') as f:
        X,Y = pickle.load(f)
    return X,Y

def save_X_Y(filename, X, Y):
    with open(filename, 'wb') as f:
        pickle.dump((X, Y), f, protocol=4)
    return


def project(X, vae,  loop=40):
    X_new = []
    for i in range(0, X.shape[0], loop):
        X_new.append(vae.embed(X[i:i + loop]).cpu().detach().numpy())

    #print (X_new)
    X_new = np.vstack(X_new)
    print (X_new.shape)
    print (np.max(X_new))
    print (np.min(X_new))
    # add small noise to break it up
    X_new = X_new + np.random.uniform(-1e-8, 1e-8, X_new.shape)
    print (np.max(X_new))
    print (np.min(X_new))

    return X_new


if __name__ == '__main__':
    """
    learn_iter working with 50000
    batch_size working with 40
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("load_path")
    parser.add_argument("save_path")
    parser.add_argument("learn_iter", type=int)
    parser.add_argument("batch_size", type=int)
    args = parser.parse_args()

    print ('embedding ppo2 now')
    
    X_tr, Y_tr = get_X_Y(args.load_path)#'baselines/baselines/ppo2_memory_obs_actions')
    
    #save_X_Y('baselines/baselines/ppo2_memory_obs_actions_small',X_tr[:100],Y_tr[:100])
    #print('saved')
    X_tr = np.transpose(X_tr,(0,3,1,2))

    print ("binarize the image a bit ? ")
    #X_tr = X_tr / 256
    X_tr[X_tr <= 128] = 0.0
    X_tr[X_tr > 128] = 1.0

    emb_dim = EMB_DIM

    import pickle
    print ("learning iter", args.learn_iter)
    print ("batch_size ", args.batch_size)
    vae = CNN(4).cuda()
    #vae.draw(X_tr[13][0],'drawings/t1.png')
    #vae.draw(X_tr[130][0],'drawings/t2.png')
    #vae.draw(X_tr[1300][0],'drawings/t3.png')
    
    vae.learn(X_tr, args.learn_iter, args.batch_size)

    # compute the embedded features
    #X_tr_emb = vae.embed(X_tr)
    X_tr_emb = project(X_tr,vae)

    data_embed_path = args.save_path#'pong_emb2_{}.p'.format(emb_dim)
    pickle.dump((X_tr_emb,Y_tr), open( data_embed_path, "wb" ) )
