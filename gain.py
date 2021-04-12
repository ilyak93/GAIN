import torch
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import re
import io
import json
import math
import models
from tensorboardX import SummaryWriter
import tqdm

model_file_reg = re.compile(
    r'saved_model_(?P<epoch>\d+)_(?P<tag>[a-zA-Z0-9-_]+)\.model')


def scalar(tensor):
    return tensor.data.cpu().item()



class AttentionGAIN:
    def __init__(self,
                 model_type=None,
                 gradient_layer_name=None,
                 weights=None,
                 heatmap_dir=None,
                 saved_model_dir=None,
                 epoch=0,
                 gpu=False,
                 alpha=1,
                 omega=10,
                 sigma=0.5,
                 labels=None,
                 input_channels=None,
                 input_dims=None,
                 batch_norm=True):

        # validation
        if not model_type:
            raise ValueError('Missing required argument model_type')
        if not gradient_layer_name:
            raise ValueError('Missing required argument gradient_layer_name')
        if not input_channels:
            raise ValueError('Missing required argument input_channels')
        if not input_dims:
            raise ValueError('Missing required argument input_dims')

        # set gpu options
        self.gpu = gpu

        # define model
        self.model_type = model_type
        self.model = models.get_model(
            self.model_type,
            len(labels),
            batch_norm=batch_norm,
            num_channels=input_channels)
        if weights:
            self.model.load_state_dict(weights)
            self.epoch = epoch
        elif epoch > 0:
            raise ValueError('epoch_offset > 0, but no weights were supplied')

        if self.gpu:
            self.model = self.model.cuda()
            self.tensor_source = torch.cuda
        else:
            self.tensor_source = torch

        # wire up our hooks for heatmap creation
        self._register_hooks(gradient_layer_name)

        # create loss function
        # TODO make this configurable
        self.loss_cl = torch.nn.BCEWithLogitsLoss()

        # output directory setup
        self.heatmap_dir = heatmap_dir
        if self.heatmap_dir:
            self.heatmap_dir = os.path.abspath(self.heatmap_dir)

        self.saved_model_dir = saved_model_dir
        if self.saved_model_dir:
            self.saved_model_dir = os.path.abspath(saved_model_dir)

        # misc. parameters
        self.omega = omega
        self.sigma = sigma
        self.alpha = alpha
        self.labels = labels
        self.input_channels = input_channels
        self.input_dims = input_dims
        self.epoch = epoch

    @staticmethod
    def load(model_path, **kwargs):
        model_dict = torch.load(model_path)
        return AttentionGAIN(weights=model_dict['state_dict'], **kwargs)

    @staticmethod
    def _parse_saved_model_name(model_name):
        result = model_file_reg.match(model_name)
        if not result:
            raise ValueError(
                'Could not parse tag from model name %s' % model_name)

        return result.group('epoch'), result.group('tag')

    def _register_hooks(self, layer_name):
        # this wires up a hook that stores both the activation and gradient of the conv layer we are interested in
        def forward_hook(module, input_, output_):
            self._last_activation = output_

        def backward_hook(module, grad_in, grad_out):
            self._last_grad = grad_out[0]

        # locate the layer that we are concerned about
        gradient_layer_found = False
        for idx, m in self.model.named_modules():
            if idx == layer_name:
                m.register_forward_hook(forward_hook)
                m.register_backward_hook(backward_hook)
                gradient_layer_found = True
                break

        # for our own sanity, confirm its existence
        if not gradient_layer_found:
            raise AttributeError(
                'Gradient layer %s not found in the internal model' %
                layer_name)

    def __str__(self):
        meta_dict = self._get_meta_dict()
        ret_str = 'Metadata:'
        for k, v in meta_dict.items():
            ret_str += '\n\t%s: %s' % (str(k), str(v))
        ret_str += '\n'

        ret_str += 'Layers:\n'
        models.model_to_str(self.model)

        return ret_str

    def _convert_data_and_label(self, data, label,
                                extra_super=None,
                                am_mask=None):
        # converts our data and label over to variables, gpu optional
        if self.gpu:
            data = data.cuda()
            label = [l.cuda() for l in label]
            if(extra_super is not None):
                extra_super = [e.cuda() for e in extra_super]
            if(am_mask is not None):
                am_mask = [e.cuda() for e in am_mask]

        data = torch.autograd.Variable(data)
        label = [torch.autograd.Variable(l) for l in label]

        if (extra_super is not None):
            extra_super = [torch.autograd.Variable(e) for e in extra_super]
        if (am_mask is not None):
            am_mask = [torch.autograd.Variable(e) for e in am_mask]

        return data, label, extra_super, am_mask

    def _maybe_save_model(self, serialization, tag='default', save_count=1):
        # TODO if a different save count but same tag is used in different circumstances, this will have
        # undefined behavior (we only delete one file if there are too many, but we should be trimming to *save_count*
        if self.saved_model_dir is None:
            return

        if not os.path.exists(self.saved_model_dir):
            try:
                os.makedirs(self.saved_model_dir)
            except OSError as e:
                print(
                    'WARNING there was an error while creating directory %s: %s'
                    % (str(self.saved_model_dir), str(e)))
                return
        if serialization == 'onnx':
            extension = '.onnx'
        else:
            extension = '.pyt'

        max_epoch = self.epoch
        delete_model_path = None
        num_models = 0
        # store the model for later deletion
        for p in os.listdir(self.saved_model_dir):
            if not os.path.splitext(p)[-1] == extension:
                continue

            try:
                temp_epoch, temp_tag = self._parse_saved_model_name(p)
            except ValueError as e:
                print('WARNING error while parsing saved model filename: %s' %
                      str(e))
                continue
            if temp_tag != tag:
                continue

            temp_epoch = int(temp_epoch)
            num_models += 1

            if temp_epoch < max_epoch:
                delete_model_path = os.path.join(self.saved_model_dir, p)
                max_epoch = temp_epoch

        # if we are less that the max saved model count, then don't worry about it
        if num_models < save_count:
            delete_model_path = None

        # save the current model
        saved_model_filename = os.path.join(
            self.saved_model_dir,
            'saved_model_%i_%s%s' % (self.epoch, tag, extension))
        if serialization == 'onnx':
            dummy_dims = (1, self.input_channels) + self.input_dims
            dummy_input = torch.autograd.Variable(torch.randn(dummy_dims))
            if self.gpu:
                dummy_input = dummy_input.cuda()
            try:
                torch.onnx.export(self.model, dummy_input,
                                  saved_model_filename)
                print('MODEL saved to %s' % saved_model_filename)
            except OSError as e:
                print('WARNING there was an error while saving model: %s' %
                      str(e))

        else:
            try:
                torch.save(self.model.state_dict(), saved_model_filename)
                print('MODEL saved to %s' % saved_model_filename)
            except OSError as e:
                print('WARNING there was an error while saving model: %s' %
                      str(e))
                return

            # delete our extra model
            if delete_model_path:
                try:
                    os.remove(delete_model_path)
                except OSError as e:
                    print(
                        'WARNING there was an error while trying to remove file %s: %s'
                        % (delete_model_path, e))

    def _maybe_save_heatmap(self, image, label, heatmap, I_star, epoch,
                            heatmap_nbr):

        if self.heatmap_dir is None:
            return

        heatmap_image = self._combine_heatmap_with_image(
            image, heatmap, self.labels[label])

        I_star = (I_star.data.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        out_image = np.concatenate((heatmap_image, I_star), axis=1)
        out_image = torch.tensor(out_image.transpose((2, 0, 1)))

        self.writer.add_image('heatmap', out_image, epoch)

    @staticmethod
    def _combine_heatmap_with_image(image,
                                    heatmap,
                                    label_name,
                                    font_scale=0.75,
                                    font_name=cv2.FONT_HERSHEY_SIMPLEX,
                                    font_color=(255, 255, 255),
                                    font_pixel_width=1):


        image = image.data.cpu().numpy().transpose((1, 2, 0)) * 255
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # get the min and max values once to be used with scaling
        min_val = heatmap.min()
        max_val = heatmap.max()

        # Scale the heatmap in range 0-255
        heatmap = (255 * (heatmap - min_val)) / (max_val - min_val + 1e-5)
        heatmap = heatmap.data.cpu().numpy().astype(np.uint8).transpose((1, 2,
                                                                         0))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # generate the heatmap
        heatmap_image = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)

        # superimpose label_name
        (_, text_size_h), baseline = cv2.getTextSize(
            label_name, font_name, font_scale, font_pixel_width)
        heatmap_image = cv2.putText(
            heatmap_image,
            label_name, (10, text_size_h + baseline + 10),
            font_name,
            font_scale,
            font_color,
            thickness=font_pixel_width)
        heatmap_image = cv2.cvtColor(heatmap_image, cv2.COLOR_BGR2RGB)
        return heatmap_image

    def generate_heatmap(self, data, label, width=3):
        data_var, label_var = self._convert_data_and_label(data, label)
        label_name = self.labels[int(label.max())]
        output_cl, loss_cl, A_c = self._attention_map_forward(
            data_var, label_var)
        heatmap = self._combine_heatmap_with_image(data[0], A_c[0], label_name)
        return output_cl, loss_cl, A_c, heatmap

    def forward(self, data, label, extra_super=None, am_mask=None):
        data, label, extra_super, am_mask = self._convert_data_and_label(
            data, label, extra_super, am_mask)
        return self._forward(data, label, extra_super, am_mask)

    def _check_dataset_compatability(self, rds):
        if rds.output_dims != self.input_dims:
            raise ValueError(
                'Dataset outputs images with dimension %s, model expects %s' %
                (str(rds.output_dims), str(self.input_dims)))
        elif rds.output_channels != self.input_channels:
            raise ValueError(
                'Dataset outputs images with channel_count %i, model expects %i'
                % (rds.output_channels, self.input_channels))
        elif rds.dataset.categories != self.labels:
            raise ValueError('Dataset has labels %s, model has labels %s' %
                             (str(rds.labels), str(self.labels)))


    def train(self,
              rds,
              epochs,
              serialization_format,
              pretrain_epochs=10,
              learning_rate=1e-5,
              test_every_n_epochs=5,
              num_heatmaps=1):
        # TODO dynamic optimizer selection
        self._check_dataset_compatability(rds)
        self.writer = SummaryWriter(self.saved_model_dir)

        last_acc = 0
        max_acc = 0
        pretrain_finished = False
        opt = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        for i in range(self.epoch, epochs, 1):
            self.epoch = i
            pretrain_finished = pretrain_finished or \
                                i > pretrain_epochs
            loss_cl_sum = 0
            loss_am_sum = 0
            acc_cl_sum = 0
            total_loss_sum = 0

            # train
            samp_ = 0
            pbar = tqdm.tqdm(total=len(rds.datasets['train']))
            train_size = len(rds.datasets['train'])

            for sample in rds.datasets['train']:

                r = self.forward(
                    sample['image'],
                    sample['label/onehot'],
                    extra_super=sample['label/truths'],
                    am_mask=sample['label/masks'])

                total_loss_sum += scalar(r['total_loss'])
                loss_cl_sum += scalar(r['loss_cl'])
                loss_am_sum += scalar(r['loss_am'])
                acc_cl_sum += scalar(r['cl_acc'])

                samp_ += 1

                # Backprop selectively based on pretraining/training
                if pretrain_finished:
                    print_prefix = 'train'

                    r['total_loss'].backward()

                    pbar.set_description('[{}] loss_cl: {:.4f}, loss_am: {:.4f}'.format(
                        print_prefix,
                        loss_cl_sum/samp_,
                        loss_am_sum/samp_))
                else:
                    print_prefix = 'pretrain'

                    r['loss_cl'].backward()

                    pbar.set_description('[{}] loss_cl: {:.4f}'.format(
                        print_prefix,
                        loss_cl_sum/samp_))

                opt.step()
                pbar.update(1)

            last_acc = acc_cl_sum / train_size

            if(pretrain_finished):
                self.writer.add_scalar('train/loss_am', loss_am_sum/train_size, i+1)
                self.writer.add_scalar('train/loss', total_loss_sum/train_size, i+1)
                self.writer.add_scalar('train/avg_acc', last_acc*100.0, i+1)
                self.writer.add_scalar('train/loss_cl', loss_cl_sum/train_size, i+1)
                print(
                    '{} Epoch {}, Loss_CL: {:.4f}, Loss_AM: {:.4f}, Loss Total: {:.4f}, Accuracy_CL: {:.4f}%%'.format(
                        print_prefix,
                        (i + 1),
                        loss_cl_sum / train_size,
                        loss_am_sum / train_size,
                        total_loss_sum / train_size,
                        last_acc * 100.0))
            else:
                self.writer.add_scalar('pretrain/loss_cl', loss_cl_sum/train_size, i+1)
                print(
                    '{} Epoch {}, Loss_CL: {:.4f}, Accuracy_CL: {:.4f}%'.format(
                        print_prefix,
                        (i + 1),
                        loss_cl_sum / train_size,
                        last_acc * 100.0))

            samp_ = 0
            pbar = tqdm.tqdm(total=len(rds.datasets['test']))
            if (i + 1) % test_every_n_epochs == 0:
                # test
                loss_cl_sum = 0
                loss_am_sum = 0
                acc_cl_sum = 0
                total_loss_sum = 0
                heatmap_count = 0
                for sample in rds.datasets['test']:
                    # test
                    r = self.forward(
                        sample['image'],
                        sample['label/onehot'],
                        extra_super=sample['label/truths'],
                        am_mask=sample['label/masks'])

                    total_loss_sum += scalar(r['total_loss'])
                    loss_cl_sum += scalar(r['loss_cl'])
                    loss_am_sum += scalar(r['loss_am'])
                    acc_cl_sum += scalar(r['cl_acc'])

                    samp_ += 1
                    pbar.set_description('[test] loss_cl: {:.4f}, loss_am: {:.4f}'.format(
                        loss_cl_sum/samp_,
                        loss_am_sum/samp_))
                    pbar.update(1)

                    if(heatmap_count < num_heatmaps):
                        self._maybe_save_heatmap(sample['image'][0],
                                                 sample['label/idx'][0],
                                                 r['gcams'][0][0],
                                                 r['I_stars'][0][0],
                                                 i + 1,
                                                 heatmap_count)
                        heatmap_count += 1
                    self._maybe_save_model('pth',
                                           tag='default',
                                           save_count=15)

                test_size = len(rds.datasets['test'])
                avg_acc = acc_cl_sum / test_size
                self.writer.add_scalar('test/loss_cl', loss_cl_sum/test_size, i+1)
                self.writer.add_scalar('test/loss_am', loss_am_sum/test_size, i+1)
                self.writer.add_scalar('test/loss', total_loss_sum, i+1)
                self.writer.add_scalar('test/avg_acc', avg_acc*100.0, i+1)

                print(
                    'TEST Loss_CL: {:.4f}, Loss_AM: {:.4f}, Loss_Total: {:.4f}, Accuracy_CL: {:.4f}%%'.format(
                        loss_cl_sum / test_size,
                        loss_am_sum / test_size,
                        total_loss_sum / test_size,
                        avg_acc * 100.0))


    def _attention_map_forward(self, data, labels):
        output_cl = self.model(data)

        # attention maps
        A_cs = []
        loss_cl = torch.tensor([0.], device=data.device)

        for label in labels:

            #grad_target = (output_cl * label).sum()
            grad_target = (output_cl * label)
            grad_target.backward(gradient=label * output_cl, retain_graph=True)
            #grad_target.backward(retain_graph=True)

            # Eq 1
            w_c = F.avg_pool2d(
                self._last_grad,
                (self._last_grad.shape[-2], self._last_grad.shape[-1]), 1)
            w_c_new_shape = (w_c.shape[0] * w_c.shape[1], w_c.shape[2],
                             w_c.shape[3])
            w_c = w_c.view(w_c_new_shape).unsqueeze(0)

            # Eq 2
            weights = self._last_activation
            weights_new_shape = (weights.shape[0] * weights.shape[1],
                                 weights.shape[2], weights.shape[3])
            weights = weights.view(weights_new_shape).unsqueeze(0)

            gcam = F.relu(F.conv2d(weights, w_c))
            A_c = F.upsample(gcam, size=data.size()[2:], mode='bilinear', align_corners=True)
            A_cs.append(A_c)

            loss_cl += self.loss_cl(output_cl, label)

        self.model.zero_grad()

        return output_cl, loss_cl, A_cs

    def _mask_image(self, gcam, image):
        gcam_min = gcam.min()
        gcam_max = gcam.max()
        scaled_gcam = (gcam - gcam_min) / (gcam_max - gcam_min)
        mask = F.sigmoid(self.omega * (scaled_gcam - self.sigma)).squeeze()
        masked_image = image - (image * mask)

        return masked_image

    def _forward(self, data, labels, extra_super=None, am_mask=None):

        I_stars = []
        loss_e = torch.tensor([0.], device=data.device)
        total_loss = torch.tensor([0.], device=data.device)

        output_cl, loss_cl, gcams = self._attention_map_forward(data, labels)
        output_cl_softmax = F.softmax(output_cl, dim=1)
        total_loss += loss_cl

        if (extra_super is None):
            extra_super = [None] * len(labels)

        # Eq 3
        for gcam, label, extra in zip(gcams, labels, extra_super):
            I_star = self._mask_image(gcam, data)
            I_stars.append(I_star)

            output_am = self.model(I_star)

            # Eq 5
            loss_am = F.sigmoid(
                output_am.view(-1)[label.view(-1).nonzero().view(-1)])

            total_loss += self.alpha * loss_am

            # Eq 7 (extra supervision)
            if (extra is not None):
                loss_e = ((gcam - extra)**2).sum()
                total_loss += self.omega * loss_e

        cl_acc = output_cl_softmax.max(dim=1)[1] == label.max(dim=1)[1]
        cl_acc = cl_acc.type(self.tensor_source.FloatTensor).mean()

        return {'total_loss': total_loss,
                'loss_cl': loss_cl,
                'loss_am': loss_am,
                'loss_e': loss_e,
                'output_cl_softmax': output_cl_softmax,
                'cl_acc': cl_acc,
                'gcams': gcams,
                'I_stars': I_stars} 
