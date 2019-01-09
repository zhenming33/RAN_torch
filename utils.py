import torch
import numpy as np
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# gen data
class data_gen:
    def __init__(self, batch_size, dataset, label_path, vocab_path, dict_path, train_percent=0.7, num_workers=8):
        train_data = dataset(label_path, vocab_path, dict_path, True, train_percent)
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_data = dataset(label_path, vocab_path, dict_path, False, train_percent)
        self.valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=1)

    def train(self):
        images, encoded_captions, caption_lengths = self.train_loader.__iter__().__next__()
        images = images.to(device).float()
        encoded_captions = encoded_captions.to(device).long()
        caption_lengths = caption_lengths.to(device).long()
        return images, encoded_captions, caption_lengths

    def valid(self):
        images, encoded_captions, caption_lengths = self.valid_loader.__iter__().__next__()
        images = images.to(device).float()
        encoded_captions = encoded_captions.to(device).long()
        caption_lengths = caption_lengths.to(device).long()
        return images, encoded_captions, caption_lengths

def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

# record logs
def record_scale(writer, value, step, tag):
    writer.add_scalar(tag, value, step)

# record images and attentions
def record_images(writer, images, step):
    # show attention in tensorboard
    image_attention = images[0]
    image_attention = image_attention.squeeze()
    image_attention = (image_attention + 0.5)*255

    image_test = image_attention.cpu().numpy()
    image_test = image_test.reshape(224, 224, 3)
    writer.add_image('valid/Image', image_test, step)

def record_attention(writer, alphas, attention_seq, step):
    # t
    alphas_attention = alphas[0]
    alphas_attention_t0 = alphas_attention[attention_seq]
    alphas_attention_t0 = alphas_attention_t0.view(1, 1,20,20)
    # alphas_attention_t0 = nn.functional.upsample_bilinear(alphas_attention_t0, (224, 224))
    alphas_attention_t0 = alphas_attention_t0.squeeze(0)
    alpha_test = alphas_attention_t0.squeeze().cpu().detach().numpy()
    ii = np.unravel_index(np.argsort(alpha_test.ravel())[-100:], alpha_test.shape)
    att = torch.zeros(20,20).to(device)
    att[ii] = 1
    writer.add_image('valid/attention' + '_t' + str(attention_seq), att, step)

def record_text(writer, value, step, tag):
    writer.add_text(tag, value, step)

def record_graph(writer, model):
    images = torch.Tensor(16, 3, 224, 224).to(device)
    encoded_captions = torch.ones(16,15).to(device).long()
    caption_lengths = torch.ones(16, 1).to(device).long() * 15
    writer.add_graph(model, (images, encoded_captions, caption_lengths))

def clip_gradient(model, grad_clip):
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

# def clip_gradient(optimizer, grad_clip):
#     """
#     Clips gradients computed during backpropagation to avoid explosion of gradients.
#     :param optimizer: optimizer with the gradients to be clipped
#     :param grad_clip: clip value
#     """
#     for group in optimizer.param_groups:
#         for param in group['params']:
#             if param.grad is not None:
#                 param.grad.data.clamp_(-grad_clip, grad_clip)



