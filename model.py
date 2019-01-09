import torch
from torch import nn
from PVANet import PVANet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels,feature_size=20):
        super(Encoder, self).__init__()
        self.cnn = PVANet(in_channels, out_channels)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((feature_size, feature_size))

    def forward(self, input):
        output = self.cnn(input)
        output = self.adaptive_pool(output)
        output = output.permute(0, 2, 3, 1)
        return output

class Attention(nn.Module):
    """
    args:
    encoder_dim:    encoder output dim
    decoder_dim:    decoder hidden layers dim
    converge_vector_channel:    channel of vector channel, always equal size of encoder output
    converge_vector_dim:    converge vector dim
    attention_dim:  attention dim
    inputs:
    encoder_out:    encoder output in size(batch size * size^2 * channels)
    decoder_hidden: decoder hidden states in size(batch size * channels)
    converge_vector_dim: sum of alphas of all past time, equals to 0 at start, in size(batch size * encoder feature size^2)
    outputs:
    context:    attention context in size(batch size * encoder dim)
    alpha:  softmax of weights of encoder feature insize(batch size * encoder size^2)
    """

    def __init__(self, encoder_dim, decoder_dim, converge_vector_channel, converge_vector_dim, attention_dim=256):
        super(Attention, self).__init__()

        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.converge = nn.Linear(converge_vector_channel, converge_vector_dim)
        self.converge_att = nn.Linear(converge_vector_dim, attention_dim)
        self.tanh = nn.Tanh()
        self.full_att = nn.Linear(attention_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden, converge_vector):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)

        if sum(sum(converge_vector)).item() != 0:
            converge_vector = self.converge(converge_vector)
            att3 = self.converge_att(converge_vector)
            att = self.full_att(self.tanh(att1 + att2.unsqueeze(1) + att3.unsqueeze(1))).squeeze(2)
        else:
            att = self.full_att(self.tanh(att1 + att2.unsqueeze(1))).squeeze(2)

        # att size (batch_size, encoder_feature_length)
        alpha = self.softmax(att)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        return context, alpha



class Decoder(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=512, encoder_fsize = 20,
                 converge_vector_dim = 256, dropout=0.5, embedding_dropout=0.5):
        super(Decoder, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.encoder_fsize = encoder_fsize
        self.encoder_fl = encoder_fsize*encoder_fsize
        self.converge_vector_dim = converge_vector_dim
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout

        self.attention = Attention(self.encoder_dim, self.decoder_dim, self.encoder_fl, self.converge_vector_dim,
                                   self.attention_dim)
        self.embeddimg = nn.Embedding(vocab_size, self.embed_dim)
        self.embedding_dropout = nn.Dropout(p=self.embedding_dropout)
        self.dropout = nn.Dropout(p=self.dropout)
        self.gru1 = nn.GRUCell(self.embed_dim, decoder_dim, bias=True)
        self.gru2 = nn.GRUCell(self.encoder_dim, decoder_dim, bias=True)
        self.s = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.fc = nn.Linear(self.decoder_dim, self.vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embeddimg.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1,0.1)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        s = self.s(mean_encoder_out)
        return s

    def decode_step(self, embedding_word, s, encoder_out, converge_vector):
        # gru cell
        st_hat = self.gru1(embedding_word, s)
        context, alpha = self.attention(encoder_out, s, converge_vector)
        st = self.gru2(context, st_hat)

        # sum of history converge vector
        converge_vector = converge_vector + alpha

        # embedding predict word
        preds = self.fc(self.dropout(st))
        preds_words = preds.topk(1)[1].squeeze()
        embedding_word = self.embeddimg(preds_words)
        embedding_word = self.embedding_dropout(embedding_word)
        embedding_word = embedding_word.view(-1, self.embed_dim)

        return embedding_word, st, converge_vector, preds, alpha

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        #Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        #sort input data by decreasing lengths
        # caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        #embedding
        start_word = encoded_captions[:,0]
        embedding_word = self.embeddimg(start_word)
        embedding_word = self.embedding_dropout(embedding_word)

        #initialize GRU state
        s = self.init_hidden_state(encoder_out)

        #remove <eos> during decoding
        # decode_lengths = (caption_lengths -1).tolist()
        decode_lengths = caption_lengths.tolist()

        #create tensors to hold word prediction scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        #decode by time t
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            embedding_word, s, converge_vector, preds, alpha = self.decode_step(embedding_word[:batch_size_t],
                                                                           s[:batch_size_t],
                                                                           encoder_out[:batch_size_t],
                                                                           converge_vector = torch.zeros(batch_size_t, num_pixels).to(device)
                                                                           if t==0 else converge_vector[:batch_size_t])
            predictions[:batch_size_t, t] = preds
            alphas[:batch_size_t, t] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind




class Model(nn.Module):
    def __init__(self, img_channels, vocab_size, encoder_dim=512, encoder_fsize=20, embed_dim=256, decoder_dim=256,
                 attention_dim=256, converge_vector_dim=256, dropout=0.5, embedding_dropout=0.1):
        super(Model, self).__init__()

        self.encoder = Encoder(img_channels, encoder_dim, encoder_fsize)
        self.decoder = Decoder(attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim, encoder_fsize,
                                converge_vector_dim, dropout, embedding_dropout)
        self.encoder.apply(self.init_weight)
        self.decoder.apply(self.init_weight)

    def init_weight(self, m):
        # 使用isinstance来判断m属于什么类型
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def forward(self, images, encoded_captions, caption_lengths):
        encoder_out = self.encoder(images)
        predictions, encoded_captions, decode_lengths, alphas, sort_ind = self.decoder(encoder_out, encoded_captions, caption_lengths)
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind








if __name__ == "__main__":
    #already converge vector
    #already not calculate padding loss
    #already allow loading pretrained embeddings
    #already add embedding dropout
    #todo add sigmoid gate to control context
    #todo add weights init methods
    #todo add gradient clip
    #todo try doubly stochastic regularization
    #todo add teaching learning
    # x = torch.Tensor(16, 3, 224, 224).to(device)
    # x = Encoder(3).to(device)(x)
    # x.size()    #(16,512,20,20)
    # encoder_out = torch.Tensor(16, 400, 512).to(device)
    # decoder_hidden = torch.Tensor(16, 256).to(device)
    # converge_vector = torch.Tensor(16, 400).to(device)
    #
    # context, alpha = Attention(512, 256, 400, 256).to(device)(encoder_out, decoder_hidden, converge_vector)

    # encoder_out = torch.Tensor(16,20,20,512).to(device)
    # encoded_captions = torch.ones(16,15).to(device).long()
    # caption_lengths = torch.ones(16,1).to(device).long()*15
    # predictions, encoded_captions, decode_lengths, alphas, sort_ind = Decoder(256, 256, 256, 13).to(device)(encoder_out, encoded_captions, caption_lengths)

    images = torch.Tensor(16, 3, 224, 224).to(device)
    encoded_captions = torch.ones(16,15).to(device).long()
    caption_lengths = torch.ones(16, 1).to(device).long() * 15
    model = Model(3, 13).to(device)
    predictions, encoded_captions, decode_lengths, alphas, sort_ind = model(images, encoded_captions, caption_lengths)











