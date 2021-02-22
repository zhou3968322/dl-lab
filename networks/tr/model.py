# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2021/2/1
from torch import nn
from networks.tr.transformation import TPS_SpatialTransformerNetwork
from networks.tr.tr_fe import PaperCNN, RCNN, ResNetF, ResNetAster, ResNetFPN
from networks.components.lstm import BidirectionalLSTM
from networks.tr.tr_seq.bert_ocr import Config, BertOcr
from networks.tr.tr_seq import TransformEncoder
from networks.tr.tr_pred import Attention, SrnDecoder


class TextRecognitionModel(nn.Module):

    def __init__(self, transformation=None, feature_extraction="ResNet",
                 sequence_modeling="SRN", prediction="SRN",
                 img_h=48, img_w=800,
                 input_channel=3, feature_output_channel=512,
                 alphabet_size=36, position_dim=256,
                 num_fiducial=20, hidden_size=256,
                 batch_max_length=25,
                 ):
        """
        alphabet_size是代表字符个数
        """
        super(TextRecognitionModel, self).__init__()
        if transformation == "TPS":
            self.transformation = TPS_SpatialTransformerNetwork(F=num_fiducial,
                                                                I_size=(img_h, img_w),
                                                                I_r_size=(img_h, img_w),
                                                                I_channel_num=input_channel)
        else:
            self.transformation = None

        if feature_extraction == "VGG":
            self.feature_extraction = PaperCNN(input_channel=input_channel, output_channel=feature_output_channel)
        elif feature_extraction == "RCNN":
            self.feature_extraction = RCNN(input_channel=input_channel, output_channel=feature_output_channel)
        elif feature_extraction == "ResNet":
            self.feature_extraction = ResNetF(input_channel=input_channel, output_channel=feature_output_channel)
        elif feature_extraction == "ResNetAster":
            self.feature_extraction = ResNetAster(input_channel=input_channel, output_channel=feature_output_channel)
        elif feature_extraction == "ResnetFpn":
            self.feature_extraction = ResNetFPN()
        else:
            raise Exception('No FeatureExtraction module specified')
        feature_extraction_output = feature_output_channel
        self.sequence_output = 512
        if sequence_modeling == "BiLSTM":
            self.sequence_modeling = nn.Sequential(
                BidirectionalLSTM(feature_extraction_output, hidden_size, hidden_size),
                BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
            self.sequence_output = hidden_size
        elif sequence_modeling == "Bert":
            cfg = Config()
            cfg.dim = feature_output_channel
            cfg.dim_c = feature_output_channel
            cfg.max_vocab_size = batch_max_length
            cfg.len_alphabet = alphabet_size
            self.sequence_modeling = BertOcr(cfg)
        elif sequence_modeling == "SRN":
            self.sequence_modeling = TransformEncoder(n_layers=2, n_position=position_dim)
            self.sequence_output = 512
        else:
            self.sequence_modeling = None
            print('No SequenceModeling module specified')

        if prediction == "CTC":
            self.prediction = nn.Linear(self.sequence_output, alphabet_size) # 增加的1是占位符
        elif prediction == "Attn":
            self.prediction = Attention(self.sequence_output, hidden_size, alphabet_size)
        elif prediction == "Bert_pred":
            self.prediction = None
        elif prediction == "SRN":
            self.prediction = SrnDecoder(n_position=position_dim, N_max_character=batch_max_length,
                                         n_class=alphabet_size)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, x, text, is_train=True):
        if self.transformation is not None:
            x = self.transformation(x)
        visual_feature = self.feature_extraction(x)

        if self.sequence_modeling is not None:
            contextual_feature = self.sequence_modeling(visual_feature)
        else:
            contextual_feature = visual_feature

        if isinstance(contextual_feature, tuple):
            contextual_feature = contextual_feature[0]

        if self.prediction is not None:
            prediction = self.prediction(contextual_feature)
        else:
            prediction = contextual_feature

        return prediction

