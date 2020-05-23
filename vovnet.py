############################################################
#  VoVnet Graph
############################################################
VoVNet19_slim_dw_eSE = {
    'stem': [64, 64, 64],
    'stage_conv_ch': [64, 80, 96, 112],
    'stage_out_ch': [112, 256, 384, 512],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "eSE": True,
    "dw": True
}

VoVNet19_dw_eSE = {
    'stem': [64, 64, 64],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "eSE": True,
    "dw": True
}

VoVNet19_slim_eSE = {
    'stem': [64, 64, 128],
    'stage_conv_ch': [64, 80, 96, 112],
    'stage_out_ch': [112, 256, 384, 512],
    'layer_per_block': 3,
    'block_per_stage': [1, 1, 1, 1],
    'eSE': True,
    "dw": False
}

VoVNet19_eSE = {
    'stem': [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "eSE": True,
    "dw": False
}

VoVNet39_eSE = {
    'stem': [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "block_per_stage": [1, 1, 2, 2],
    "eSE": True,
    "dw": False
}

VoVNet57_eSE = {
    'stem': [64, 64, 128],  # the number of filter
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "block_per_stage": [1, 1, 4, 3],
    "eSE": True,
    "dw": False
}

VoVNet99_eSE = {
    'stem': [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "block_per_stage": [1, 3, 9, 3],
    "eSE": True,
    "dw": False
}

_STAGE_SPECS = {
    "V-19-slim-dw-eSE": VoVNet19_slim_dw_eSE,
    "V-19-dw-eSE": VoVNet19_dw_eSE,
    "V-19-slim-eSE": VoVNet19_slim_eSE,
    "V-19-eSE": VoVNet19_eSE,
    "V-39-eSE": VoVNet39_eSE,
    "V-57-eSE": VoVNet57_eSE,
    "V-99-eSE": VoVNet99_eSE,
}


def conv3x3(input_tensor, filters, module_name, postfix, stride=1, kernel_size=3):
    """
    通过这个函数，实现输出五个和输入相同尺寸的feature map
    filters由stage_ch控制
    """
    x = KL.Conv2D(filters, (kernel_size, kernel_size), strides=stride, padding='same', use_bias=False,
                  name="{}_{}/conv".format(module_name, postfix))(input_tensor)
    x = BatchNorm(name="{}_{}/norm".format(module_name, postfix))(x, training=False)
    x = KL.Activation('relu')(x)
    return x


def conv1x1(input_tensor, filters, module_name, postfix, stride=1, kernel_size=1):
    """
    通过这个函数，实现六个特征图级联结果降维
    filters由concat_ch控制
    """
    x = KL.Conv2D(filters, (kernel_size, kernel_size), strides=stride, padding='same', use_bias=False,
                  name="{}_{}/conv".format(module_name, postfix))(input_tensor)
    x = BatchNorm(name="{}_{}/norm".format(module_name, postfix))(x, training=False)
    x = KL.Activation('relu')(x)
    return x

# http://osask.cn/front/ask/view/590259
def Hsigmoid(x):
    return


def eSEModule(input_tensor,out_ch):
    x = KL.GlobalAveragePooling2D()(input_tensor)
    x = KL.Dense(out_ch, activation='relu', use_bias=False)(x)
    # x = Hsigmoid(x)
    return KL.Multiply()([input_tensor, x])


def _OSA_module(input_tensor, stage_ch, concat_ch, layer_per_block, module_name, SE=False, identity=False):
    """
    in_ch是输入图像(第一个块)的Dimension,stage_ch是剩下五个块的D。
    _OSA_module是一个OSA_stage的一个块
    输入是上一个阶段的输出，将进入Conv3x3
    输出是conv1x1的输出，eSE的输入
    """
    cat = input_tensor
    for i in range(layer_per_block):
        # 每层五个块（阶梯部分）
        x = conv3x3(input_tensor, stage_ch, module_name, i)
        # in_channel = stage_ch
        # 级联部分
        cat = KL.Concatenate(axis=3)([cat, x])
    # in_channel = in_ch+layer_per_block*stage_ch
    # x 是级联的输出
    xt = conv1x1(cat, concat_ch, module_name, "concat")
    identity_feat = input_tensor
    if SE:
        xt = eSEModule(xt,concat_ch)
    if identity:
        # xt = xt + identity_feat
        xt = KL.Add()([xt,identity_feat])
    return xt


def _OSA_stage(input_tensor, stage_num, concat_ch, block_per_stage, layer_per_block, SE=True):
    """
    :param input_tensor: 输入向量
            in_ch: 输入向量维度
    :param stage_num: OSA的第几阶段（总共4个阶段）
    :param concat_ch: conv1x1后channal数目
    :param block_pre_stage: 一个OSAstage有几个OSAmodule
    :param layer_per_block: 5
    :param SE: eSEmodule
    :return:
    """

    if not stage_num == 2:
        # self.add_module("Pooling", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        input_tensor = KL.MaxPooling2D()(input_tensor)
    if block_per_stage != 1:
        SE = False
    module_name = "OSA{}_1".format(stage_num)
    stage_ch = VoVNet57_eSE["stage_conv_ch"][stage_num - 2]
    x = _OSA_module(input_tensor, stage_ch, concat_ch, layer_per_block, module_name, SE)
    if block_per_stage < 2:
        return x
    if block_per_stage == 4:  # stage 4
        x = _OSA_module(x, stage_ch, concat_ch, layer_per_block, 'OSA4_2', SE=False, identity=True)
        x = _OSA_module(x, stage_ch, concat_ch, layer_per_block, 'OSA4_3', SE=False, identity=True)
        x = _OSA_module(x, stage_ch, concat_ch, layer_per_block, 'OSA4_4', SE=True, identity=True)
        x = _OSA_module(x, stage_ch, concat_ch, layer_per_block, 'OSA4_5', SE=False, identity=True)
        return x
    if block_per_stage == 3:  # stage 5
        x = _OSA_module(x, stage_ch, concat_ch, layer_per_block, 'OSA5_2', SE=False, identity=True)
        x = _OSA_module(x, stage_ch, concat_ch, layer_per_block, 'OSA5_3', SE=True, identity=True)
        x = _OSA_module(x, stage_ch, concat_ch, layer_per_block, 'OSA5_4', SE=False, identity=True)
        return x

    # for i in range(block_per_stage - 1):
    #     if i != block_per_stage - 2: 
    #         SE = False
    #     module_name = 'OSA{}_{}'.format(stage_num,i + 2)
    #     _OSA_module(x,stage_ch,concat_ch, layer_per_block, module_name, SE, identity=True)


def vovnet_graph(input_image, architecture, train_bn=False, SE=True):
    assert architecture in ["resnet50", "resnet101", "vovnet"]
    stage_specs = _STAGE_SPECS['V-57-eSE']

    stem_ch = stage_specs["stem"]  # 茎通道数(茎通道中filters的数量)
    # config_stage_ch = stage_specs["stage_conv_ch"] # 不同OSA_stage的OSA_module中那五个feature map的channal
    config_concat_ch = stage_specs["stage_out_ch"]  # 经过1x1卷积后的输出尺寸（conv1x1的filters）
    block_per_stage = stage_specs["block_per_stage"]  # 某个OSA_Stage有几个OSA_module
    layer_per_block = stage_specs["layer_per_block"]  # 有几个OSA_stage
    SE = stage_specs["eSE"]
    # Stem module 见 peleenet
    stem = conv3x3(input_image, stem_ch[0], "stem", "1", 2)  # input_channel, output_channel, model_name,postfix
    stem_1= conv3x3(stem, stem_ch[1], "stem", "2", 1)
    stem_1 = KL.Add()([stem,stem_1])
    stem = KL.Conv2D(stem_ch[2],(3,3),padding='same',use_bias=False)(stem_1)
    stem = BatchNorm()(stem,training = False)
    stem = KL.Activation('relu')(stem)
    stem_2= conv3x3(stem, stem_ch[2], "stem", "3", 2)
    stem = KL.Add()([stem,stem_2])
    C1 = stem
    # C1 = KL.Concatenate(axis=1)([stem_2, stem_3])
    # C1 = KL.Concatenate(axis=0)([stem, stem_2])
    # 输出是256*256*256 "stem_3/relu"
    # stage 2~5
    C2 = _OSA_stage(C1, 2, config_concat_ch[0], block_per_stage[0], layer_per_block, SE=True)
    C3 = _OSA_stage(C2, 3, config_concat_ch[1], block_per_stage[1], layer_per_block, SE=True)
    C4 = _OSA_stage(C3, 4, config_concat_ch[2], block_per_stage[2], layer_per_block, SE=False)
    C5 = _OSA_stage(C4, 5, config_concat_ch[3], block_per_stage[3], layer_per_block, SE=False)
    return [C1, C2, C3, C4, C5]
