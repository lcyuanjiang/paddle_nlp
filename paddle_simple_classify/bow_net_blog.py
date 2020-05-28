import paddle.fluid as fluid
def bow_net(data,
            seq_len,
            label,
            dict_dim,
            emb_dim=128,
            hid_dim=128,
            hid_dim2=96,
            class_dim=3,
            is_prediction=False):
    """
    Bow net
    """
    # embedding layer
    emb = fluid.embedding(input=data, size=[dict_dim, emb_dim])
    emb = fluid.layers.sequence_unpad(emb, length=seq_len)
    # bow layer
    bow = fluid.layers.sequence_pool(input=emb, pool_type='sum')
    bow_tanh = fluid.layers.tanh(bow)
    # full connect layer
    fc_1 = fluid.layers.fc(input=bow_tanh, size=hid_dim, act="tanh")
    fc_2 = fluid.layers.fc(input=fc_1, size=hid_dim2, act="tanh")
    # softmax layer
    prediction = fluid.layers.fc(input=[fc_2], size=class_dim, act="softmax")
    if is_prediction:
        return prediction
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    return avg_cost, prediction