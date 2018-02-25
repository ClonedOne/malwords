from sklearn.preprocessing import LabelBinarizer, normalize, StandardScaler
import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)


# Pre-Processing

def pp_vectors(xm_train, xm_dev, xm_test):
    """
    Takes as input the data matrices and averages each vector over the vector length.
    Then transpose it to feed it to the NN.

    :param xm_train: train matrix
    :param xm_dev: dev matrix
    :param xm_test: test matrix
    :return:
    """

    scaler = StandardScaler().fit(xm_train)

    train = scaler.transform(xm_train).T

    dev = scaler.transform(xm_dev).T

    test = scaler.transform(xm_test).T

    return train, dev, test


def pp_labels(y_train, y_dev, y_test):
    """
    Generates the one-hot representation of the labels of each sample.

    :param y_train: train labels
    :param y_dev: dev labels
    :param y_test: test labels
    :return:
    """

    lb = LabelBinarizer()
    ym_train = lb.fit_transform(y_train).T
    ym_dev = lb.fit_transform(y_dev).T
    ym_test = lb.fit_transform(y_test).T

    return ym_train, ym_dev, ym_test


def view_shapes(xm_train, xm_dev, xm_test, ym_train, ym_dev, ym_test):
    """
    Prints the shapes of the data sets.

    :param xm_train: train matrix
    :param xm_dev: dev matrix
    :param xm_test: test matrix
    :param ym_train: train labels
    :param ym_dev: dev labels
    :param ym_test: test labels
    :return:
    """

    print('X_train shape: ' + str(xm_train.shape))
    print('Y_train shape: ' + str(ym_train.shape))
    print('X_dev shape: ' + str(xm_dev.shape))
    print('Y_dev shape: ' + str(ym_dev.shape))
    print('X_test shape: ' + str(xm_test.shape))
    print('Y_test shape: ' + str(ym_test.shape))
    print('\n')


# Model definition

def init_ph(n_feats, n_classes):
    """
    Initializes placeholders for the input matrices.

    :param n_feats: number of features
    :param n_classes: number of classes
    :return:
    """

    x = tf.placeholder(dtype=tf.float32, shape=(n_feats, None))
    y = tf.placeholder(dtype=tf.float32, shape=(n_classes, None))
    keep_prob = tf.placeholder(tf.float32)

    return x, y, keep_prob


def init_weights(n_layers, layer_sizes):
    """
    Initializes weights using the Xavier algorithm.

    :param n_layers: number of layers
    :param layer_sizes: list of the dimensions of each layer
    :return:
    """

    params = {}

    for i in range(n_layers):
        wn = 'W{}'.format(i)
        bn = 'b{}'.format(i)

        params[wn] = tf.get_variable(
            wn,
            layer_sizes[i * 2],
            initializer=tf.contrib.layers.xavier_initializer(seed=42)
        )

        params[bn] = tf.get_variable(
            bn,
            layer_sizes[(i * 2) + 1],
            initializer=tf.zeros_initializer()
        )

    return params


def fwd(x, params, keep_prob, n_h_layers):
    """
    Forward propagation step.

    :param x: mini-batch data matrix
    :param params: dictionary of weights per layer
    :param keep_prob: retention probability for dropout normalization
    :param n_h_layers: number of hidden layers
    :return:
    """

    zn = None
    epsilon = 1e-4

    an = tf.nn.dropout(x, keep_prob)
    # an = x

    for i in range(n_h_layers):
        wn = 'W{}'.format(i)
        bn = 'b{}'.format(i)

        zn = tf.add(tf.matmul(params[wn], an), params[bn])

        batch_mean, batch_var = tf.nn.moments(zn, [0])
        b_n = tf.nn.batch_normalization(
            x=zn,
            mean=batch_mean,
            variance=batch_var,
            offset=None,
            scale=None,
            variance_epsilon=epsilon
        )

        # an = tf.nn.dropout(tf.nn.relu(b_n), keep_prob)
        an = tf.nn.dropout(tf.nn.leaky_relu(b_n), keep_prob)

    return zn


def compute_cost(zn, y, reg, params, n_layers):
    """
    Computes the cost function.

    :param zn: activation at final layer
    :param y: labels for mini-batch
    :param reg: regularization parameter
    :param params: dictionary of weights
    :param n_layers: number of layers
    :return:
    """

    logits = tf.transpose(zn)
    labels = tf.transpose(y)

    regularization = 0.0
    for i in range(n_layers):
        wn = 'W{}'.format(i)
        regularization += tf.nn.l2_loss(params[wn])

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)) + (
        reg * regularization)

    return cost


def dan(xm_train, ym_train, xm_dev, ym_dev, l_rate, n_epochs, m_b_size, n_h_layers, layers, k_prob, reg, costs, sess):
    """
    Builds and runs the deep averaging network model.

    :param xm_train: train matrix
    :param ym_train: train labels
    :param xm_dev: dev matrix
    :param ym_dev: dev labels
    :param l_rate: learning rate
    :param n_epochs: number of training epochs
    :param m_b_size: mini-batch size
    :param n_h_layers: number of hidden layers
    :param layers: list with hidden layers sizes
    :param k_prob: dropout keep probability
    :param reg: regularization factor
    :param costs: list to fill with costs
    :param sess: current tensorflow session
    :return:
    """

    x, y, keep_prob = init_ph(xm_train.shape[0], ym_train.shape[0])

    params = init_weights(n_h_layers, layers)

    z = fwd(x, params, keep_prob, n_h_layers)

    cost = compute_cost(z, y, reg, params, n_h_layers)

    global_step = tf.Variable(0, trainable=False)

    learning_rate = tf.train.exponential_decay(l_rate, global_step, 1000, 0.90)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

    y_pred = tf.argmax(z)

    y_true = tf.argmax(y)

    correct_prediction = tf.equal(y_pred, y_true)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    init = tf.global_variables_initializer()

    num_minibatches = int(xm_train.shape[1] / m_b_size)

    sess.run(init)

    for epoch in range(n_epochs):
        epoch_cost = 0.

        minibatch_idxs = np.random.permutation(xm_train.shape[1])

        for i in range(num_minibatches):
            minibatch_x = np.take(
                xm_train,
                minibatch_idxs[i * m_b_size: (i + 1) * m_b_size],
                axis=1
            )

            # minibatch_x *= np.random.binomial([np.ones(minibatch_x.shape)],  k_prob)[0] * (1.0 / k_prob)

            minibatch_y = np.take(
                ym_train,
                minibatch_idxs[i * m_b_size: (i + 1) * m_b_size],
                axis=1
            )

            _, minibatch_cost = sess.run(
                [optimizer, cost],
                feed_dict={
                    x: minibatch_x,
                    y: minibatch_y,
                    keep_prob: k_prob
                }
            )

            epoch_cost += minibatch_cost / num_minibatches

        if epoch % 100 == 0:
            print('Cost after epoch %i: %f' % (epoch, epoch_cost))
            print('Train Accuracy:', accuracy.eval({x: xm_train, y: ym_train, keep_prob: 1.0}, session=sess))
            print('Dev Accuracy:', accuracy.eval({x: xm_dev, y: ym_dev, keep_prob: 1.0}, session=sess))
            print('Learning Rate:', learning_rate.eval(session=sess))
            print('')

        if epoch % 5 == 0:
            costs.append(epoch_cost)

    tr_acc = accuracy.eval({x: xm_train, y: ym_train, keep_prob: 1.0}, session=sess)
    dv_acc = accuracy.eval({x: xm_dev, y: ym_dev, keep_prob: 1.0}, session=sess)

    print('Final epoch')
    print('Train Accuracy:', tr_acc)
    print('Dev Accuracy:', dv_acc)

    return accuracy, tr_acc, dv_acc, x, y, keep_prob, y_pred, y_true


# noinspection PyUnusedLocal
def classify(xm_train, xm_dev, xm_test, y_train, y_dev, y_test, config, params):
    """
    Classify the documents using the deep averaging network and the AVClass labels as base truth.

    :param xm_train: Training data matrix
    :param xm_dev: Development data matrix
    :param xm_test: Testing data matrix
    :param y_train: List of train set labels
    :param y_dev: List of dev set labels
    :param y_test: List of test set labels
    :param config: Global configuration dictionary
    :param params: Dictionary of parameters for the algorithm
    :return: Predicted test labels and trained model
    """

    xm_train, xm_dev, xm_test = pp_vectors(xm_train, xm_dev, xm_test)
    ym_train, ym_dev, ym_test = pp_labels(y_train, y_dev, y_test)
    view_shapes(xm_train, xm_dev, xm_test, ym_train, ym_dev, ym_test)

    costs = []

    # Hyper-parameters
    learning_rate = params.get('learning_rate', 0.01)
    n_epochs = params.get('num_epochs', 1000)
    mini_batch_size = params.get('batch_size', 256)
    # n_h_layers = params.get('num_layers', 6)
    n_h_layers = params.get('num_layers', 5)
    ls = [
        [2048, xm_train.shape[0]], [2048, 1],
        [2048, 2048], [2048, 1],
        [2048, 2048], [2048, 1],
        [2048, 2048], [2048, 1],
        # [1024, 2048], [1024, 1],
        # [512, 1024], [512, 1],
        # [256, 512], [256, 1],
        # [ym_train.shape[0], 256], [ym_train.shape[0], 1]
        [ym_train.shape[0], 2048], [ym_train.shape[0], 1]
    ]
    layers = params.get('layers', ls)
    keep_prob = params.get('keep_prob', 0.9)
    reg = params.get('regularization', 0.0)

    tf.reset_default_graph()
    sess = tf.Session()

    accuracy, tr_acc, dv_acc, x, y, keep_prob, y_pred, y_true = dan(
        xm_train,
        ym_train,
        xm_dev,
        ym_dev,
        learning_rate,
        n_epochs,
        mini_batch_size,
        n_h_layers,
        layers,
        keep_prob,
        reg,
        costs,
        sess
    )

    ts_acc, y_predicted, y_true = sess.run(
        [accuracy, y_pred, y_true],
        feed_dict={x: xm_test, y: ym_test, keep_prob: 1.0}
    )

    print('\nFinal accuracy values:\n')
    print('Train Accuracy:', tr_acc)
    print('Dev Accuracy:', dv_acc)
    print('Test Accuracy:', ts_acc)
    print('\n')

    return y_predicted, (y_true, costs), str(n_h_layers)
