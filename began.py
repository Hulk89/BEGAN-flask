import numpy as np
import imageio
import tensorflow as tf
import os
from tqdm import tqdm

def batch(file_list, batch_size=16):
    fileQ = tf.train.string_input_producer(file_list, shuffle=False)
    reader = tf.WholeFileReader()

    filename, data = reader.read(fileQ)
    image = tf.image.decode_jpeg(data, channels=3)

    img = imageio.imread(file_list[0])
    w, h, c = img.shape
    shape = [w, h, 3]

    image.set_shape(shape)
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size

    queue = tf.train.shuffle_batch(
        [image], batch_size=batch_size,
        num_threads=4, capacity=capacity,
        min_after_dequeue=min_after_dequeue, name='synthetic_inputs')

    crop_queue = tf.image.crop_to_bounding_box(queue, 50, 25, 128, 128)
    resized = tf.image.resize_nearest_neighbor(crop_queue, [64, 64])

    return tf.to_float(resized)


dir_ = './CelebA/images'
jpgs = os.listdir(dir_)
filelist = [os.path.join(dir_, jpg) for jpg in jpgs]
data_loader = batch(filelist)


def Encoder(images,  ## N x 32 x 32 x (3 or 1)
            n,
            h,
            reuse=False,
            stddev=0.02,
            normalize=True,
            gray=False):
    input_ = images
    if gray is True:
        channel = 1
    else:
        channel = 3
    filter_shapes = [[3, 3, channel, n],
                     [3, 3, n, n],
                     [3, 3, n, 2*n],
                     [3, 3, 2*n, 2*n],
                     [3, 3, 2*n, 3*n],
                     [3, 3, 3*n, 3*n],
                     [3, 3, 3*n, 4*n],
                     [3, 3, 4*n, 4*n],
                     [3, 3, 4*n, 4*n]]

    subsampling_layer = [2, 4, 6]
    subsampling_size = [[32, 32], [16, 16], [8, 8]]

    with tf.variable_scope("encoder", reuse=reuse) as scope:
        # Convolution
        for i, filter_shape in enumerate(filter_shapes):
            conv_weight = tf.get_variable("conv_weight_{}".format(i),
                                          filter_shape,
                                          initializer=tf.truncated_normal_initializer(stddev=stddev))
            res = tf.nn.conv2d(input_,
                               conv_weight,
                               [1, 1, 1, 1],
                               "SAME")
            if normalize:
                res = tf.contrib.layers.layer_norm(res)

            res = tf.nn.elu(res)

            if normalize and (tf.shape(input_) == tf.shape(res)):
                res = input_ + res

            if i in subsampling_layer:  # subsampling
                res = tf.image.resize_nearest_neighbor(res,
                                                       subsampling_size[subsampling_layer.index(i)])
            input_ = res
        '''
        16 x 8*8*3*n
        '''
        before_fnn = tf.contrib.layers.flatten(input_)
        '''
        16 x h
        '''
        after_fnn = tf.layers.dense(before_fnn, h)

        variables = tf.contrib.framework.get_variables(scope)

        return after_fnn, variables

def Decoder(encoded,  ## N x h
            n,
            h,
            name="D",
            reuse=False,
            stddev=0.02,
            normalize=True,
            gray=False):
    if gray is True:
        channel = 1
    else:
        channel = 3
        
    filter_shapes = [[3, 3, 4*n, 4*n],
                     [3, 3, 4*n, 4*n],
                     [3, 3, 4*n, 3*n],
                     [3, 3, 3*n, 3*n],
                     [3, 3, 3*n, 3*n],
                     [3, 3, 3*n, 2*n],
                     [3, 3, 2*n, 2*n],
                     [3, 3, 2*n, n],
                     [3, 3, n, n],
                     [3, 3, n, channel]]

    upscaling_layer = [1, 3, 5]
    upscaling_size = [[16, 16],[32, 32], [64, 64]]

    with tf.variable_scope("{}_decoder".format(name), reuse=reuse) as scope:
        # fnn
        fnn_res = tf.layers.dense(encoded, 8 * 8 * 4*n)
        #fnn_res = tf.nn.elu(fnn_res)
        input_ = tf.reshape(fnn_res, [-1, 8, 8, 4*n])
        # Convolution
        for i, filter_shape in enumerate(filter_shapes):
            conv_weight = tf.get_variable("conv_weight_{}".format(i),
                                          filter_shape,
                                          initializer=tf.truncated_normal_initializer(stddev=stddev))
            res = tf.nn.conv2d(input_,
                               conv_weight,
                               [1, 1, 1, 1],
                               "SAME")

            if i != len(filter_shapes) -1:  # 마지막에 elu가 끼면 안될 듯...
                if normalize:
                    res = tf.contrib.layers.layer_norm(res)
                res = tf.nn.elu(res)

            if i in upscaling_layer:  # subsampling
                res = tf.image.resize_nearest_neighbor(res,
                                                       upscaling_size[upscaling_layer.index(i)])
            if normalize:
                if tf.shape(input_) == tf.shape(res):
                    res = input_ + res

            input_ = res

        res = 2 * tf.sigmoid(res) - 1

        variables = tf.contrib.framework.get_variables(scope)

        return res, variables
# config
num_iter = 300000
B =        16
h =        32
n =        64
gamma_ =   0.5
model_folder = "/root/BEGAN_Celeb/model"


gamma = tf.constant(gamma_, dtype=tf.float32)
lambda_ = tf.Variable(0.001,
                      dtype=tf.float32,
                      trainable=False)
starter_learning_rate = 0.00001

k_initial = tf.constant(0, dtype=tf.float32)


global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(starter_learning_rate, 
                                global_step,
                                100,
                                0.98,
                                staircase=True)
lr_ = tf.Variable(lr,
                  dtype=tf.float32,
                  trainable=False)


real_images = data_loader / 127.5 - 1  # Real Image
z = tf.placeholder(tf.float32, (None, h), name="z")
k_prev = tf.placeholder(tf.float32, [])

def denorm_img(img):
    return tf.cast((img+1)*127.5, tf.uint8)

with tf.device("/gpu:0"):
    # Real
    latent_in_real, varE = Encoder(real_images, n, h)
    restored_real, varD = Decoder(latent_in_real, n, h, name="D")
    varDisc = varE + varD  # Discriminator의 variable을 가져와야지

    tf.summary.image("input_real",  denorm_img(real_images))
    tf.summary.image("output_real", denorm_img(restored_real))

    # fake
    fake_images, varGen = Decoder(z, n, h, name="G")
    latent_in_fake, _   = Encoder(fake_images, n, h, reuse=True)
    restored_fake, _    = Decoder(latent_in_fake, n, h, name="D", reuse=True)
    tf.summary.image("input_fake", denorm_img(fake_images))
    tf.summary.image("output_fake", denorm_img(restored_fake))

    # real loss
    L_x =  tf.reduce_mean(tf.abs(real_images - restored_real))
    tf.summary.scalar("Real Loss", L_x)
    # fake loss
    L_z = tf.reduce_mean(tf.abs(fake_images - restored_fake))
    tf.summary.scalar("Fake Loss", L_z)

    # Discriminator/Generator loss
    L_D = L_x - k_prev * L_z
    L_G = L_z

    tf.summary.scalar("Discriminator Loss", L_D)
    tf.summary.scalar("Generator Loss", L_G)

    # control?
    k_next = k_prev + lambda_*(gamma*L_x - L_z)
    tf.summary.scalar("curr_K", k_prev)
    tf.summary.scalar("next_K", k_next)

    # convergence measure
    M_global = L_x + tf.abs(gamma*L_x - L_z)
    tf.summary.scalar("approx_convergence_measure", M_global)
    summary = tf.summary.merge_all()

    # gradient descent
    opt_D = tf.train.AdamOptimizer(lr_)
    opt_G = tf.train.AdamOptimizer(lr_)

    # 주의! : loss에 따라 gradient를 적용할 variable들이 다르다!!
    train_op_D = opt_D.minimize(L_D, var_list=varDisc)
    train_op_G = opt_G.minimize(L_G, global_step, var_list=varGen)

saver = tf.train.Saver()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(model_folder,
                                         sess.graph)

    k_t_ = sess.run(k_initial)
    """
    queue runner 시작
    """
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    t = tqdm(range(num_iter), desc="training BEGAN")

    for epoch in t:
        #### fake_image ####
        Z = np.random.uniform(-1, 1, B * h)  # batch size가 달라도 된다.
        Z = np.reshape(Z, [B, h])

        r = sess.run([train_op_D,
                      train_op_G,
                      L_D,
                      L_G,
                      M_global,
                      k_next,
                      summary],
                     feed_dict={z:Z,
                                k_prev: min(max(k_t_, 0), 1)})

        _, _, loss_D, loss_G, M_, k_t_, summary_ = r
        t.set_postfix(loss_D=loss_D, loss_G=loss_G, M_global=M_, k_t=k_t_)
        if np.isnan(k_t_):
            break
        if epoch % 300 == 0:
            saver.save(sess, os.path.join(model_folder, "model.ckpt"))
            train_writer.add_summary(summary_, epoch)

    coord.request_stop()
    coord.join(threads)
    print("training done with {}-iteration.".format(num_iter), flush=True)
