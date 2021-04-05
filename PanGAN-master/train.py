import tensorflow as tf
import os
import time
from PanGan import PanGan
from DataSet import DataSet
from config import FLAGES

def print_current_training_stats(error_pan_model, error_ms_model, error_g_model, global_step, learning_rate, time_elapsed):
    stats = 'Step: {}/{} ----- Cur_lr: {:1.7f} ----- Time: {:>2.2f} sec.'.format(global_step, FLAGES.iters,
                                                                                 learning_rate, time_elapsed)
    losses =  ' | spatial loss: {}'.format(error_pan_model)
    losses += ' | spectrual loss: {}'.format(error_ms_model)
    losses += ' | generator loss: {}'.format(error_g_model)
    print(stats)
    print(losses + '\n')
    
def print_current_training_stats_valid(error_spatial, error_spectrual, global_step, learning_rate, time_elapsed):
    stats = 'Valid_Step: {}/{} ----- Cur_lr: {:1.7f} ----- Time: {:>2.2f} sec.'.format(global_step, FLAGES.iters,
                                                                                 learning_rate, time_elapsed)
    losses =  ' | spatial error: {}'.format(error_spatial)
    losses += ' | spectrual error: {}'.format(error_spectrual)
    print(stats)
    print(losses + '\n')

def main(argv):
    model=PanGan(FLAGES.pan_size, FLAGES.ms_size, FLAGES.batch_size, FLAGES.num_spectrum, FLAGES.ratio,
                 FLAGES.lr,FLAGES.decay_rate,FLAGES.decay_step,is_training=True)
    model.train()
    dataset=DataSet(FLAGES.pan_size, FLAGES.ms_size, FLAGES.img_path, FLAGES.data_path, FLAGES.batch_size,
                    FLAGES.stride)
    DataGenerator=dataset.data_generator
    
    dataset_valid=DataSet(FLAGES.pan_size, FLAGES.ms_size, FLAGES.img_path, FLAGES.data_path, FLAGES.batch_size,
                    FLAGES.stride, 'valid')
    DataGenerator_valid=dataset_valid.data_generator

    merge_summary=tf.summary.merge_all()
    if not os.path.exists(FLAGES.log_dir):
        os.makedirs(FLAGES.log_dir)
    if not os.path.exists(FLAGES.model_save_dir):
        os.makedirs(FLAGES.model_save_dir)

    with tf.Session() as sess:
        train_writer=tf.summary.FileWriter(FLAGES.log_dir, sess.graph)
        saver=tf.train.Saver(max_to_keep=None)
        saver_g=tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Pan_model'),max_to_keep=None)
        init_op=tf.global_variables_initializer()
        sess.run(init_op)

        if FLAGES.is_pretrained:
            saver.restore(sess, './model/qk/PanNet-107000')
        for training_itr in range(FLAGES.iters):
            t1 = time.time()
            pan_batch, ms_batch=next(DataGenerator)
            for i in range(2):
                _, error_pan_model = sess.run([model.train_spatial_discrim, model.spatial_loss],feed_dict={model.pan_img: pan_batch, model.ms_img: ms_batch})
                _, error_ms_model = sess.run([model.train_spectrum_discrim, model.spectrum_loss],feed_dict={model.pan_img: pan_batch, model.ms_img: ms_batch })

            _, error_g_model, global_step, summary, learning_rate = sess.run([model.train_Pan_model, model.g_loss,model.global_step, merge_summary,model.learning_rate],
                                                                             feed_dict={model.pan_img: pan_batch, model.ms_img: ms_batch})
            error_pan_model=sess.run(model.spatial_loss,feed_dict={model.pan_img: pan_batch, model.ms_img: ms_batch})
            error_ms_model=sess.run(model.spectrum_loss,feed_dict={model.pan_img: pan_batch, model.ms_img: ms_batch})					
            print_current_training_stats(error_pan_model, error_ms_model, error_g_model, global_step, learning_rate, time.time()-t1)
            train_writer.add_summary(summary, global_step)
            
            # if (global_step + 1) %  FLAGES.valid_iters == 0:
                # t1 = time.time()
                # pan_valid_batch, ms_valid_batch=next(DataGenerator_valid)
                # error_spatial,error_spectrum = sess.run([model.valid_spatital_error, model.valid_spectrum_error],feed_dict={model.pan_img: pan_valid_batch, model.ms_img: ms_valid_batch})
                # print_current_training_stats_valid(error_spatial, error_spectrum, global_step, learning_rate, time.time()-t1)
            
            if (global_step + 1) %  FLAGES.model_save_iters == 0:
                saver.save(sess=sess, save_path=FLAGES.model_save_dir + '/' + 'PanNet', global_step=(global_step+1) )
                saver_g.save(sess=sess, save_path=FLAGES.model_save_dir + '/' + 'Generator', global_step=(global_step+1)  )
                print('\nModel checkpoint saved...\n')

            if global_step == FLAGES.iters:
                break
        print('Training done.')

if __name__ == '__main__':
    tf.app.run()



