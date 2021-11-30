import os
import numpy as np
import argparse
import time
import librosa
import soundfile as sf
from preprocess import *
from model import CycleGAN
import tensorflow as tf
import pyworld as pw

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True 
sess = tf.Session(config=config)

def train(train_A_dir, train_B_dir, model_dir, model_name, random_seed, validation_A_dir, validation_B_dir, output_dir, tensorboard_log_dir, n_frames):

    np.random.seed(random_seed)

    num_epochs = 500
    mini_batch_size = 1 # mini_batch_size = 1 is better
    generator_learning_rate = 0.0002
    generator_learning_rate_decay = generator_learning_rate / 5000000
    discriminator_learning_rate = 0.0001
    discriminator_learning_rate_decay = discriminator_learning_rate / 5000000
    sampling_rate = 24000
    num_mcep = 24
    frame_period = 5.0
#     n_frames = 128
    lambda_cycle = 10
    lambda_identity = 5

    print('Preprocessing Data...')

    start_time = time.time()

    wavs_A = load_wavs(wav_dir = train_A_dir, sr = sampling_rate)
    wavs_B = load_wavs(wav_dir = train_B_dir, sr = sampling_rate)

    f0s_A, timeaxes_A, sps_A, aps_A, coded_sps_A = world_encode_data(wavs = wavs_A, fs = sampling_rate, frame_period = frame_period, coded_dim = num_mcep)
    f0s_B, timeaxes_B, sps_B, aps_B, coded_sps_B = world_encode_data(wavs = wavs_B, fs = sampling_rate, frame_period = frame_period, coded_dim = num_mcep)

    log_f0s_mean_A, log_f0s_std_A = logf0_statistics(f0s_A)
    log_f0s_mean_B, log_f0s_std_B = logf0_statistics(f0s_B)

    print('Log Pitch A')
    print('Mean: %f, Std: %f' %(log_f0s_mean_A, log_f0s_std_A))
    print('Log Pitch B')
    print('Mean: %f, Std: %f' %(log_f0s_mean_B, log_f0s_std_B))


    coded_sps_A_transposed = transpose_in_list(lst = coded_sps_A)
    coded_sps_B_transposed = transpose_in_list(lst = coded_sps_B)

    coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = coded_sps_normalization_fit_transoform(coded_sps = coded_sps_A_transposed)
    print("Input data fixed.")
    coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std = coded_sps_normalization_fit_transoform(coded_sps = coded_sps_B_transposed)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    np.savez(os.path.join(model_dir, 'logf0s_normalization.npz'), mean_A = log_f0s_mean_A, std_A = log_f0s_std_A, mean_B = log_f0s_mean_B, std_B = log_f0s_std_B)
    np.savez(os.path.join(model_dir, 'mcep_normalization.npz'), mean_A = coded_sps_A_mean, std_A = coded_sps_A_std, mean_B = coded_sps_B_mean, std_B = coded_sps_B_std)

    if validation_A_dir is not None:
        validation_A_output_dir = os.path.join(output_dir, 'converted_A'+str(n_frames))
        if not os.path.exists(validation_A_output_dir):
            os.makedirs(validation_A_output_dir)

#     if validation_B_dir is not None:
#         validation_B_output_dir = os.path.join(output_dir, 'converted_B'+str(n_frames))
#         if not os.path.exists(validation_B_output_dir):
#             os.makedirs(validation_B_output_dir)

    end_time = time.time()
    time_elapsed = end_time - start_time

    print('Preprocessing Done.')

    print('Time Elapsed for Data Preprocessing: %02d:%02d:%02d' % (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))

    model = CycleGAN(num_features = num_mcep, log_dir = tensorboard_log_dir+str(n_frames))
    if n_frames!=128:
        model.load(model_dir+model_name)

    for epoch in range(num_epochs):
        print('Epoch: %d' % epoch)

        start_time_epoch = time.time()

        dataset_A, dataset_B = sample_train_data(dataset_A = coded_sps_A_norm, dataset_B = coded_sps_B_norm, n_frames = n_frames)

        n_samples = dataset_A.shape[0]

        for i in range(n_samples // mini_batch_size):

            num_iterations = n_samples // mini_batch_size * epoch + i

            if num_iterations > 100000:
                lambda_identity = 0.5
            if num_iterations > 100000:
                generator_learning_rate = max(0.00001, generator_learning_rate - generator_learning_rate_decay)
                discriminator_learning_rate = max(0.00001, discriminator_learning_rate - discriminator_learning_rate_decay)

            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size

            generator_loss, discriminator_loss = model.train(input_A = dataset_A[start:end], input_B = dataset_B[start:end], lambda_cycle = lambda_cycle, lambda_identity = lambda_identity, generator_learning_rate = generator_learning_rate, discriminator_learning_rate = discriminator_learning_rate)

            if i % 200 == 0:
                print('Iteration: {:07d}, Generator Learning Rate: {:.7f}, Discriminator Learning Rate: {:.7f}, Generator Loss : {:.6f}, Discriminator Loss : {:.6f}'.format(num_iterations, generator_learning_rate, discriminator_learning_rate, generator_loss, discriminator_loss))

        model.save(directory = model_dir, filename = str(n_frames)+model_name)

        end_time_epoch = time.time()
        time_elapsed_epoch = end_time_epoch - start_time_epoch

        print('Time Elapsed for This Epoch: %02d:%02d:%02d' % (time_elapsed_epoch // 3600, (time_elapsed_epoch % 3600 // 60), (time_elapsed_epoch % 60 // 1)))

        if validation_A_dir is not None:
            if epoch % 5 == 0:
                print('Generating Validation Data B from A...')
                for file in os.listdir(validation_A_dir):
                    filepath = os.path.join(validation_A_dir, file)
                    wav, _ = librosa.load(filepath, sr = sampling_rate, mono = True)
                    wav = wav_padding(wav = wav, sr = sampling_rate, frame_period = frame_period, multiple = 4)
                    f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)
                    f0_converted = pitch_conversion(f0 = f0, mean_log_src = log_f0s_mean_A, std_log_src = log_f0s_std_A, mean_log_target = log_f0s_mean_B, std_log_target = log_f0s_std_B)
                    coded_sp = world_encode_spectral_envelop(sp = sp, fs = sampling_rate, dim = num_mcep)
                    coded_sp_transposed = coded_sp.T
                    coded_sp_norm = (coded_sp_transposed - coded_sps_A_mean) / coded_sps_A_std
                    coded_sp_converted_norm = model.test(inputs = np.array([coded_sp_norm]), direction = 'A2B')[0]
                    coded_sp_converted = coded_sp_converted_norm * coded_sps_B_std + coded_sps_B_mean
                    coded_sp_converted = coded_sp_converted.T
                    coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                    decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = sampling_rate)
                    wav_transformed = world_speech_synthesis(f0 = f0_converted, decoded_sp = decoded_sp_converted, ap = ap, fs = sampling_rate, frame_period = frame_period)
                    wav_transformed = np.nan_to_num(wav_transformed)
                    sf.write(os.path.join(validation_A_output_dir, str(epoch)+'_'+os.path.basename(file)), wav_transformed, sampling_rate)
#                     file_name = file.split('.')[0]+'.wav'
#                     validation_A_output_dir = validation_A_output_dir+'/'+str(epoch)+'/'
#                     if not os.path.exists(validation_A_output_dir):
#                         os.makedirs(validation_A_output_dir)
            
#                     sf.write(os.path.join(validation_A_output_dir, os.path.basename(file_name)), wav_transformed, sampling_rate)
                    
#                     if train_B_dir.split('/')[-2]=='anger':

#                         x, fs = sf.read(os.path.join(validation_A_output_dir, os.path.basename(file)), sampling_rate)
#                         _f0, t = pw.dio(x, fs, f0_floor=100.0, f0_ceil=800.0,
#                                             channels_in_octave=2,
#                                             frame_period=5.2,
#                                             speed=1)
#                         _f0 = _f0*0.98
#                         _sp = pw.cheaptrick(x, _f0, t, fs)
#                         _ap = pw.d4c(x, _f0, t, fs)
#                         lent = int(0.65*_sp.shape[0])
#                         print('_sp', _sp.shape)
#                         print('lent', lent)
#                         _y1 = pw.synthesize(_f0[:lent], _sp[:lent]*1.5, _ap[:lent], fs, 5)
#                         _y2 = pw.synthesize(_f0[lent:], _sp[lent:]*1.5, _ap[lent:], fs, 4.5)
# #                         _y = np.concatenate((_y1, _y2), 0)*2
#                         _y = np.asarray(list(_y1)+list(_y2))
#                         print('_y', len(_y))
#                         sf.write(os.path.join(validation_A_output_dir, 'pw_'+os.path.basename(file)), _y, sampling_rate)
                        
#         if validation_B_dir is not None:
#             if epoch % 5 == 0:
#                 print('Generating Validation Data A from B...')
#                 for file in os.listdir(validation_B_dir):
#                     filepath = os.path.join(validation_B_dir, file)
#                     wav, _ = librosa.load(filepath, sr = sampling_rate, mono = True)
#                     wav = wav_padding(wav = wav, sr = sampling_rate, frame_period = frame_period, multiple = 4)
#                     f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)
#                     f0_converted = pitch_conversion(f0 = f0, mean_log_src = log_f0s_mean_B, std_log_src = log_f0s_std_B, mean_log_target = log_f0s_mean_A, std_log_target = log_f0s_std_A)
#                     coded_sp = world_encode_spectral_envelop(sp = sp, fs = sampling_rate, dim = num_mcep)
#                     coded_sp_transposed = coded_sp.T
#                     coded_sp_norm = (coded_sp_transposed - coded_sps_B_mean) / coded_sps_B_std
#                     coded_sp_converted_norm = model.test(inputs = np.array([coded_sp_norm]), direction = 'B2A')[0]
#                     coded_sp_converted = coded_sp_converted_norm * coded_sps_A_std + coded_sps_A_mean
#                     coded_sp_converted = coded_sp_converted.T
#                     coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
#                     decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = sampling_rate)
#                     wav_transformed = world_speech_synthesis(f0 = f0_converted, decoded_sp = decoded_sp_converted, ap = ap, fs = sampling_rate, frame_period = frame_period)
#                     wav_transformed = np.nan_to_num(wav_transformed)
# #                     librosa.output.write_wav(os.path.join(validation_B_output_dir, str(epoch)+'_'+os.path.basename(file)), wav_transformed, sampling_rate)
# #                     sf.write(os.path.join(validation_B_output_dir, str(epoch)+'_'+os.path.basename(file)), wav_transformed, sampling_rate)
#                     sf.write(os.path.join(validation_B_output_dir, os.path.basename(file)), wav_transformed, sampling_rate)

    model.save(directory = model_dir, filename = model_name)
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Train CycleGAN model for datasets.')

    train_A_dir_default = './data/training/NEUTRAL'
    train_B_dir_default = './data/training/SURPRISE'
    model_dir_default = './model/neutral_to_suprise_mceps'
    model_name_default = 'neutral_to_suprise_mceps.ckpt'
    random_seed_default = 0
    validation_A_dir_default = './data/evaluation_all/NEUTRAL'
    validation_B_dir_default = './data/evaluation_all/SURPRISE'
    output_dir_default = './validation_output'
    tensorboard_log_dir_default = './log'
    
    
    parser.add_argument('--num_f', type = int, help = 'frame length.')
    parser.add_argument('--train_A_dir', type = str, help = 'Directory for A.', default = train_A_dir_default)
    parser.add_argument('--train_B_dir', type = str, help = 'Directory for B.', default = train_B_dir_default)
    parser.add_argument('--model_dir', type = str, help = 'Directory for saving models.', default = model_dir_default)
    parser.add_argument('--model_name', type = str, help = 'File name for saving model.', default = model_name_default)
    parser.add_argument('--random_seed', type = int, help = 'Random seed for model training.', default = random_seed_default)
    parser.add_argument('--validation_A_dir', type = str, help = 'Convert validation A after each training epoch. If set none, no conversion would be done during the training.', default = validation_A_dir_default)
    parser.add_argument('--validation_B_dir', type = str, help = 'Convert validation B after each training epoch. If set none, no conversion would be done during the training.', default = validation_B_dir_default)
    parser.add_argument('--output_dir', type = str, help = 'Output directory for converted validation voices.', default = output_dir_default)
    parser.add_argument('--tensorboard_log_dir', type = str, help = 'TensorBoard log directory.', default = tensorboard_log_dir_default)

    argv = parser.parse_args()

    num_f = argv.num_f
    print('frame length:',num_f)
    train_A_dir = argv.train_A_dir
    train_B_dir = argv.train_B_dir
    model_dir = argv.model_dir
#     model_name = str(num_f)+'_'+argv.model_name
    model_name = argv.model_name
    random_seed = argv.random_seed
    validation_A_dir = None if argv.validation_A_dir == 'None' or argv.validation_A_dir == 'none' else argv.validation_A_dir
    validation_B_dir = None if argv.validation_B_dir == 'None' or argv.validation_B_dir == 'none' else argv.validation_B_dir
    output_dir = argv.output_dir
    tensorboard_log_dir = argv.tensorboard_log_dir

    train(train_A_dir = train_A_dir, train_B_dir = train_B_dir, model_dir = model_dir, model_name = model_name, random_seed = random_seed, validation_A_dir = validation_A_dir, validation_B_dir = validation_B_dir, output_dir = output_dir, tensorboard_log_dir = tensorboard_log_dir, n_frames = num_f)
