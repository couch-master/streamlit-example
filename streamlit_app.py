import streamlit as st
from PIL import Image
import io
import sys
sys.path.append('thirdparty/AdaptiveWingLoss')
import os, glob
import numpy as np
#import cv2
import argparse
from src.approaches.train_image_translation import Image_translation_block
import torch
import pickle
import face_alignment
from src.autovc.AutoVC_mel_Convertor_retrain_version import AutoVC_mel_Convertor
import shutil
import util.utils as util
from scipy.signal import savgol_filter

from src.approaches.train_audio2landmark import Audio2landmark_model

default_head_name = 'dali'
ADD_NAIVE_EYE = True
CLOSE_INPUT_FACE_MOUTH = False

uploaded_file = st.file_uploader("Choose an image...", type=['jpg'])

parser = argparse.ArgumentParser()
parser.add_argument('--jpg', type=str, default='{}.jpg'.format(uploaded_file))
parser.add_argument('--close_input_face_mouth', default=CLOSE_INPUT_FACE_MOUTH, action='store_true')

parser.add_argument('--load_AUTOVC_name', type=str, default='examples/ckpt/ckpt_autovc.pth')
parser.add_argument('--load_a2l_G_name', type=str, default='examples/ckpt/ckpt_speaker_branch.pth')
parser.add_argument('--load_a2l_C_name', type=str, default='examples/ckpt/ckpt_content_branch.pth') #ckpt_audio2landmark_c.pth')
parser.add_argument('--load_G_name', type=str, default='examples/ckpt/ckpt_116_i2i_comb.pth') #ckpt_image2image.pth') #ckpt_i2i_finetune_150.pth') #c

parser.add_argument('--amp_lip_x', type=float, default=2.)
parser.add_argument('--amp_lip_y', type=float, default=2.)
parser.add_argument('--amp_pos', type=float, default=.5)
parser.add_argument('--reuse_train_emb_list', type=str, nargs='+', default=[]) #  ['iWeklsXc0H8']) #['45hn7-LXDX8']) #['E_kmpT-EfOg']) #'iWeklsXc0H8', '29k8RtSUjE0', '45hn7-LXDX8',
parser.add_argument('--add_audio_in', default=False, action='store_true')
parser.add_argument('--comb_fan_awing', default=False, action='store_true')
parser.add_argument('--output_folder', type=str, default='examples')

parser.add_argument('--test_end2end', default=True, action='store_true')
parser.add_argument('--dump_dir', type=str, default='', help='')
parser.add_argument('--pos_dim', default=7, type=int)
parser.add_argument('--use_prior_net', default=True, action='store_true')
parser.add_argument('--transformer_d_model', default=32, type=int)
parser.add_argument('--transformer_N', default=2, type=int)
parser.add_argument('--transformer_heads', default=2, type=int)
parser.add_argument('--spk_emb_enc_size', default=16, type=int)
parser.add_argument('--init_content_encoder', type=str, default='')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--reg_lr', type=float, default=1e-6, help='weight decay')
parser.add_argument('--write', default=False, action='store_true')
parser.add_argument('--segment_batch_size', type=int, default=1, help='batch size')
parser.add_argument('--emb_coef', default=3.0, type=float)
parser.add_argument('--lambda_laplacian_smooth_loss', default=1.0, type=float)
parser.add_argument('--use_11spk_only', default=False, action='store_true')

opt_parser = parser.parse_args()

''' STEP 1: preprocess input single image '''
img =cv2.imread('examples/' + opt_parser.jpg)
predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda', flip_input=True)
shapes = predictor.get_landmarks(img)
if (not shapes or len(shapes) != 1):
    print('Cannot detect face landmarks. Exit.')
    exit(-1)
shape_3d = shapes[0]

if(opt_parser.close_input_face_mouth):
    util.close_input_face_mouth(shape_3d)


''' Additional manual adjustment to input face landmarks (slimmer lips and wider eyes) '''
# shape_3d[48:, 0] = (shape_3d[48:, 0] - np.mean(shape_3d[48:, 0])) * 0.95 + np.mean(shape_3d[48:, 0])
shape_3d[49:54, 1] += 1.
shape_3d[55:60, 1] -= 1.
shape_3d[[37,38,43,44], 1] -=2
shape_3d[[40,41,46,47], 1] +=2


''' STEP 2: normalize face as input to audio branch '''
shape_3d, scale, shift = util.norm_input_face(shape_3d)


''' STEP 3: Generate audio data as input to audio branch '''
# audio real data
au_data = []
au_emb = []
ains = glob.glob1('examples', '*.wav')
ains = [item for item in ains if item is not 'tmp.wav']
ains.sort()
for ain in ains:
    os.system('ffmpeg -y -loglevel error -i examples/{} -ar 16000 examples/tmp.wav'.format(ain))
    shutil.copyfile('examples/tmp.wav', 'examples/{}'.format(ain))

    # au embedding
    from thirdparty.resemblyer_util.speaker_emb import get_spk_emb
    me, ae = get_spk_emb('examples/{}'.format(ain))
    au_emb.append(me.reshape(-1))

    print('Processing audio file', ain)
    c = AutoVC_mel_Convertor('examples')

    au_data_i = c.convert_single_wav_to_autovc_input(audio_filename=os.path.join('examples', ain),
           autovc_model_path=opt_parser.load_AUTOVC_name)
    au_data += au_data_i
if(os.path.isfile('examples/tmp.wav')):
    os.remove('examples/tmp.wav')

# landmark fake placeholder
fl_data = []
rot_tran, rot_quat, anchor_t_shape = [], [], []
for au, info in au_data:
    au_length = au.shape[0]
    fl = np.zeros(shape=(au_length, 68 * 3))
    fl_data.append((fl, info))
    rot_tran.append(np.zeros(shape=(au_length, 3, 4)))
    rot_quat.append(np.zeros(shape=(au_length, 4)))
    anchor_t_shape.append(np.zeros(shape=(au_length, 68 * 3)))

if(os.path.exists(os.path.join('examples', 'dump', 'random_val_fl.pickle'))):
    os.remove(os.path.join('examples', 'dump', 'random_val_fl.pickle'))
if(os.path.exists(os.path.join('examples', 'dump', 'random_val_fl_interp.pickle'))):
    os.remove(os.path.join('examples', 'dump', 'random_val_fl_interp.pickle'))
if(os.path.exists(os.path.join('examples', 'dump', 'random_val_au.pickle'))):
    os.remove(os.path.join('examples', 'dump', 'random_val_au.pickle'))
if (os.path.exists(os.path.join('examples', 'dump', 'random_val_gaze.pickle'))):
    os.remove(os.path.join('examples', 'dump', 'random_val_gaze.pickle'))

with open(os.path.join('examples', 'dump', 'random_val_fl.pickle'), 'wb') as fp:
    pickle.dump(fl_data, fp)
with open(os.path.join('examples', 'dump', 'random_val_au.pickle'), 'wb') as fp:
    pickle.dump(au_data, fp)
with open(os.path.join('examples', 'dump', 'random_val_gaze.pickle'), 'wb') as fp:
    gaze = {'rot_trans':rot_tran, 'rot_quat':rot_quat, 'anchor_t_shape':anchor_t_shape}
    pickle.dump(gaze, fp)


''' STEP 4: RUN audio->landmark network'''
model = Audio2landmark_model(opt_parser, jpg_shape=shape_3d)
if(len(opt_parser.reuse_train_emb_list) == 0):
    model.test(au_emb=au_emb)
else:
    model.test(au_emb=None)


''' STEP 5: de-normalize the output to the original image scale '''
fls = glob.glob1('examples', 'pred_fls_*.txt')
fls.sort()

for i in range(0,len(fls)):
    fl = np.loadtxt(os.path.join('examples', fls[i])).reshape((-1, 68,3))
    fl[:, :, 0:2] = -fl[:, :, 0:2]
    fl[:, :, 0:2] = fl[:, :, 0:2] / scale - shift

    if (ADD_NAIVE_EYE):
        fl = util.add_naive_eye(fl)

    # additional smooth
    fl = fl.reshape((-1, 204))
    fl[:, :48 * 3] = savgol_filter(fl[:, :48 * 3], 15, 3, axis=0)
    fl[:, 48*3:] = savgol_filter(fl[:, 48*3:], 5, 3, axis=0)
    fl = fl.reshape((-1, 68, 3))

    ''' STEP 6: Imag2image translation '''
    model = Image_translation_block(opt_parser, single_test=True)
    with torch.no_grad():
        model.single_test(jpg=img, fls=fl, filename=fls[i], prefix=opt_parser.jpg.split('.')[0])
        print('finish image2image gen')
    os.remove(os.path.join('examples', fls[i]))
# import streamlit as st
# import os
# import numpy as np
# import cv2
# import tempfile
# import shutil
# from src.approaches.train_image_translation import Image_translation_block
# from src.approaches.train_audio2landmark import Audio2landmark_model
# from src.autovc.AutoVC_mel_Convertor_retrain_version import AutoVC_mel_Convertor
# import util.utils as util
# from scipy.signal import savgol_filter
# import face_alignment
# import torch

# # Set up face alignment predictor
# predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda', flip_input=True)

# def main():
#     st.title('Audio-Visual Facial Animation')

#     # Upload image and audio file
#     uploaded_image = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
#     uploaded_audio = st.file_uploader("Choose an audio file...", type=['wav'])

#     if uploaded_image and uploaded_audio:
#         with tempfile.TemporaryDirectory() as temp_dir:
#             image_path = os.path.join(temp_dir, uploaded_image.name)
#             audio_path = os.path.join(temp_dir, uploaded_audio.name)

#             # Save uploaded files to temp directory
#             with open(image_path, "wb") as f:
#                 f.write(uploaded_image.getvalue())
#             with open(audio_path, "wb") as f:
#                 f.write(uploaded_audio.getvalue())

#             # Load and preprocess image
#             img = cv2.imread(image_path)
#             shapes = predictor.get_landmarks(img)
#             if not shapes or len(shapes) != 1:
#                 st.error('Cannot detect face landmarks. Please try another image.')
#             else:
#                 shape_3d = shapes[0]
#                 process_image_and_audio(image_path, audio_path, shape_3d, temp_dir)

# def process_image_and_audio(image_path, audio_path, shape_3d, temp_dir):
#     # Process audio to get embeddings and audio features
#     audio_features, audio_embeddings = process_audio(audio_path)

#     # Run the audio to landmark model
#     landmarks = run_audio_to_landmark_model(audio_features, audio_embeddings)

#     # Apply landmark adjustments and normalization
#     adjusted_landmarks = adjust_landmarks(landmarks, shape_3d)

#     # Generate facial animation from landmarks
#     animation_frames = generate_animation(adjusted_landmarks, image_path)

#     # Display the animation or the first frame for now
#     st.image(animation_frames[0], caption='Generated Animation Frame', use_column_width=True)
#     st.success('Processing complete!')

# def process_audio(audio_path):
#     # Placeholder: Implement audio processing
#     return "audio_features", "audio_embeddings"

# def run_audio_to_landmark_model(audio_features, audio_embeddings):
#     # Placeholder: Implement audio to landmark model processing
#     return "landmarks"

# def adjust_landmarks(landmarks, original_landmarks):
#     # Placeholder: Adjust landmarks based on the model's output
#     return landmarks

# def generate_animation(landmarks, image_path):
#     # Placeholder: Generate animation frames based on adjusted landmarks
#     return [cv2.imread(image_path)]  # This should return a list of images (frames)

# if __name__ == "__main__":
#     main()

# # import sys
# # sys.path.append('thirdparty/AdaptiveWingLoss')
# # import os, glob
# # import numpy as np
# # import cv2
# # import argparse
# # from src.approaches.train_image_translation import Image_translation_block
# # import torch
# # import pickle
# # import face_alignment
# # from src.autovc.AutoVC_mel_Convertor_retrain_version import AutoVC_mel_Convertor
# # import shutil
# # import util.utils as util
# # from scipy.signal import savgol_filter

# # from src.approaches.train_audio2landmark import Audio2landmark_model

# # default_head_name = 'dali'
# # ADD_NAIVE_EYE = True
# # CLOSE_INPUT_FACE_MOUTH = False


# # parser = argparse.ArgumentParser()
# # parser.add_argument('--jpg', type=str, default='{}.jpg'.format(default_head_name))
# # parser.add_argument('--close_input_face_mouth', default=CLOSE_INPUT_FACE_MOUTH, action='store_true')

# # parser.add_argument('--load_AUTOVC_name', type=str, default='examples/ckpt/ckpt_autovc.pth')
# # parser.add_argument('--load_a2l_G_name', type=str, default='examples/ckpt/ckpt_speaker_branch.pth')
# # parser.add_argument('--load_a2l_C_name', type=str, default='examples/ckpt/ckpt_content_branch.pth') #ckpt_audio2landmark_c.pth')
# # parser.add_argument('--load_G_name', type=str, default='examples/ckpt/ckpt_116_i2i_comb.pth') #ckpt_image2image.pth') #ckpt_i2i_finetune_150.pth') #c

# # parser.add_argument('--amp_lip_x', type=float, default=2.)
# # parser.add_argument('--amp_lip_y', type=float, default=2.)
# # parser.add_argument('--amp_pos', type=float, default=.5)
# # parser.add_argument('--reuse_train_emb_list', type=str, nargs='+', default=[]) #  ['iWeklsXc0H8']) #['45hn7-LXDX8']) #['E_kmpT-EfOg']) #'iWeklsXc0H8', '29k8RtSUjE0', '45hn7-LXDX8',
# # parser.add_argument('--add_audio_in', default=False, action='store_true')
# # parser.add_argument('--comb_fan_awing', default=False, action='store_true')
# # parser.add_argument('--output_folder', type=str, default='examples')

# # parser.add_argument('--test_end2end', default=True, action='store_true')
# # parser.add_argument('--dump_dir', type=str, default='', help='')
# # parser.add_argument('--pos_dim', default=7, type=int)
# # parser.add_argument('--use_prior_net', default=True, action='store_true')
# # parser.add_argument('--transformer_d_model', default=32, type=int)
# # parser.add_argument('--transformer_N', default=2, type=int)
# # parser.add_argument('--transformer_heads', default=2, type=int)
# # parser.add_argument('--spk_emb_enc_size', default=16, type=int)
# # parser.add_argument('--init_content_encoder', type=str, default='')
# # parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
# # parser.add_argument('--reg_lr', type=float, default=1e-6, help='weight decay')
# # parser.add_argument('--write', default=False, action='store_true')
# # parser.add_argument('--segment_batch_size', type=int, default=1, help='batch size')
# # parser.add_argument('--emb_coef', default=3.0, type=float)
# # parser.add_argument('--lambda_laplacian_smooth_loss', default=1.0, type=float)
# # parser.add_argument('--use_11spk_only', default=False, action='store_true')

# # opt_parser = parser.parse_args()

# # ''' STEP 1: preprocess input single image '''
# # img =cv2.imread('examples/' + opt_parser.jpg)
# # predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda', flip_input=True)
# # shapes = predictor.get_landmarks(img)
# # if (not shapes or len(shapes) != 1):
# #     print('Cannot detect face landmarks. Exit.')
# #     exit(-1)
# # shape_3d = shapes[0]

# # if(opt_parser.close_input_face_mouth):
# #     util.close_input_face_mouth(shape_3d)


# # ''' Additional manual adjustment to input face landmarks (slimmer lips and wider eyes) '''
# # # shape_3d[48:, 0] = (shape_3d[48:, 0] - np.mean(shape_3d[48:, 0])) * 0.95 + np.mean(shape_3d[48:, 0])
# # shape_3d[49:54, 1] += 1.
# # shape_3d[55:60, 1] -= 1.
# # shape_3d[[37,38,43,44], 1] -=2
# # shape_3d[[40,41,46,47], 1] +=2


# # ''' STEP 2: normalize face as input to audio branch '''
# # shape_3d, scale, shift = util.norm_input_face(shape_3d)


# # ''' STEP 3: Generate audio data as input to audio branch '''
# # # audio real data
# # au_data = []
# # au_emb = []
# # ains = glob.glob1('examples', '*.wav')
# # ains = [item for item in ains if item is not 'tmp.wav']
# # ains.sort()
# # for ain in ains:
# #     os.system('ffmpeg -y -loglevel error -i examples/{} -ar 16000 examples/tmp.wav'.format(ain))
# #     shutil.copyfile('examples/tmp.wav', 'examples/{}'.format(ain))

# #     # au embedding
# #     from thirdparty.resemblyer_util.speaker_emb import get_spk_emb
# #     me, ae = get_spk_emb('examples/{}'.format(ain))
# #     au_emb.append(me.reshape(-1))

# #     print('Processing audio file', ain)
# #     c = AutoVC_mel_Convertor('examples')

# #     au_data_i = c.convert_single_wav_to_autovc_input(audio_filename=os.path.join('examples', ain),
# #            autovc_model_path=opt_parser.load_AUTOVC_name)
# #     au_data += au_data_i
# # if(os.path.isfile('examples/tmp.wav')):
# #     os.remove('examples/tmp.wav')

# # # landmark fake placeholder
# # fl_data = []
# # rot_tran, rot_quat, anchor_t_shape = [], [], []
# # for au, info in au_data:
# #     au_length = au.shape[0]
# #     fl = np.zeros(shape=(au_length, 68 * 3))
# #     fl_data.append((fl, info))
# #     rot_tran.append(np.zeros(shape=(au_length, 3, 4)))
# #     rot_quat.append(np.zeros(shape=(au_length, 4)))
# #     anchor_t_shape.append(np.zeros(shape=(au_length, 68 * 3)))

# # if(os.path.exists(os.path.join('examples', 'dump', 'random_val_fl.pickle'))):
# #     os.remove(os.path.join('examples', 'dump', 'random_val_fl.pickle'))
# # if(os.path.exists(os.path.join('examples', 'dump', 'random_val_fl_interp.pickle'))):
# #     os.remove(os.path.join('examples', 'dump', 'random_val_fl_interp.pickle'))
# # if(os.path.exists(os.path.join('examples', 'dump', 'random_val_au.pickle'))):
# #     os.remove(os.path.join('examples', 'dump', 'random_val_au.pickle'))
# # if (os.path.exists(os.path.join('examples', 'dump', 'random_val_gaze.pickle'))):
# #     os.remove(os.path.join('examples', 'dump', 'random_val_gaze.pickle'))

# # with open(os.path.join('examples', 'dump', 'random_val_fl.pickle'), 'wb') as fp:
# #     pickle.dump(fl_data, fp)
# # with open(os.path.join('examples', 'dump', 'random_val_au.pickle'), 'wb') as fp:
# #     pickle.dump(au_data, fp)
# # with open(os.path.join('examples', 'dump', 'random_val_gaze.pickle'), 'wb') as fp:
# #     gaze = {'rot_trans':rot_tran, 'rot_quat':rot_quat, 'anchor_t_shape':anchor_t_shape}
# #     pickle.dump(gaze, fp)


# # ''' STEP 4: RUN audio->landmark network'''
# # model = Audio2landmark_model(opt_parser, jpg_shape=shape_3d)
# # if(len(opt_parser.reuse_train_emb_list) == 0):
# #     model.test(au_emb=au_emb)
# # else:
# #     model.test(au_emb=None)


# # ''' STEP 5: de-normalize the output to the original image scale '''
# # fls = glob.glob1('examples', 'pred_fls_*.txt')
# # fls.sort()

# # for i in range(0,len(fls)):
# #     fl = np.loadtxt(os.path.join('examples', fls[i])).reshape((-1, 68,3))
# #     fl[:, :, 0:2] = -fl[:, :, 0:2]
# #     fl[:, :, 0:2] = fl[:, :, 0:2] / scale - shift

# #     if (ADD_NAIVE_EYE):
# #         fl = util.add_naive_eye(fl)

# #     # additional smooth
# #     fl = fl.reshape((-1, 204))
# #     fl[:, :48 * 3] = savgol_filter(fl[:, :48 * 3], 15, 3, axis=0)
# #     fl[:, 48*3:] = savgol_filter(fl[:, 48*3:], 5, 3, axis=0)
# #     fl = fl.reshape((-1, 68, 3))

# #     ''' STEP 6: Imag2image translation '''
# #     model = Image_translation_block(opt_parser, single_test=True)
# #     with torch.no_grad():
# #         model.single_test(jpg=img, fls=fl, filename=fls[i], prefix=opt_parser.jpg.split('.')[0])
# #         print('finish image2image gen')
# #     os.remove(os.path.join('examples', fls[i]))
# # import altair as alt
# # import numpy as np
# # import pandas as pd
# # import streamlit as st

# # """
# # # Welcome to Streamlit!

# # Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
# # If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
# # forums](https://discuss.streamlit.io).

# # In the meantime, below is an example of what you can do with just a few lines of code:
# # """

# # num_points = st.slider("Number of points in spiral", 1, 10000, 1100)
# # num_turns = st.slider("Number of turns in spiral", 1, 300, 31)

# # indices = np.linspace(0, 1, num_points)
# # theta = 2 * np.pi * num_turns * indices
# # radius = indices

# # x = radius * np.cos(theta)
# # y = radius * np.sin(theta)

# # df = pd.DataFrame({
# #     "x": x,
# #     "y": y,
# #     "idx": indices,
# #     "rand": np.random.randn(num_points),
# # })

# # st.altair_chart(alt.Chart(df, height=700, width=700)
# #     .mark_point(filled=True)
# #     .encode(
# #         x=alt.X("x", axis=None),
# #         y=alt.Y("y", axis=None),
# #         color=alt.Color("idx", legend=None, scale=alt.Scale()),
# #         size=alt.Size("rand", legend=None, scale=alt.Scale(range=[1, 150])),
# #     ))
