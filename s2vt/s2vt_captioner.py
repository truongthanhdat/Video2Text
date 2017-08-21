DEVICE_ID = 0

""" Script to generate captions from video features"""

from collections import OrderedDict
import math
import numpy as np
import caffe
from framefc7_text_to_hdf5_data import *

def vocab_inds_to_sentence(vocab, inds):
  sentence = ' '.join([vocab[i] for i in inds])
  # Capitalize first character.
  sentence = sentence[0].upper() + sentence[1:]
  # Replace <EOS> with '.', or append '...'.
  if sentence.endswith(' ' + vocab[0]):
    sentence = sentence[:-(len(vocab[0]) + 1)] + '.'
  else:
    sentence += '...'
  return sentence

def video_to_descriptor(video_id, fsg):
  video_features = []
  assert video_id in fsg.vid_framefeats
  all_frames_fc7 = fsg.vid_framefeats[video_id]
  for frame_fc7 in all_frames_fc7:
    frame_fc7 = fsg.float_line_to_stream(frame_fc7)
    video_features.append(np.array(frame_fc7).reshape(1, len(frame_fc7)))
  return video_features

def encode_video_frames(net, video_features, previous_word=-1):
  for frame_feature in video_features:
    cont_input = 0 if previous_word == -1 else 1
    previous_word = 0
    cont = np.array([cont_input])
    data_en = np.array([previous_word])
    stage_ind = np.array([0]) # encoding stage
    image_features = np.zeros_like(net.blobs['frames_fc7'].data)
    image_features[:] = frame_feature
    net.forward(frames_fc7=image_features, cont_sentence=cont, input_sentence=data_en,
       stage_indicator=stage_ind)

def predict_single_word(net, pad_img_feature, previous_word, output='probs'):
  cont_input = 1
  cont = np.array([cont_input])
  data_en = np.array([previous_word])
  stage_ind = np.array([1]) # decoding stage
  image_features = np.zeros_like(net.blobs['frames_fc7'].data)
  image_features[:] = pad_img_feature
  net.forward(frames_fc7=image_features, cont_sentence=cont, input_sentence=data_en,
     stage_indicator=stage_ind)
  output_preds = net.blobs[output].data.reshape(-1)
  return output_preds

def predict_single_word_from_all_previous(net, pad_img_feature, previous_words):
  probs = predict_single_word(net, pad_img_feature, 0)
  for index, word in enumerate(previous_words):
    probs = predict_single_word(net, pad_img_feature, word)
  return probs


def predict_image_caption(net, pad_img_feature, vocab_list, strategy, max_length=50):
  # Note: This code support S2VT only for beam-width 1.
  beam_size = 1
  beams = [[]]
  beams_complete = 0
  beam_probs = [[]]
  beam_log_probs = [0.]
  while beams_complete < len(beams):
    expansions = []
    for beam_index, beam_log_prob, beam in \
        zip(range(len(beams)), beam_log_probs, beams):
      if beam:
        previous_word = beam[-1]
        if len(beam) >= max_length or previous_word == 0:
          exp = {'prefix_beam_index': beam_index, 'extension': [],
                 'prob_extension': [], 'log_prob': beam_log_prob}
          expansions.append(exp)
          # Don't expand this beam; it was already ended with an EOS,
          # or is the max length.
          continue
      else:
        previous_word = 0  # EOS is first word
      if beam_size == 1:
        probs = predict_single_word(net, pad_img_feature, previous_word)
      else:
        probs = predict_single_word_from_all_previous(net, pad_img_feature, beam)
      assert len(probs.shape) == 1
      assert probs.shape[0] == len(vocab_list)
      expansion_inds = probs.argsort()[-beam_size:]
      for ind in expansion_inds:
        prob = probs[ind]
        extended_beam_log_prob = beam_log_prob + math.log(prob)
        exp = {'prefix_beam_index': beam_index, 'extension': [ind],
               'prob_extension': [prob], 'log_prob': extended_beam_log_prob}
        expansions.append(exp)
    # Sort expansions in decreasing order of probabilitf.
    expansions.sort(key=lambda expansion: -1 * expansion['log_prob'])
    expansions = expansions[:beam_size]
    new_beams = \
        [beams[e['prefix_beam_index']] + e['extension'] for e in expansions]
    new_beam_probs = \
        [beam_probs[e['prefix_beam_index']] + e['prob_extension'] for e in expansions]
    beam_log_probs = [e['log_prob'] for e in expansions]
    beams_complete = 0
    for beam in new_beams:
      if beam[-1] == 0 or len(beam) >= max_length: beams_complete += 1
    beams, beam_probs = new_beams, new_beam_probs
  return beams, beam_probs

def run_pred_iter(net, pad_image_feature, vocab_list, strategies=[{'type': 'beam'}]):
  outputs = []
  for strategy in strategies:
    captions, probs = predict_image_caption(net, pad_image_feature, vocab_list, strategy=strategy)
    for caption, prob in zip(captions, probs):
      output = {}
      output['caption'] = caption
      output['prob'] = prob
      output['gt'] = False
      output['source'] = strategy
      outputs.append(output)
  return outputs

def next_video_gt_pair(tsg):
  streams = tsg.get_streams()
  video_id = tsg.lines[tsg.line_index-1][0]
  gt = streams['target_sentence']
  return video_id, gt

def all_video_gt_pairs(fsg):
  data = OrderedDict()
  if len(fsg.lines) > 0:
    prev_video_id = None
    while True:
      video_id, gt = next_video_gt_pair(fsg)
      if video_id in data:
        if video_id != prev_video_id:
          break
        data[video_id].append(gt)
      else:
        data[video_id] = [gt]
      prev_video_id = video_id
    #print 'Found %d videos with %d captions' % (len(data.keys()), len(data.values()))
  else:
    data = OrderedDict(((key, []) for key in fsg.vid_framefeats.keys()))
  return data

def gen_stats(prob, normalizer=None):
  stats = {}
  stats['length'] = len(prob)
  stats['log_p'] = 0.0
  eps = 1e-12
  for p in prob:
    assert 0.0 <= p <= 1.0
    stats['log_p'] += math.log(max(eps, p))
  stats['log_p_word'] = stats['log_p'] / stats['length']
  try:
    stats['perplex'] = math.exp(-stats['log_p'])
  except OverflowError:
    stats['perplex'] = float('inf')
  try:
    stats['perplex_word'] = math.exp(-stats['log_p_word'])
  except OverflowError:
    stats['perplex_word'] = float('inf')
  if normalizer is not None:
    norm_stats = gen_stats(normalizer)
    stats['normed_perplex'] = \
        stats['perplex'] / norm_stats['perplex']
    stats['normed_perplex_word'] = \
        stats['perplex_word'] / norm_stats['perplex_word']
  return stats

def run_pred_iters(pred_net, vidids, video_gt_pairs, fsg,
                   strategies=[{'type': 'beam'}], display_vocab=None):
  outputs = OrderedDict()
  num_pairs = 0
  descriptor_video_id = ''
  pad_img_feature = None
  for video_id in vidids:
    assert video_id not in outputs
    num_pairs += 1
    if descriptor_video_id != video_id:
      # get fc7 feature for the video
      video_features = video_to_descriptor(video_id, fsg)
      #print 'Num video features: %d ' % len(video_features)
      #print 'Dimension of video features: {0}'.format(video_features[0].shape)
      # run lstm on all the frames of video before predicting
      encode_video_frames(pred_net, video_features)
      # use the last frame from the video as padding
      pad_img_feature = video_features[-1]
      # Make padding all 0 when predicting
      pad_img_feature[pad_img_feature > 0] = 0

    outputs[video_id] = run_pred_iter(pred_net, pad_img_feature, display_vocab, strategies=strategies)

  return outputs

def to_text_output(outputs, vocab):
  out_types = {}
  num_videos = 0
  for video_id, captions in outputs.iteritems():
    num_videos += 1
    for c in captions:
      if not 'stats' in c:
        c['stats'] = gen_stats(c['prob'])
    # Sort captions by log probability.
    if 'normed_perplex' in captions[0]['stats']:
      captions.sort(key=lambda c: c['stats']['normed_perplex'])
    else:
      captions.sort(key=lambda c: -c['stats']['log_p_word'])
    for c in captions:
      caption = c['caption']
      caption_string = vocab_inds_to_sentence(vocab, caption)
      #source_meta = 'beam_size_%d' % source['beam_size']
      #out = '%s\t%s\t%s\n' % (source_meta, video_id,caption_string)
      # if len(out_types[source_meta]) < num_videos:
      #out_types[source_meta].append(out)
      out_types[video_id] = caption_string
  return out_types


def generateCaption(fileNames):
  answers = {}

  # TODO: Input the snapshot directory, vocab path, frames (and sents) path
  DIR = './s2vt/snapshots'
  VOCAB_FILE = './s2vt/yt_coco_mvad_mpiimd_vocabulary.txt'

  LSTM_NET_FILE = './s2vt/s2vt.words_to_preds.deploy.prototxt'
  MODEL_FILE = '%s/%s.caffemodel' % (DIR, 's2vt_vgg_rgb')
  SENTS_FILE = None #args.gold if args.gold else None # optional
  if DEVICE_ID >= 0:
    caffe.set_mode_gpu()
    caffe.set_device(DEVICE_ID)
  else:
    caffe.set_mode_cpu()
  print "Setting up LSTM NET"
  lstm_net = caffe.Net(LSTM_NET_FILE, MODEL_FILE, caffe.TEST)
  print "Done"

  STRATEGIES = [
    {'type': 'beam', 'beam_size': 1},
  ]
  NUM_OUT_PER_CHUNK = 30
  START_CHUNK = 0

  vocab_file = VOCAB_FILE
  DATASETS = [ ] # split_name, data_split_name, aligned
  for fileName in fileNames:
      DATASETS.append(('test', fileName, 'False'))

  for split_name, data_split_name, aligned in DATASETS:
    filenames = [(data_split_name, SENTS_FILE)]
    fsg = fc7FrameSequenceGenerator(filenames, BUFFER_SIZE,
          vocab_file, max_words=MAX_WORDS, align=aligned, shuffle=False,
          pad=aligned, truncate=aligned)
    video_gt_pairs = all_video_gt_pairs(fsg)
    print 'Read %d videos pool feats' % len(fsg.vid_framefeats)
    NUM_CHUNKS = (len(fsg.vid_framefeats)/NUM_OUT_PER_CHUNK) + 1
    eos_string = '<EOS>'
    # add english inverted vocab
    vocab_list = [eos_string] + fsg.vocabulary_inverted
    for c in range(START_CHUNK, NUM_CHUNKS):
      chunk_start = c * NUM_OUT_PER_CHUNK
      chunk_end = (c + 1) * NUM_OUT_PER_CHUNK
      chunk = video_gt_pairs.keys()[chunk_start:chunk_end]

      outputs = run_pred_iters(lstm_net, chunk, video_gt_pairs,
                    fsg, strategies=STRATEGIES, display_vocab=vocab_list)

      answers.update(to_text_output(outputs, vocab_list))


  return answers


