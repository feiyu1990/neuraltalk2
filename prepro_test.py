
import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
from scipy.misc import imread, imresize
import nltk


def prepro_captions(imgs):
  # preprocess all the captions
  print 'example processed tokens:'
  tokenize = params['tokenize_method']
  punc = '!#$%&()*+,./:;<=>?@[\\]^`{|}~'
  for i,img in enumerate(imgs):
    img['processed_tokens'] = []
    for j,s in enumerate(img['captions']):
      if tokenize == 'nltk':
        sentence = str(s).lower().translate(None, punc)
        txt = nltk.word_tokenize(sentence)
      elif tokenize == 'neuraltalk':
        sentence = str(s).lower().translate(None, string.punctuation).strip()
        txt = sentence.split()
        #TODO add phrase here
        # else:
        #     sentence = str(i['captions'][0]).lower().translate(None, punc)
        #     start_phrase = set(temp['start'])
        #     token = tokeninze_phrase(sentence.split(), priority_phrase, start_phrase)
      else:
        raise ValueError('unrecognized tokenize method!')
      img['processed_tokens'].append(txt)
      if i < 10 and j == 0: print txt


def build_vocab(imgs, params):
  vocab_dict = json.load(open('training_json'))['ix_to_word']
  for img in imgs:
    img['final_captions'] = []
    for txt in img['processed_tokens']:
      caption = [w if w in vocab_dict else 'UNK' for w in txt]
      img['final_captions'].append(caption)

  return len(vocab_dict)


def assign_splits(imgs, params):
  for i,img in enumerate(imgs):
    img['split'] = 'test'


def encode_captions(imgs, params, wtoi):
  """
  encode all captions into one large array, which will be 1-indexed.
  also produces label_start_ix and label_end_ix which store 1-indexed
  and inclusive (Lua-style) pointers to the first and last caption for
  each image in the dataset.
  """

  max_length = params['max_length']
  N = len(imgs)
  M = sum(len(img['final_captions']) for img in imgs) # total number of captions

  label_arrays = []
  label_start_ix = np.zeros(N, dtype='uint32') # note: these will be one-indexed
  label_end_ix = np.zeros(N, dtype='uint32')
  label_length = np.zeros(M, dtype='uint32')
  caption_counter = 0
  counter = 1
  for i,img in enumerate(imgs):
    n = len(img['final_captions'])
    assert n > 0, 'error: some image has no captions'

    Li = np.zeros((n, max_length), dtype='uint32')
    for j,s in enumerate(img['final_captions']):
      label_length[caption_counter] = min(max_length, len(s)) # record the length of this sequence
      caption_counter += 1
      for k,w in enumerate(s):
        if k < max_length:
          Li[j,k] = wtoi[w]

    # note: word indices are 1-indexed, and captions are padded with zeros
    label_arrays.append(Li)
    label_start_ix[i] = counter
    label_end_ix[i] = counter + n - 1

    counter += n

  L = np.concatenate(label_arrays, axis=0) # put all the labels together
  assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
  assert np.all(label_length > 0), 'error: some caption had no words?'

  print 'encoded captions to array of size ', `L.shape`
  return L, label_start_ix, label_end_ix, label_length

def main(params):

  imgs = json.load(open(params['input_json'], 'r'))
  seed(123) # make reproducible
  shuffle(imgs) # shuffle the order

  if params['max_imgs']:
      imgs = imgs[:params['max_imgs']]

  # tokenization and preprocessing
  prepro_captions(imgs)

 # create the vocab
  vocab = build_vocab(imgs, params)
  itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
  wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table

  # assign the splits
  assign_splits(imgs, params)

  # encode captions in large arrays, ready to ship to hdf5 file
  L, label_start_ix, label_end_ix, label_length = encode_captions(imgs, params, wtoi)

  # create output h5 file
  N = len(imgs)
  f = h5py.File(params['output_h5'], "w")
  f.create_dataset("labels", dtype='uint32', data=L)
  f.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
  f.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
  f.create_dataset("label_length", dtype='uint32', data=label_length)
  if params['image_precompute']:
    file_name = params['image_precompute']
    img_orders_already_have = json.load(open(file_name+'.json'))['images']
    for i,img in enumerate(imgs):
      if img_orders_already_have[i]['split'] != img['split'] or img_orders_already_have[i]['id'] != img['id']:
        raise ValueError('DATASET NOT MATCH!')
    # f['images'] = h5py.ExternalLink(file_name.split('/')[-1]+'.h5', '/'.join(file_name.split('/')[:-1]))
    fr = h5py.File(file_name + '.h5', 'r')
    fr.copy('images', f)
  else:
    dset = f.create_dataset("images", (N,3,256,256), dtype='uint8') # space for resized images
    for i,img in enumerate(imgs):
        I = imread(os.path.join(params['images_root'], img['file_path']))
        try:
            Ir = imresize(I, (256,256))
        except:
            print 'failed resizing image %s - see http://git.io/vBIE0' % (img['file_path'],)
            raise
        # handle grayscale input images
        if len(Ir.shape) == 2:
          Ir = Ir[:,:,np.newaxis]
          Ir = np.concatenate((Ir,Ir,Ir), axis=2)
        # and swap order of axes from (256,256,3) to (3,256,256)
        Ir = Ir.transpose(2,0,1)
        # write to h5
        dset[i] = Ir
        if i % 1000 == 0:
          print 'processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N)
  f.close()
  print 'wrote ', params['output_h5']


  # create output json file
  out = {}
  out['ix_to_word'] = itow # encode the (1-indexed) vocab
  out['images'] = []
  for i,img in enumerate(imgs):

    jimg = {}
    jimg['split'] = img['split']
    if 'file_path' in img: jimg['file_path'] = img['file_path'] # copy it over, might need
    if 'id' in img: jimg['id'] = img['id'] # copy over & mantain an id, if present (e.g. coco ids, useful)

    out['images'].append(jimg)

  json.dump(out, open(params['output_json'], 'w'))
  print 'wrote ', params['output_json']

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
  parser.add_argument('--output_json', default='data.json', help='output json file')
  parser.add_argument('--training_json', required=True)
  parser.add_argument('--output_h5', default='data.h5', help='output h5 file')
  parser.add_argument('--image_precompute', default=None)
  parser.add_argument('--max_imgs', type=int, default=None)
  parser.add_argument('--tokenize_method', required=True)

  # options
  parser.add_argument('--max_length', default=16, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
  parser.add_argument('--images_root', default='', help='root location in which images are stored, to be prepended to file_path in input json')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print 'parsed input parameters:'
  print json.dumps(params, indent = 2)
  main(params)
