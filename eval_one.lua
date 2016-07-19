require 'inn'
require 'torch'
require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'
require 'cudnn'
-- exotics
require 'loadcaffe'
-- local imports
local utils = require 'misc.utils'
require 'misc.DataLoader'
require 'misc.DataLoaderRaw'
require 'misc.LanguageModel'
local net_utils = require 'misc.net_utils'

--local checkpoint = torch.load('../fotolia/training/snapshot/model_idhaystack_minlen2_50w_mirrored.t7')
--local protos = checkpoint.protos

local protos = {}
local cnn_raw = torch.load('model/haystack.t7')
protos.cnn = net_utils.build_cnn_google_withoutencoding(cnn_raw, {encoding_size = -1, backend = 'cudnn'})

protos.cnn:evaluate()
for k,v in pairs(protos) do v:cuda() end

local img_batch_raw = torch.ByteTensor(1, 3, 224, 224)
local h5_file = hdf5.open('../fotolia/test_caffe_input_torch.h5', 'r')
local img = h5_file:read('/images'):partial({1,1},{1,3},
                            {1,224},{1,224})
img_batch_raw[1] = img
--print(img_batch_raw)
img_batch_raw = net_utils.prepro(img_batch_raw, false, true) -- preprocess in place, and don't augment
local feats = protos.cnn:forward(img_batch_raw)
print(feats)
utils.write_json('../fotolia/test_torch_result.json', torch.totable(feats))
