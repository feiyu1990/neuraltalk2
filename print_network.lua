require 'torch'
require 'nn'
require 'nngraph'
-- exotics
require 'loadcaffe'
-- local imports
local utils = require 'misc.utils'
require 'misc.DataLoader'
require 'misc.DataLoaderRaw'
require 'misc.LanguageModel'
local net_utils = require 'misc.net_utils'
require 'cutorch'
require 'cunn'
require 'cudnn'
local checkpoint = torch.load('../fotolia/training/snapshot/model_id3.t7')
local proto = checkpoint.protos
local cnn_proto = proto.cnn
local lstm_proto = proto.lm
print(cnn_proto)
print(lstm_proto)



