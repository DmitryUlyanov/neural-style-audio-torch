require 'torch'
require 'nn'
require 'image'
require 'optim'
npy4th = require 'npy4th'

require 'loadcaffe'

local cmd = torch.CmdLine()

-- Basic options
cmd:option('-style', 'data/inputs/usa.npy', 'Style spectrogram')
cmd:option('-content', 'data/inputs/imperial.npy', 'Content spectrogram')
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')

-- Optimization options
cmd:option('-alpha', 1e-2)
cmd:option('-num_iterations', 300)

-- Other options
cmd:option('-backend', 'cudnn', 'nn|cudnn|clnn')
cmd:option('-seed', -1)

local function main(params)

  if params.gpu >= 0 then
    if params.backend ~= 'clnn' then
      require 'cutorch'
      require 'cunn'
      params.dtype = 'torch.CudaTensor'
      cutorch.setDevice(params.gpu + 1)
    else
      require 'clnn'
      require 'cltorch'
      params.dtype = 'torch.ClTensor'
      cltorch.setDevice(params.gpu + 1)
    end
  else
    params.backend = 'nn'
    params.dtype = 'torch.FloatTensor'
  end

  if params.backend == 'cudnn' then
    require 'cudnn'
    cudnn.benchmark = true
    cudnn.SpatialConvolution.accGradParameters = nn.SpatialConvolutionMM.accGradParameters -- ie: nop
  end
  
  local backend = params.backend
  if params.backend == 'clnn' then backend = 'nn' end

  -- Load spectrograms
  local content = load_data(params.content):type(params.dtype)
  local style = load_data(params.style):type(params.dtype)

  print (content:size())
  local N_CHANNELS = content:size(2)
  local N_FILTERS = 4096

  local net = nn.Sequential()

    local conv_layer = nn.SpatialConvolution(N_CHANNELS, N_FILTERS ,1, 11,1,1,0,0)
    conv_layer.bias:zero()
    local std = math.sqrt(2) * math.sqrt(2.0 / ((N_CHANNELS + N_FILTERS) * 11))
    conv_layer.weight:normal():mul(std)
    net = nn.Sequential():add(conv_layer):add(cudnn.ReLU()):type(params.dtype)

    local content_features = net:forward(content):clone()
    local style_features = net:forward(style):clone()

    -- Content loss
    local content_module = nn.ContentLoss(params.alpha, content_features):type(params.dtype)
    net:add(content_module)

    -- Style loss
    local gram = GramMatrix():type(params.dtype)
    local style_gram = gram:forward(style_features):clone()
    style_gram:div(style_features:size(3))
          
    local style_module = nn.StyleLoss(style_gram):type(params.dtype)
    net:add(style_module)

    print(backend)
  net = cudnn.convert(net, backend)
  collectgarbage()
  
  local img  = torch.randn(content:size()):float():mul(1e-3):type(params.dtype)
  local y_grad = content_features:clone():zero()

  optim_state_lbfgs = {
      maxIter = params.num_iterations,
      verbose=true,
      tolX = -1,
      tolFun = -1,
      learningRate = 1}

  local function feval(x)
    net:forward(x)
    local grad = net:updateGradInput(x,y_grad)
      
    local loss = style_module.loss + content_module.loss 
    collectgarbage()
    return loss, grad:view(grad:nElement())
  end

  print('Running optimization with L-BFGS')
  local x, losses = optim.lbfgs(feval, img, optim_state_lbfgs)
  torch.save('out.t7', x:view(x:size(2),x:size(3)):double():float())
end
  

-- Define an nn Module to compute content loss in-place
local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')

function ContentLoss:__init(strength, target)
  parent.__init(self)
  self.strength = strength
  self.target = target
  self.loss = 0
  self.crit = nn.MSECriterion(false)
end

function ContentLoss:updateOutput(input)
  if input:nElement() == self.target:nElement() then
    self.loss = self.crit:forward(input, self.target) * self.strength
  end
  self.output = input
  return self.output
end

function ContentLoss:updateGradInput(input, gradOutput)
  if input:nElement() == self.target:nElement() then
    self.gradInput = self.crit:backward(input, self.target)
  end

  self.gradInput:mul(self.strength):add(gradOutput)
  return self.gradInput
end

-- Returns a network that computes the CxC Gram matrix from inputs
-- of size C x H x W
function GramMatrix()
  local net = nn.Sequential()
  net:add(nn.View(-1):setNumInputDims(2))
  local concat = nn.ConcatTable()
  concat:add(nn.Identity())
  concat:add(nn.Identity())
  net:add(concat)
  net:add(nn.MM(false, true))
  return net
end


-- Define an nn Module to compute style loss in-place
local StyleLoss, parent = torch.class('nn.StyleLoss', 'nn.Module')

function StyleLoss:__init(target)
  parent.__init(self)
  self.target = target
  self.loss = 0
  self.gram = GramMatrix()
  self.G = nil
  self.crit = nn.MSECriterion(false)
end

function StyleLoss:updateOutput(input)
  self.G = self.gram:forward(input)

  self.G:div(input:size(3))
  
  self.loss = self.crit:forward(self.G, self.target)
  self.loss = self.loss 
  self.output = input
  return self.output
end

function StyleLoss:updateGradInput(input, gradOutput)
  local dG = self.crit:backward(self.G, self.target)
  dG:div(input:size(3))
  
  self.gradInput = self.gram:backward(input, dG):add(gradOutput)
  return self.gradInput
end


function load_data(path)
  local d =  npy4th.loadnpy(path)
  d = d:view(1,d:size(1),d:size(2),1)
  return d
end

params = cmd:parse(arg)
main(params)
