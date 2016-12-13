require 'torch'
require 'nn'
require 'image'
require 'optim'
npy4th = require 'npy4th'

require 'loadcaffe'

local cmd = torch.CmdLine()

-- Basic options
cmd:option('-style', '', 'Style spectrogram')
cmd:option('-content', '', 'Content spectrogram')
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')

-- Optimization options
cmd:option('-content_weight', 5e0)
cmd:option('-style_weight', 1e2)
cmd:option('-num_iterations', 5000)
cmd:option('-normalize_gradients', false)
cmd:option('-init', 'random', 'random|content')
cmd:option('-optimizer', 'lbfgs', 'lbfgs|adam')
cmd:option('-learning_rate', 1e1)

-- Output options
cmd:option('-print_iter', 50)
cmd:option('-save_iter', 100)
cmd:option('-output_image', 'out.png')
cmd:option('-save', 'data/out/', 'Where to store intermediate results.')

-- Loss options
cmd:option('-content_layers', '', 'Layers for content.')
cmd:option('-style_layers', '', 'Layers for style.')
cmd:option('-how_div', 's', 's|s2, Different options for style loss normalization.')
cmd:option('-loss', 'l2', 'l1|l2 Loss for gram matrix matching.')
cmd:option('-lowres', false, 'Process also in low resolution, this will increase receptive field and "textures" width.')

-- Other options
cmd:option('-model_t7', 'data/net.t7', 'Path to model file')
cmd:option('-mean_file_t7', 'data/mean.t7', 'Path to file with mean spectrogram.')
cmd:option('-backend', 'cudnn', 'nn|cudnn|clnn')
cmd:option('-cudnn_autotune', true)
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
    if params.cudnn_autotune then
      cudnn.benchmark = true
    end
    cudnn.SpatialConvolution.accGradParameters = nn.SpatialConvolutionMM.accGradParameters -- ie: nop
  end
  
  local loadcaffe_backend = params.backend
  if params.backend == 'clnn' then loadcaffe_backend = 'nn' end

  
  if params.content_layers == '' then
    params.content = params.style
  end
  
  -- Load spectrograms
  local content = load_data(params.content):type(params.dtype)
  local style = load_data(params.style):type(params.dtype)

  local content_layers = params.content_layers:split(",")
  local style_layers = params.style_layers:split(",")

  local content_losses, style_losses = {}, {}

  local net = nn.Sequential()

  function get_net(net)
    local cnn = torch.load(params.model_t7):type(params.dtype)
    cnn:evaluate()

    -- Set up the network, inserting style and content loss modules
    local next_content_idx, next_style_idx = 0, 0
    for i = 0, #cnn do
      if next_content_idx <= #content_layers or next_style_idx <= #style_layers then
        
        if i > 0 then
          local layer = cnn:get(i)

          net:add(layer)
        end

        if i == tonumber(content_layers[next_content_idx+1]) then
          print("Setting up content layer", i)
          local target = net:forward(content):clone()
          local norm = params.normalize_gradients
          local loss_module = nn.ContentLoss(params.content_weight, target, norm):type(params.dtype)

          net:add(loss_module)
          table.insert(content_losses, loss_module)
          next_content_idx = next_content_idx + 1
        end
        if i == tonumber(style_layers[next_style_idx+1]) then
          print("Setting up style layer  ", i)
          local gram = GramMatrix():type(params.dtype)

          local target_features = net:forward(style):clone()
          local target = gram:forward(target_features):clone()

          local to_div = 1
          if how_div == 's2' then
            to_div =  target_features:size(2)
          elseif how_div == 's' then
            to_div =  target_features:nElement()
          end
          target:div(to_div)
          local norm = params.normalize_gradients

          local loss_module
          if i == 0 then 
            loss_module = nn.StyleLoss(params.style_weight*10, target, norm):type(params.dtype)
          else
            loss_module = nn.StyleLoss(params.style_weight, target, norm):type(params.dtype)
          end
          
          net:add(loss_module)
          table.insert(style_losses, loss_module)
          next_style_idx = next_style_idx + 1
        end
      end
    end
    net:add(nn.DummyGradOutput():type(params.dtype))

    for i=1,#net.modules do
      local module = net.modules[i]
      if torch.type(module) == 'nn.SpatialConvolutionMM' then
          -- remove these, not used, but uses gpu memory
          module.gradWeight = nil
          module.gradBias = nil
      end
    end

    return net
  end
  net = get_net(net)

  if params.lowres then
    print('Setting up a net on low resolution.')

    local net1 = net
    
    -- Add pooling to the start 
    net = nn.Sequential():add(nn.SpatialMaxPooling(1,2,1,2):type(params.dtype))
    local net2 = get_net(net)

    net = nn.Sequential()
                :add(nn.ConcatTable():type(params.dtype):add(net1):add(net2))
                :add(nn.DummyGradOutput():type(params.dtype))
  end 

  collectgarbage()
  
  -- Initialize the spectrogram
  if params.seed >= 0 then
    torch.manualSeed(params.seed)
    cutorch.manualSeed(params.seed)
  end
  local img = nil
  if params.init == 'random' then
    img = torch.randn(content:size()):float():mul(0.001)
  elseif params.init == 'content' then
    img = content:clone():float()
  else
    error('Invalid init type')
  end

  img = img:type(params.dtype)
    
  optim_state_lbfgs = {
      maxIter = params.num_iterations,
      verbose=true,
      tolX = -1,
      tolFun = -1,
      learningRate = 1,

  }
  
  optim_state_adam = {
      learningRate = params.learning_rate,
  }


  local function maybe_print(t, loss)
    local verbose = (params.print_iter > 0 and t % params.print_iter == 0)
    if verbose then
      print(string.format('Iteration %d / %d', t, params.num_iterations))
      for i, loss_module in ipairs(content_losses) do
        print(string.format('  Content %d loss: %f', i, loss_module.loss))
      end
      for i, loss_module in ipairs(style_losses) do
        print(string.format('  Style %d loss: %f', i, loss_module.loss))
      end
      print(string.format('  Total loss: %f', loss))
    end
  end

  local function maybe_save(t)
    local should_save = params.save_iter > 0 and t % params.save_iter == 0
    should_save = should_save or t == params.num_iterations
    if should_save then
      local mean = torch.load(params.mean_file_t7)
      mean = mean:view(1, mean:size(1),mean:size(2),mean:size(3)):type(params.dtype)

      local disp = torch.add(img,mean:expandAs(img))
      disp = disp:view(img:size(2),img:size(3)):double()
      
      local filename = build_filename(params.output_image, t)
      if t == params.num_iterations then
        filename = params.output_image
      end

      image.save(params.save .. '/' .. filename, disp:float())
      torch.save(params.save .. '/' .. filename .. '.t7', disp:float())
    end
  end

  local num_calls = 0
  local function feval(x)
    num_calls = num_calls + 1
    net:forward(x)
    local grad = net:updateGradInput(x)
  
    local loss = 0
    for _, mod in ipairs(content_losses) do
      loss = loss + mod.loss
    end
    for _, mod in ipairs(style_losses) do
      loss = loss + mod.loss
    end
    maybe_print(num_calls, loss)
    maybe_save(num_calls)

    collectgarbage()

    -- optim.lbfgs expects a vector for gradients
    return loss, grad:view(grad:nElement())
  end

  -- Run optimization.
  if params.optimizer == 'lbfgs' then
    print('Running optimization with L-BFGS')
    local x, losses = optim.lbfgs(feval, img, optim_state_lbfgs)
  elseif params.optimizer == 'adam' then
    print('Running optimization with ADAM')
    for t = 1, params.num_iterations do
      local x, losses = optim.adam(feval, img, optim_state_adam)
    end
  end
end
  

function build_filename(output_image, iteration)
  local ext = paths.extname(output_image)
  local basename = paths.basename(output_image, ext)
  local directory = paths.dirname(output_image)
  return string.format('%s/%s_%d.%s',directory, basename, iteration, ext)
end

-- Define an nn Module to compute content loss in-place
local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')

function ContentLoss:__init(strength, target, normalize)
  parent.__init(self)
  self.strength = strength
  self.target = target
  self.normalize = normalize or false
  self.loss = 0
  self.crit = nn.MSECriterion(false)
end

function ContentLoss:updateOutput(input)
  if input:nElement() == self.target:nElement() then
    self.loss = self.crit:forward(input, self.target) * self.strength
  else
    print('WARNING: Skipping content loss')
  end
  self.output = input
  return self.output
end

function ContentLoss:updateGradInput(input, gradOutput)
  if input:nElement() == self.target:nElement() then
    self.gradInput = self.crit:backward(input, self.target)
  end
  if self.normalize then
    self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
  end
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
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

function StyleLoss:__init(strength, target, normalize)
  parent.__init(self)
  self.normalize = normalize or false
  self.strength = strength
  self.target = target
  self.loss = 0
  
  self.gram = GramMatrix()
  self.G = nil
  if params.loss == 'l2' then
    self.crit = nn.MSECriterion(false)
  else
    self.crit = nn.SmoothL1Criterion(false)
  end
end

function StyleLoss:updateOutput(input)
  self.G = self.gram:forward(input)
  local to_div = 1
  if how_div == 's2' then
    to_div =  input:size(2)
  elseif how_div == 's' then
    to_div =  input:nElement()
  end
  -- self.G:div(input:size(2))
  self.G:div(to_div)
  -- self.G:div(input:nElement())
  
  self.loss = self.crit:forward(self.G, self.target)
  self.loss = self.loss * self.strength
  self.output = input
  return self.output
end

function StyleLoss:updateGradInput(input, gradOutput)
  local dG = self.crit:backward(self.G, self.target)
  local to_div = 1
  if how_div == 's2' then
    to_div =  input:size(2)
  elseif how_div == 's' then
    to_div =  input:nElement()
  end
  -- dG:div(input:size(2))
  dG:div(to_div)
  
  -- dG:div(input:nElement())
  self.gradInput = self.gram:backward(input, dG)
  if self.normalize then
    self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
  end
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end

-- Simpulates Identity operation with 0 gradOutput
local DummyGradOutput, parent = torch.class('nn.DummyGradOutput', 'nn.Module')

function DummyGradOutput:__init()
  parent.__init(self)
  self.gradInput = nil
end


function DummyGradOutput:updateOutput(input)
  self.output = input
  return self.output
end

function DummyGradOutput:updateGradInput(input, gradOutput)
  if torch.type(input) == 'table' then
    if not self.gradInput or 
                       not input[1]:isSameSizeAs(self.gradInput[1]) or
                       not input[2]:isSameSizeAs(self.gradInput[2]) then
      self.gradInput = {input[1].new(),input[2].new()}
      self.gradInput[1]:resizeAs(input[1]):fill(0)
      self.gradInput[2]:resizeAs(input[2]):fill(0)
  end
  elseif not self.gradInput or not input:isSameSizeAs(self.gradInput) then
    self.gradInput = input.new():resizeAs(input):fill(0)
  end

  -- print(input:size())
  return self.gradInput 
end


function load_data(path)
  local d =  npy4th.loadnpy(path)
  d = d:narrow(1,1,320):contiguous()
  
  local mean = torch.load(params.mean_file_t7)
  mean = mean:view(1, mean:size(1),mean:size(2),mean:size(3))

  d = d:view(1,d:size(1),d:size(2),1)
  d:csub(mean:expandAs(d))
  return d
end

params = cmd:parse(arg)

how_div = params.how_div
main(params)
