require 'nn'
require 'rnn'
local utils = require 'utils'
local LSTM = require 'LSTM'


-------------------------------------------------------------------------------
-- Language Model core
-------------------------------------------------------------------------------

local layer, parent = torch.class('nn.LM_encoder', 'nn.Module')
function layer:__init(opt)
  parent.__init(self)

  -- options for core network
  self.vocab_size = utils.getopt(opt, 'vocab_size') -- required
  self.input_encoding_size = utils.getopt(opt, 'input_encoding_size')
  self.rnn_size = utils.getopt(opt, 'rnn_size')
  self.num_layers = utils.getopt(opt, 'num_layers', 1)
  local dropout_l = utils.getopt(opt, 'dropout_l', 0.5)
  local dropout_t = utils.getopt(opt, 'dropout_t', 0)
  local rnn_type = utils.getopt(opt, 'rnn_type', 'lstmb')
  local res_rnn = utils.getopt(opt, 'res_rnn', false)
  self.lookup_table = nn.LookupTableMaskZero(self.vocab_size + 1, self.input_encoding_size)
  -- options for Language Model
  -- create the core lstm network. note +1 for both the START and END tokens
  if rnn_type == 'lstm' then
    self.core = LSTM.lstm(self.input_encoding_size, 1, self.rnn_size, self.num_layers, dropout_l, dropout_t, res_rnn, 0, false, 0)
  elseif rnn_type == 'rnn' then
    self.core = LSTM.rnn(self.input_encoding_size, 1, self.rnn_size, self.num_layers, dropout_l, dropout_t, res_rnn, 0)
  elseif rnn_type == 'lstmb' then
    self.core = LSTM.lstm(self.input_encoding_size, 1, self.rnn_size, self.num_layers, dropout_l, dropout_t, res_rnn, 0, false, 1)
  elseif rnn_type == 'clstm' then
    self.core = LSTM.clstm(self.input_encoding_size, 1, self.rnn_size, self.num_layers, dropout_l, dropout_t, res_rnn, 0, false, 1)
  elseif rnn_type == 'slstm' then
    self.core = LSTM.lstm(self.input_encoding_size, 1, self.rnn_size, self.num_layers, dropout_l, dropout_t, res_rnn, 0, true, 1)
  elseif rnn_type == 'nlstm' then
    self.core = LSTM.lstm(self.input_encoding_size, 1, self.rnn_size, self.num_layers, dropout_l, dropout_t, res_rnn, 2, true, 0)
  
  else
    assert(1==0, 'unsupport rnn type')
  end
  self:_createInitState(1) -- will be lazily resized later during forward passes
end

function layer:_createInitState(batch_size)
  assert(batch_size ~= nil, 'batch size must be provided')
  -- construct the initial state for the LSTM
  if not self.init_state then self.init_state = {} end -- lazy init
  for h=1,self.num_layers*2 do
    -- note, the init state Must be zeros because we are using init_state to init grads in backward call too
    if self.init_state[h] then
      if self.init_state[h]:size(1) ~= batch_size then
        self.init_state[h]:resize(batch_size, self.rnn_size):zero() -- expand the memory
      end
    else
      self.init_state[h] = torch.zeros(batch_size, self.rnn_size)
    end
  end
  self.num_state = #self.init_state
end

function layer:createClones(seq_length)
  -- construct the net clones
  print('constructing clones inside the LanguageModel')
  self.clones = {self.core}
  self.lookup_tables = {self.lookup_table}
  for t=2,seq_length do
    self.clones[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
    self.lookup_tables[t] = self.lookup_table:clone('weight', 'gradWeight')
  end
end

function layer:getModulesList()
  return {self.core, self.lookup_table}
end

function layer:parameters()
  -- we only have two internal modules, return their params
  local p1,g1 = self.core:parameters()
  local p2,g2 = self.lookup_table:parameters()

  local params = {}
  for k,v in pairs(p1) do table.insert(params, v) end
  for k,v in pairs(p2) do table.insert(params, v) end
  
  local grad_params = {}
  for k,v in pairs(g1) do table.insert(grad_params, v) end
  for k,v in pairs(g2) do table.insert(grad_params, v) end

  -- todo: invalidate self.clones if params were requested?
  -- what if someone outside of us decided to call getParameters() or something?
  -- (that would destroy our parameter sharing because clones 2...end would point to old memory)

  return params, grad_params
end

function layer:training()
  if self.clones == nil then self:createClones(16) end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:training() end
end

function layer:evaluate()
  if self.clones == nil then self:createClones(16) end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:evaluate() end
end

--[[
input is T x B x F
timestep
batchsize
feature

--]]
function layer:updateOutput(seq)
  local seq_length = seq:size(1)
  local seq_per_img = seq:size(2)
  local batch_size = seq:size(3)
  seq = torch.reshape(seq,seq_length, batch_size * seq_per_img) 
  
  -- self.output:resize(self.seq_length+2, batch_size, self.vocab_size+1)
  if self.clones == nil then self:createClones(seq_length) end -- lazily create clones on first forward pass
  
  self:_createInitState(batch_size*seq_per_img)
  self.inputs = {}
  self.state = {[0] = self.init_state}
  
  -- print(seq) 

  for t=1,seq_length do
    -- construct the inputs
    local seq_embed = self.lookup_tables[t]:forward(seq[t])
    self.inputs[t] = {seq_embed,unpack(self.state[t-1])}
    -- forward the network
    
    local out = self.clones[t]:forward(self.inputs[t])
    -- process the outputs
    -- self.output[t] = out[self.num_state+1] -- last element is the output vector
    self.state[t] = {} -- the rest is state
    for i=1,batch_size do
      if seq[t][i] == 0 then
        for j = 1,self.num_state do
          out[j][i]:zero()
        end
      end
    end
    for i=1,self.num_state do table.insert(self.state[t], out[i]) end
  end

  
 local state_last = {torch.reshape(self.state[seq_length][1],seq_per_img,batch_size,512),torch.reshape(self.state[seq_length][2],seq_per_img,batch_size,512)}
  return state_last

end


--[[
gradOutput is an (D+2)xNx(M+1) Tensor.
--]]
function layer:updateGradInput(seq, gradOutput)
  -- self.gradInput:resizeAs(seq)
  local seq_length = seq:size(1)
  local seq_per_img = seq:size(2)
  local batch_size = seq:size(3)
  seq = torch.reshape(seq,seq_length, batch_size * seq_per_img) 
  gradOutput = {torch.reshape(gradOutput[1], batch_size*seq_per_img, self.rnn_size), torch.reshape(gradOutput[2], batch_size*seq_per_img, self.rnn_size)}

  -- go backwards and lets compute gradients
  local dstate = {[seq_length] = gradOutput} -- this works when init_state is all zeros
  for t=seq_length,1,-1 do
    -- concat state gradients and output vector gradients at time step t
    local dout = {}
    for k=1,#dstate[t] do table.insert(dout, dstate[t][k]) end
    table.insert(dout, torch.zeros(batch_size * seq_per_img, 1))
    local dinputs = self.clones[t]:backward(self.inputs[t], dout)
    -- split the gradient to xt and to state
    dstate[t-1] = {} -- copy over rest to state grad
    for i=1,batch_size do
      if seq[t][i] == 0 then
        for j = 2,self.num_state+1 do
          dinputs[j][i]:zero()
        end
      end
    end
    -- self.gradInput[t] = dinputs[1] -- first element is the input vector
    self.lookup_tables[t]:backward(seq[t], dinputs[1])
    for k=2,self.num_state+1 do table.insert(dstate[t-1], dinputs[k]) end
  end

  return torch.Tensor()
end
