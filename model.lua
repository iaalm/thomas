require 'LM_encoder'
require 'LM_decoder'
require 'LM_mistery'


local layer, parent = torch.class('nn.LM', 'nn.Module')
function layer:__init(opt)
  parent.__init(self)
  self.encoder = nn.LM_encoder(opt)
  self.m = nn.MisteryMax()
  opt.seq_length = 16
  self.decoder = nn.LM_decoder(opt)
  self.expand = nn.ParallelTable()
  local m = nn.Sequential()
  m:add(nn.Replicate(5, 2))
  m:add(nn.Reshape(80, 512, false))
  self.expand:add(m)
  local m = nn.Sequential()
  m:add(nn.Replicate(5, 2))
  m:add(nn.Reshape(80, 512, false))
  self.expand:add(m)
  -- self.decoder.lookup_table:share(self.encoder.lookup_table, 'weight', 'bias')
end

function layer:parameters()
  -- we only have two internal modules, return their params
  local p1,g1 = self.encoder:parameters()
  local p2,g2 = self.m:parameters()
  local p3,g3 = self.decoder:parameters()

  local params = {}
  for k,v in pairs(p1) do table.insert(params, v) end
  for k,v in pairs(p2) do table.insert(params, v) end
  for k,v in pairs(p3) do table.insert(params, v) end
  
  local grad_params = {}
  for k,v in pairs(g1) do table.insert(grad_params, v) end
  for k,v in pairs(g2) do table.insert(grad_params, v) end
  for k,v in pairs(g3) do table.insert(grad_params, v) end

  -- todo: invalidate self.clones if params were requested?
  -- what if someone outside of us decided to call getParameters() or something?
  -- (that would destroy our parameter sharing because clones 2...end would point to old memory)

  return params, grad_params
end

function layer:training()
  self.encoder:training()
  self.m:training()
  self.decoder:training()
end

function layer:evaluate()
  self.encoder:evaluate()
  self.m:evaluate()
  self.decoder:evaluate()
end

function layer:updateOutput(input)
  self.sent_embd = self.encoder:forward(input.densecap)
  self.compact_embd = self.m:forward(self.sent_embd)
  self.exp_embd = self.expand:forward(self.compact_embd)
  self.output = self.decoder:forward{self.exp_embd, input.seq}
  return self.output
end

function layer:updateGradInput(input, gradOutput)
  local dexp_embd = self.decoder:backward({self.epx_embd, input.seq}, gradOutput)
  local dcompact_embd = self.expand:backward(self.compact_embd, dexp_embd)
  local dsent_embd = self.m:backward(self.sent_embd, dcompact_embd)
  self.gradInput = self.encoder:backward(input.densecap, dsent_embd)
  return self.gradInput
end

function layer:sample(input)
  self.sent_embd = self.encoder:forward(input.densecap)
  self.compact_embd = self.m:forward(self.sent_embd)
  self.output = self.decoder:sample(self.compact_embd)
  return self.output
end

