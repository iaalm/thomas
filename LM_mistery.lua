
local layer, parent = torch.class('nn.MisteryMax', 'nn.ParallelTable')

function layer:__init()
  parent.__init(self)
  self:add(nn.Max(1))
  self:add(nn.Max(1))
end
