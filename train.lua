require 'torch'
require 'nn'
require 'sys'
require 'nngraph'
-- local imports
require 'model'
require 'DataLoader'
require 'optim_updates'
local net_utils = require 'net_utils'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an WHZ Image Captioning model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_h5_cocotalk','/s/coco/cocotalk.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_h5_densecap','cocodens.h5','path to the h5file containing the preprocessed dataset')

cmd:option('-input_json','/s/coco/cocotalk.json','path to the json file containing additional info and vocab')
cmd:option('-input_val','annotations/captions_val2014.json','path to the json file containing caption for val')
cmd:option('-gpuid',0,'GPU id')
cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-input_encoding_size',512,'the encoding size of each token in the vocabulary, and the image.')
cmd:option('-seed',0,'random seed')
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-batch_size',16,'what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-grad_clip',0.1,'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-seq_per_img',5,'number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
cmd:option('-val_images_use', 3200, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-language_eval', 1, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
cmd:option('-id', '', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-losses_log_every', 25, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
cmd:option('-learning_rate',4e-4,'learning rate')
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 50000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim','adam','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-optim_alpha',0.8,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-save_checkpoint_every', 2500, 'how often to save a model checkpoint?')

--densecap
--TODO:cmd:option('densecap_per_img',10, 'number of densecaptions for each image.')

cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local task_hash = torch.random()
print('task hash:', task_hash)
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader = DataLoader{h5_file_densecap = opt.input_h5_densecap, h5_file_cocotalk = opt.input_h5_cocotalk, json_file = opt.input_json}
local model = nn.LM{vocab_size = loader.vocab_size, input_encoding_size = opt.input_encoding_size, rnn_size = opt.rnn_size}
local crit = nn.LanguageModelCriterion()
local params, grad_params= model:getParameters()
print('total number of parameters',params:nElement())
assert(params:nElement() == grad_params:nElement())

local function train_batch()
	model:training()
	grad_params:zero()

	-- get batch of data  
	local data = loader:getBatch{batch_size = opt.batch_size, split = 'train', seq_per_img = opt.seq_per_img}
	-- data: 
	-- data.densecap batch_size*densecap_per_img*densecao_length; 
	-- data.location batch_size*densecap_per_img*4; 
	-- data.score batch_size*densecap_per_img*1;
	-- data.seq batch_size*densecap_per_img*seq_length;
	-- data.info  batch_size
	-- data.bounds.3 (it_pos_now, it_max, wrapped)

	-- forward
	local logprobs=model:forward(data)
	local loss= crit:forward(logprobs, data.seq)

	-- backprop criterion
	local dlogprobs = crit:backward(logprobs, data.seq)
	-- backprop 
	local ddata= model:backward(data,dlogprobs)

	-- clip gradients
	grad_params:clamp(-opt.grad_clip, opt.grad_clip)
	
	-- and lets get out!
	local losses = { total_loss = loss }
	return losses

end



local function eval_batch()
	--local val_images_use = utils.getopt(evalopt, 'val_images_use', 0)
	local val_images_use = opt.val_images_use
	model:evaluate()
	loader:resetIterator('val') -- rewind iteator back to first datapoint in the split
	local n = 0
	local loss_sum = 0
	local loss_evals = 0
	local predictions = {}
	local vocab = loader:getVocab()
	
	while true do
	
	    -- fetch a batch of data
	    local data = loader:getBatch{batch_size = opt.batch_size, split = 'val', seq_per_img = 		opt.seq_per_img}
            -- forward to get loss
	    local logprobs=model:forward(data)
	    local loss= crit:forward(logprobs, data.seq)
	    loss_sum = loss_sum + loss
	    loss_evals = loss_evals + 1
	    
	    -- forward the model to also get generated samples for each image
	    sample_input = {densecap = data.densecap, score = data.score, location = data.location}
	    local generated_seq = model:sample(sample_input)
	    local genetated_sents = net_utils.decode_sequence(vocab, generated_seq)
	    for k=1,#genetated_sents do
	    	local entry = {image_id = data.infos[k].id, caption = genetated_sents[k]}
	    	table.insert(predictions, entry)
        print(string.format('image %s: %s', entry.image_id, entry.caption))
	    end
	    
	    -- if we wrapped around the split or used up val imgs budget then bail
	    local ix0 = data.bounds.it_pos_now
	    local ix1 = math.min(data.bounds.it_max, val_images_use)
	    print(string.format('evaluating validation performance... %d/%d (%f)', ix0-1, ix1, loss))
	    
	    if loss_evals % 10 == 0 then collectgarbage() end
	    if data.bounds.wrapped then break end -- the split ran out of data, lets break out
      n = n + opt.batch_size
	    if n >= val_images_use then break end -- we've used enough images
	end
	
	local lang_stats
	if opt.language_eval == 1 then
		lang_stats = net_utils.language_eval(predictions, opt.input_val, task_hash)
	end
	
	return loss_sum/loss_evals, predictions, lang_stats
end


-------------------------------------------------------------------------------
-- Main Loop
-------------------------------------------------------------------------------
local loss0
local optim_state = {}
local cnn_optim_state = {}
local loss_history = {}
local val_lang_stats_history = {}
local val_loss_history = {}
local best_score
local iter = 1

while true do
-- eval loss/gradient
  local losses = train_batch()
  if iter % opt.losses_log_every == 0 then loss_history[iter] = losses.total_loss end
  print(string.format('iter %d: %f', iter, losses.total_loss))

  if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then

    -- evaluate the validation performance
    local val_loss, val_predictions, lang_stats = eval_batch{val_images_use = opt.val_images_use}
    print('validation loss: ', val_loss)
    print(lang_stats)
  end
  
  -- decay the learning rate for both LM and CNN
  local learning_rate = opt.learning_rate
  if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
    local decay_factor = math.pow(0.5, frac)
    learning_rate = learning_rate * decay_factor -- set the decayed rate
  end
  
  -- perform a parameter update
  if opt.optim == 'rmsprop' then
    rmsprop(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'adagrad' then
    adagrad(params, grad_params, learning_rate, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'sgd' then
    sgd(params, grad_params, opt.learning_rate)
  elseif opt.optim == 'sgdm' then
    sgdm(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'sgdmom' then
    sgdmom(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'adam' then
    adam(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state)
  else
    error('bad option opt.optim')
  end
  
 -- stopping criterions
  iter = iter + 1
  if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
  if loss0 == nil then loss0 = losses.total_loss end
  if losses.total_loss > loss0 * 20 or losses.total_loss > 100 then
    print('loss seems to be exploding, quitting.')
    break
  end
  if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion

end
