ENV["MOCHA_USE_NATIVE_EXT"] = "true"

using Mocha

n_hidden_layer   = 3
n_hidden_unit    = 1000
neuron           = Neurons.Sigmoid()
param_key_prefix = "ip-layer"
corruption_rates = [0.1,0.2,0.3]
pretrain_epoch   = 15
finetune_epoch   = 1000
batch_size       = 100
momentum         = 0.0
pretrain_lr      = 0.001
finetune_lr      = 0.1

param_keys       = ["$param_key_prefix-$i" for i = 1:n_hidden_layer]

backend = CPUBackend()
init(backend)

data_layer = AsyncHDF5DataLayer(name="data", source="train_data.txt", batch_size=100, shuffle=true)

# --start-basic-layers--
rename_layer = IdentityLayer(bottoms=[:data], tops=[:ip0])
hidden_layers = [InnerProductLayer(name="ip-$i", param_key=param_keys[i],
																								 output_dim=n_hidden_unit, neuron=neuron,
																								 bottoms=[symbol("ip$(i-1)")],
																								 tops=[symbol("ip$i")])
									for i = 1:n_hidden_layer]
# --end-basic-layers--

for i = 1:n_hidden_layer
				ae_data_layer = SplitLayer(bottoms=[symbol("ip$(i-1)")], tops=[:orig_data, :corrupt_data])
				corrupt_layer = RandomMaskLayer(ratio=corruption_rates[i], bottoms=[:corrupt_data])

				encode_layer  = copy(hidden_layers[i], bottoms=[:corrupt_data])
				recon_layer   = TiedInnerProductLayer(name="tied-ip-$i", tied_param_key=param_keys[i],
																																 tops=[:recon],
																																 bottoms=[symbol("ip$i")])
				recon_loss_layer = SquareLossLayer(bottoms=[:recon, :orig_data])

				da_layers = [data_layer, rename_layer, ae_data_layer, corrupt_layer,
												 hidden_layers[1:i-1]..., encode_layer, recon_layer, recon_loss_layer]
				da = Net("Denoising-Autoencoder-$i", backend, da_layers)
				println(da)

				# freeze all but the layers for auto-encoder
				freeze_all!(da)
				unfreeze!(da, "ip-$i", "tied-ip-$i")

				loss = SquareLossLayer(name="loss", bottoms=[:pred, :label])

				base_dir = "pretrain-$i"
				pretrain_params  = SolverParameters(max_iter=div(pretrain_epoch*60000,batch_size),
																				regu_coef=0.0, mom_policy=MomPolicy.Fixed(momentum),
																				lr_policy=LRPolicy.Fixed(pretrain_lr), load_from=base_dir)
				solver = SGD(pretrain_params)

				add_coffee_break(solver, TrainingSummary(), every_n_iter=1000)
				add_coffee_break(solver, Snapshot(base_dir), every_n_iter=3000)
				solve(solver, da)

				destroy(da)
end

pred_layer = InnerProductLayer(name="pred", output_dim=10,
																						    bottoms=[symbol("ip$n_hidden_layer")], tops=[:pred])
loss_layer = SoftmaxLossLayer(bottoms=[:pred, :label])

net = Net("finetune", backend,
					[data_layer, rename_layer, hidden_layers..., pred_layer, loss_layer])

base_dir = "finetune"
params = SolverParameters(max_iter=div(finetune_epoch*60000,batch_size),
													 regu_coef=0.0, mom_policy=MomPolicy.Fixed(momentum),
													 lr_policy=LRPolicy.Fixed(finetune_lr), load_from=base_dir)
solver = SGD(params)

setup_coffee_lounge(solver, save_into="$base_dir/statistics.jld", every_n_iter=10000)

add_coffee_break(solver, TrainingSummary(), every_n_iter=1000)
add_coffee_break(solver, Snapshot(base_dir), every_n_iter=10000)

data_test = AsyncHDF5DataLayer(name="data-test",source="train_test_data.txt",batch_size=100)
acc_layer = AccuracyLayer(name="accuracy-test", bottoms=[:pred, :label])
net_test = Net("finetune-test", backend, [data_layer_test, rename_layer,
																																     hidden_layers..., pred_layer, acc_layer])
add_coffee_break(solver, ValidationPerformance(net_test), every_n_iter=5000)

solve(solver, net)

destroy(net)
destroy(net_test)

shutdown(backend)
