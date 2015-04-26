using Mocha

backend = CPUBackend()
init(backend)

data = AsyncHDF5DataLayer(name="data", source="train_data.txt", batch_size=250, shuffle=true)
loss = SoftmaxLossLayer(name="loss",bottoms=[:pred,:label])

common_layers = [InnerProductLayer(name="ip1",output_dim=10,tops=[:ip1],bottoms=[:data]),
								 IdentityLayer(name="pred",tops=[:pred],bottoms=[:ip1])]
net = Net("images", backend, [data, common_layers..., loss])

exp_dir= "snapshots"
params = SolverParameters(max_iter=10000, regu_coef=0.0005,
	mom_policy=MomPolicy.Fixed(0.9),
	lr_policy=LRPolicy.Inv(0.01,0.0001,0.75),
	load_from=exp_dir)
solver=SGD(params)

setup_coffee_lounge(solver, save_into="$exp_dir/output.jld", every_n_iter=1000)

# report training progress every 100 iterations
add_coffee_break(solver,TrainingSummary(),every_n_iter=250)

# save snapshots every 1000 iterations
# add_coffee_break(solver, Snapshot(exp_dir), every_n_iter=1000)

# show performance on test data every 1000 iterations
data_test = AsyncHDF5DataLayer(name="data-test",source="train_test_data.txt",batch_size=250)
accuracy_test = AccuracyLayer(name="accuracy-test",bottoms=[:pred, :label])
net_test = Net("images-test", backend, [data_test, common_layers..., accuracy_test])
add_coffee_break(solver, ValidationPerformance(net_test), every_n_iter=1000)

solve(solver, net)

data_val = AsyncHDF5DataLayer(name="data-val",source="validate_data.txt",batch_size=250,tops=[:data])
output_val = HDF5OutputLayer(name="output-val",filename="validate_output.h5",force_overwrite=true,bottoms=[:pred],datasets=[:label])
net_val = Net("images-val", backend, [data_val, common_layers..., output_val])
forward(net_val)

destroy(net)
destroy(net_test)
destroy(net_val)
shutdown(backend)
