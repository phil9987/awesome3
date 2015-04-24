using Mocha



data = HDF5DataLayer(name="data", source="train_data.txt", batch_size=64, shuffle=true)
fc1 = InnerProductLayer(name="ip1",output_dim=10,tops=[:ip1],bottoms=[:data],)
loss = SoftmaxLossLayer(name="loss",bottoms=[:ip1,:label])

backend = CPUBackend()
init(backend)

common_layers = [fc1]
net = Net("images", backend, [data, common_layers..., loss])

exp_dir= "snapshots"
params = SolverParameters(max_iter=10000, regu_coef=0.0005,
	mom_policy=MomPolicy.Fixed(0.9),
	lr_policy=LRPolicy.Inv(0.01,0.0001,0.75),
	load_from=exp_dir)
solver=SGD(params)

setup_coffe_lounge(solver, save_into="$exp_dir/output.jld", every_n_iter=1000)

# report training progress every 100 iterations
add_coffee_break(solver,TrainingSummary(),every_n_iter=100)

# save snapshots every 5000 iterations
add_coffee_break(solver, Snapshot(exp_dir), every_n_iter=5000)

# show performance on test data every 1000 iterations
data_test = AsyncHDF5DataLayer(name="test-data",source="validate_data.txt",batch_size=100)
accuracy = AccuracyLayer(name="test-accuracy",bottoms=[:ip1, :label])
test_net = Net("images-test", backend, [data_test, common_layers..., accuracy])
add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=1000)

solve(solver, net)

destroy(net)
destroy(test_net)
shutdown(backend)