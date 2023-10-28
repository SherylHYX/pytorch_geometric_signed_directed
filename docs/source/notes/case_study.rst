Case Study Examples
=============

In the following we will overview two case studies where PyTorch Geometric Signed Directed can be used to solve relevant machine learning problems. One is on signed networks and the other is on directed networks.

Case Study on Signed Networks
---------------------------

Here, we overview a simple end-to-end machine learning pipeline designed with *PyTorch Geometric Signed Directed* for signed networks. These code snippets solve a signed clustering problem on a Signed Stochastic Block Model -- clustering the nodes in the signed network into 5 groups. The pipeline consists of data preparation, model definition, training, and evaluation phases.

.. code-block:: python

    from sklearn.metrics import adjusted_rand_score
    import scipy.sparse as sp
    import torch
    from torch_geometric_signed_directed.nn import \
        SSSNET_node_clustering
    from torch_geometric_signed_directed.data import \
        SignedData, SSBM
    from torch_geometric_signed_directed.utils import \
        (Prob_Balanced_Normalized_Loss, 
    extract_network, triplet_loss_node_classification)

    device = torch.device('cuda' if \
    torch.cuda.is_available() else 'cpu')

    num_classes = 5
    eta = 0.1
    num_nodes = 1000
    p = 0.1
    (A_p_scipy, A_n_scipy), labels = SSBM(num_nodes, \ 
    num_classes, p, eta)
    A = A_p_scipy - A_n_scipy
    A, labels = extract_network(A=A, labels=labels)
    data = SignedData(A=A, y=torch.LongTensor(labels))
    data.set_spectral_adjacency_reg_features(num_classes)
    data.node_split(train_size_per_class=0.8, \ 
    val_size_per_class=0.1, \ 
    test_size_per_class=0.1, seed_size_per_class=0.1)
    data.separate_positive_negative()
    data = data.to(device)

In the above code snippet, as a first step, we import the SSBM data generator, SignedData class, the network to be used, and evaluation functions. We then define the device to be used in this example. 
After that, we define default values to be used in the network generation process, generate the synthetic network and extract the largest connected component. As no node features are available initially, we use the ``set_signed_Laplacian_features()`` class method to set up the node feature matrix. We then create a train-validation-test-seed split of the node set by using the node splitting function and calculate separated positive and negative parts of the signed network to be stored inside the data object. 
Finally, we move the data object to the device.

.. code-block:: python

    loss_func_ce = torch.nn.NLLLoss()

    model = SSSNET_node_clustering(nfeat=data.x.shape[1], dropout=0.5,  
    hop=2, fill_value=0.5, hidden=32, nclass=num_classes).to(device)

For the second snippet, we first initialize the cross-entropy loss function as part of the supervised loss. Then we construct the neural network model and map it to the device. 

.. code-block:: python

    def train(features, edge_index_p, edge_weight_p,
                    edge_index_n, edge_weight_n, mask, seed_mask,
                    loss_func_pbnc, y):
        model.train()
        Z, log_prob, _, prob = model(edge_index_p, edge_weight_p,
                    edge_index_n, edge_weight_n, features)
        loss_pbnc = loss_func_pbnc(prob[mask])
        loss_triplet = triplet_loss_node_classification(y=y[seed_mask], 
        Z=Z[seed_mask], n_sample=500, thre=0.1)
        loss_ce = loss_func_ce(log_prob[seed_mask], y[seed_mask])
        loss = 50*(loss_ce + 0.1*loss_triplet) + loss_pbnc
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_ari = adjusted_rand_score(y[mask].cpu(),
        (torch.argmax(prob, dim=1)).cpu()[mask])
        return loss.detach().item(), train_ari

    def test(features, edge_index_p, edge_weight_p,
                    edge_index_n, edge_weight_n, mask, y):
        model.eval()
        with torch.no_grad():
            _, _, _, prob = model(edge_index_p, edge_weight_p,
                    edge_index_n, edge_weight_n, features)
        test_ari = adjusted_rand_score(y[mask].cpu(),
        (torch.argmax(prob, dim=1)).cpu()[mask])
        return test_ari

In the third snippet, we define the training and evaluation functions. Setting the model to be trainable, we obtain node embedding matrix Z and cluster assignment probablities prob and its logarithm log_prob with a forward pass of the model instance. We then obtain the probabilistic balanced normalized cut loss, triplet loss, and cross entropy loss. The weighted sum of the three losses then serves as the training loss value. We then backpropagate and update the model parameters. After that, we calculate the Adjusted Rand Index (ARI) \cite{hubert1985comparing} of the training samples. Finally, we return the loss value as well as the training ARI score.

For the evaluation function (named ``test``), we do not set the model to be trainable. With a forward pass, we obtain the probability assignment matrix. Taking argmax for the probabilities, we obtain test ARI result. Finally, we return the result.

.. code-block:: python

    for split in range(data.train_mask.shape[1]):
        optimizer = torch.optim.Adam(model.parameters(),
        lr=0.01, weight_decay=0.0005)
        train_index = data.train_mask[:, split].cpu().numpy()
        val_index = data.val_mask[:, split]
        test_index = data.test_mask[:, split]
        seed_index = data.seed_mask[:, split]
        loss_func_pbnc = Prob_Balanced_Normalized_Loss(
        A_p=sp.csr_matrix(data.A_p)[train_index][:, train_index], 
        A_n=sp.csr_matrix(data.A_n)[train_index][:, train_index])
        for epoch in range(300):
            train_loss, train_ari = train(data.x,
            data.edge_index_p,
            data.edge_weight_p, data.edge_index_n,
            data.edge_weight_n, train_index,
            seed_index, loss_func_pbnc, data.y)
            Val_ari = test(data.x, data.edge_index_p,
            data.edge_weight_p, data.edge_index_n,
            data.edge_weight_n, val_index, data.y)
            print(f'Split: {split:02d}, Epoch: {epoch:03d}, 
            Train_Loss: {train_loss:.4f},
            Train_ARI: {train_ari:.4f},
            Val_ARI: {Val_ari:.4f}')
        
        test_ari = test(data.x, data.edge_index_p, 
        data.edge_weight_p, data.edge_index_n,
        data.edge_weight_n, test_index, data.y)
        print(f'Split: {split:02d}, Test_ARI: {test_ari:.4f}')
        model._reset_parameters_undirected()
    
We run the actual experiments in this final snippet. For each of the data splits, we first initialize the Adam optimizer. We then obtain the data split indices, initialize the self-supervised loss function, and start the training process. For each epoch,  we apply the training function to obtain training loss and ARI score, then evaluate with the ``test()`` function on validation nodes.  We then print the training and validation results. 
After training, we obtain the test performance and print some logs. Finally, we reset model parameters and iterate to the next data split loop.

Case Study on Directed Networks
----------------------

In the following code snippets, we overview a simple end-to-end machine learning pipeline designed with *PyTorch Geometric Signed Directed* for directed networks. These code snippets solve a link direction prediction problem on a real-world data set. The pipeline consists of data preparation, model definition, training, and evaluation phases.

.. code-block:: python

    from sklearn.metrics import accuracy_score
    import torch

    from torch_geometric_signed_directed.utils import \ 
    link_class_split, in_out_degree
    from torch_geometric_signed_directed.nn.directed import \ 
    MagNet_link_prediction
    from torch_geometric_signed_directed.data import \ 
    load_directed_real_data

    device = torch.device('cuda' if \
    torch.cuda.is_available() else 'cpu')

    data = load_directed_real_data(dataset='webkb', 
    root=path, name='cornell').to(device)
    link_data = link_class_split(data, prob_val=0.15, 
    prob_test=0.05, task = 'direction', device=device)

First of all, after importing and defining the device, we load the ``DirectedData`` object for the selected data set and map it to the device. We then create a train-validation-test split of the edge set by using the directed link splitting function. 

.. code-block:: python

    model = MagNet_link_prediction(q=0.25, K=1, num_features=2, 
    hidden=16, label_dim=2).to(device)
    criterion = torch.nn.NLLLoss()

In the second snippet, we first construct the model instance, then initialize the cross-entropy loss function.

.. code-block:: python

    def train(X_real, X_img, y, edge_index,
    edge_weight, query_edges):
        model.train()
        out = model(X_real, X_img, edge_index=edge_index, 
                        query_edges=query_edges, 
                        edge_weight=edge_weight)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc = accuracy_score(y.cpu(),
        out.max(dim=1)[1].cpu())
        return loss.detach().item(), train_acc

    def test(X_real, X_img, y, edge_index, edge_weight, 
    query_edges):
        model.eval()
        with torch.no_grad():
            out = model(X_real, X_img, edge_index=edge_index, 
                        query_edges=query_edges, 
                        edge_weight=edge_weight)
        test_acc = accuracy_score(y.cpu(),
        out.max(dim=1)[1].cpu())
        return test_acc

In the third part, we define the training and evaluation functions. Setting the model to be trainable, we obtain edge class assignment probablities with a forward pass of the model instance. We then obtain the training loss value. After that, we backpropagate and update the model parameters. Then, we calculate the accuracy of the training samples. Finally, we return the loss value as well as the training accuracy.

For the evaluation function (named ``test``), we do not set the model to be trainable. With a forward pass, we obtain the probability assignment matrix. We then obtain test accuracy and return the result.

.. code-block:: python

    for split in list(link_data.keys()):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, 
        weight_decay=0.0005)
        edge_index = link_data[split]['graph']
        edge_weight = link_data[split]['weights']
        query_edges = link_data[split]['train']['edges']
        y = link_data[split]['train']['label']
        X_real = in_out_degree(edge_index,
        size=len(data.x)).to(device)
        X_img = X_real.clone()
        query_val_edges = link_data[split]['val']['edges']
        y_val = link_data[split]['val']['label']
        for epoch in range(200):
            train_loss, train_acc = train(X_real,
            X_img, y, edge_index, edge_weight, query_edges)
            val_acc = test(X_real, X_img, y_val,
            edge_index, edge_weight, query_val_edges)
            print(f'Split: {split:02d}, Epoch: {epoch:03d}, \
            Train_Loss: {train_loss:.4f}, Train_Acc: \
            {train_acc:.4f}, Val_Acc: {val_acc:.4f}')
        
        query_test_edges = link_data[split]['test']['edges']
        y_test = link_data[split]['test']['label']  
        test_acc = test(X_real, X_img, y_test, edge_index, 
        edge_weight, query_test_edges)
        print(f'Split: {split:02d}, Test_Acc: {test_acc:.4f}')
        model.reset_parameters()

We run the actual experiments in the last code snippet. For each of the data splits, we first initialize the optimizer. We then prepare data objects to be used, and start the training process. For each epoch,  we apply the training function to obtain training loss and accuracy, then evaluate with the ``test()`` function on validation nodes.  We then print the training and validation results. 
After training, we prepare test data, obtain the test performance, and print some logs. Finally, we reset model parameters and iterate to the next data split loop.
