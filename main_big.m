clear
close all

% load data
load data/Xtrain_3_5.mat
load data/Xtest_3_5.mat
load data/yTrain_3_5.mat
load data/yTest_3_5.mat

% Make labels
ytrain(ytrain == 3) = 1;
ytrain(ytrain == 5) = -1;
ytest(ytest == 3) = 1;
ytest(ytest == 5) = -1;

% load dicts
DD(1) = load("dicts/mnist_init_dict_k_2_d_256_p_50.mat");
DD(2) = load("dicts/mnist_init_dict_k_2_d_256_p_100.mat");
DD(3) = load("dicts/mnist_init_dict_k_2_d_256_p_200.mat");
DD(4) = load("dicts/mnist_init_dict_k_2_d_256_p_300.mat");
DD(5) = load("dicts/mnist_init_dict_k_2_d_256_p_400.mat");
w = zeros(400,5);
for zzz = 1:3
    for ddd = 1:size(DD,2)
        %% Problem parameters
        % Choose dictionary
        clear astar
        clear astar_unsup
        clear w_unsup
        D = DD(ddd).D; % will choose ith dictionary
        [m, n] = size(Xtrain); % dimension of data and dictionary
        [~, p] = size(D); % dimension of data and dictionary
        lam = 0.15; % lam = 0.15, lam2 = 0
        rho = 0.1;
        v = 10^-9;
        numIter = 5000; % number of iterations of TDDL
        t0 = numIter / 10;
        % Initialize w with D
        if zzz == 1
            for i = 1:size(Xtrain, 2)
                astar(:, i) = lasso(D, Xtrain(:, i), 'Lambda', lam);
            end
            [w(1:p,ddd), ~] = lrsgd(astar, ytrain, lam, 1000);
            fprintf("i: %d, D: %d x %d, msg - calculated initial w\n", ddd, m, p)
        end
        %% Get unsupervised
        wcurr = w(1:p,ddd);
        w_unsup = wcurr;
        for t = 1:numIter
            % draw random xt from Xtrain
            ind = randi([1,n], 1, 1);
            xt = Xtrain(:, ind);
            yt = ytrain(ind);
            % use lasso to get astar min alpha st Dxt - alpha
            astar = lasso(D, xt, 'Lambda', lam);
            % choose learning rate
            rhot = min(rho,rho*(t0/t));
            w_unsup = w_unsup - rhot*(((-yt*astar)/(1 + exp(yt*w_unsup'*astar))) + v*w_unsup);
        end
        % Calculate astar for unsupervised dictionary
        for i = 1:size(Xtest, 2)
            astar_unsup(:, i) = lasso(D, Xtest(:, i), 'Lambda', lam);
        end
        % calculate error
        err_raw(ddd,1,zzz) = sum(sign(astar_unsup'*w_unsup) ~= ytest) / length(ytest);
        fprintf("i: %d, D: %d x %d, msg - calculated unsup err: %d\n", ddd, m, p, err_raw(ddd,1,zzz))
        %% Get supervised
        for t = 1:numIter
            % draw random xt from Xtrain
            ind = randi([1,n], 1, 1);
            xt = Xtrain(:, ind);
            yt = ytrain(ind);
            % use lasso to get astar min alpha st Dxt - alpha
            astar = lasso(D, xt, 'Lambda', lam);
            % compute activeset
            Lam = find(astar);
            bstar = zeros(p,1);
            Dlam = D(:,Lam);
            bstar(Lam) = (Dlam'*Dlam + (1e-3)*eye(size(Dlam,2)))\((-(yt*w(Lam))/(exp(yt*w(Lam)'*astar(Lam)) + 1)));
            % choose learning rate
            rhot = min(rho,rho*(t0/t));
            % update w
            wcurr = wcurr - rhot*(((-yt*astar)/(1 + exp(yt*wcurr'*astar))) + v*wcurr);
            % project and update D
            D = D - rhot*(-D * bstar*astar' + (xt - D*astar)*bstar');
            % test if needs to be normalized
            cumul = 0;
            for col = 1:p
                cumul = cumul + norm(D(:,col));
            end
            if cumul ~= p
                xx = sqrt(sum(D.^2,1)); % Compute norms of columns
                D = bsxfun(@rdivide,D,xx); % Normalize D
                D(isnan(D)) = 0; % remove NaNs
            end
        end
        % Calculate astar for supervised dictionary
        for i = 1:size(Xtest, 2)
            astar(:, i) = lasso(D, Xtest(:, i), 'Lambda', lam);
        end
        % calculate error super
        err_raw(ddd,2,zzz) = sum(sign(astar'*wcurr) ~= ytest) / length(ytest);
        fprintf("i: %d, D: %d x %d, msg - calculated sup err: %d\n", ddd, m, p, err_raw(ddd,2,zzz))
    end
end
err = mean(err_raw, 3);