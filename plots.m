clear
close all

% load error values
err(1) = load("results/err_100.mat");
err(2) = load("results/err_500.mat");
err(3) = load("results/err_1000.mat");
err(4) = load("results/err_5000.mat");

% dictionary size vector
x = [50,100,200,300,400];

% Plot unsupervised dictionary size versus error
figure
hold on
plot(x, err(1).err(:,1)*100)
plot(x, err(2).err(:,1)*100)
plot(x, err(3).err(:,1)*100)
plot(x, err(4).err(:,1)*100)
ylim([0,10])
title("Unsupervised - Dictionary Size vs. Error by Iteration")
xlabel("Dictionary Size")
ylabel("Percent Error")
legend('T = 100','T = 500','T = 1000','T = 5000')
f = gcf;
exportgraphics(f,'unsup.png')
% Plot supervised dictionary size versus error
figure
hold on
plot(x, err(1).err(:,2)*100)
plot(x, err(2).err(:,2)*100)
plot(x, err(3).err(:,2)*100)
plot(x, err(4).err(:,2)*100)
ylim([0,10])
title("Supervised - Dictionary Size vs. Error by Iteration")
xlabel("Dictionary Size")
ylabel("Percent Error")
legend('T = 100','T = 500','T = 1000','T = 5000')
f = gcf;
exportgraphics(f,'sup.png')
