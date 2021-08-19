function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

sum = 0;

%Expected theta values (approx)

% -3.6303
%  1.1664
%
%theta = [-1; 1];
%Theta found by gradient descent:
%-1.029113
%0.841718
%theta = [-1; 1.5];
%-1.119050
%0.852740
%theta = [-1.5; 2];
%-1.697965
%0.923687
%theta = [-2; 3];
%-2.366816
%1.005656
%theta = [-2.5; 3.5];
%-2.945731
%1.076603
%theta = [-3.5; 4.5];
%-4.103561
%1.218498
%theta = [-3; 4];
%-3.524646
%1.147550
%theta = [-3.2; 4.3];
%-3.774200
%1.178134
##theta = [-3.1; 4.2];
##-3.658417
##1.163944
##theta = [-3.05; 4.25];
##?-3.618513
##1.159054
##theta = [-3.08; 4.28];
##3.653247
##1.163311
##theta = [-3.07; 4.29];
##-3.645267
##1.162333
##theta = [-3.059; 4.298];
##-3.636926
##1.161310
##theta = [-3.055; 4.299];
##-3.632216
##1.160733
##theta = [-3.053; 4.2995];
##-3.630350
##1.160505

##theta = [-3.0535; 4.2999];
##-3.630911
##1.160573
##theta = [-3.0531; 4.3];
##-3.630538
##1.160528
##theta = [-3.05302; 4.4];
##-3.648447
##1.162722
##theta = [-3.0530; 4.5];
##-3.666415
##1.164924
##expect
## -3.6303
##  1.1664
%theta = [-3.051; 4.6];


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    theta -= alpha / m * X' * (X * theta - y);
##    for i = 1:m
##      h = theta' * X(i,:)';
##      %disp(X(i, :));
##      %disp(theta);
##      %disp(h);
##      %pause;
##      sum = sum + (h - y(i));
##      %disp(sum);
##      %pause;
##    endfor

##    tmp1 = theta(1) - (alpha / m) * sum;
##    tmp2 = theta(2) - (alpha / m) * sum * X(i, 2);
##    theta(1) = tmp1;
##    theta(2) = tmp2;
##    sum = 0;
##    disp(theta);
    %pause;






    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
