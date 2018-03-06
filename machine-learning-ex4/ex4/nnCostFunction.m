function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Add ones to the X data matrix
X = [ones(m, 1) X];
% Convert y from (1-10) class into num_labels vector
yd = eye(num_labels);
y = yd(y,:);
 
%%% Map from Layer 1 to Layer 2
a1=X;
% Coverts to matrix of 5000 examples x 26 thetas
z2=X*Theta1';
% Sigmoid function converts to p between 0 to 1
a2=sigmoid(z2);

%%% Map from Layer 2 to Layer 3
% Add ones to the h1 data matrix
a2=[ones(m, 1) a2];
% Converts to matrix of 5000 exampls x num_labels 
z3=a2*Theta2';
% Sigmoid function converts to p between 0 to 1
a3=sigmoid(z3);

% Compute cost
%logisf=(-y)'*log(a3)-(1-y)'*log(1-a3);
logisf=(-y).*log(a3)-(1-y).*log(1-a3); % Becos y is now a matrix, so use dot product, unlike above
%J=((1/m).*sum(sum(logisf)));	% This line is correct if there is no regularization
% Try with ...
% J=((1/m).*sum((logisf)));  
% That will give J in 10 columns (it has summed m samples), so need to sum again

%% Regularized cost
Theta1s=Theta1(:,2:end);
Theta2s=Theta2(:,2:end);
J=((1/m).*sum(sum(logisf)))+(lambda/(2*m)).*(sum(sum(Theta1s.^2))+sum(sum(Theta2s.^2)));





% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

% Set all the D to zeros
Delta_1=0;  
Delta_2=0;

% Compute delta, tridelta and big D
	delta_3=a3-y;
    z2=[ones(m,1) z2];
	delta_2=delta_3*Theta2.*sigmoidGradient(z2);
    delta_2=delta_2(:,2:end);
	Delta_1=Delta_1+delta_2'*a1; % Same size as Theta1_grad (25x401)
    Delta_2=Delta_2+delta_3'*a2; % Same size as Theta2_grad (10x26)
% 	Theta1_grad=(1/m).*Delta_1;
%     Theta2_grad=(1/m).*Delta_2;
   

% %end




% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad=1/m*Delta_1;
lambdaPlus=Theta1*lambda/m;
lambdaPlus(:,1)=0;
Theta1_grad=Theta1_grad+lambdaPlus;


Theta2_grad=1/m*Delta_2;
lambdaPlus=Theta2*lambda/m;
lambdaPlus(:,1)=0;
Theta2_grad=Theta2_grad+lambdaPlus;



%  另一种方法 （nnCostFunction）

% X = [ones(m, 1) X];
% z2 = X * Theta1';
% a2 = sigmoid(z2);
% a2 = [ones(m, 1) a2];
% z3 = a2 * Theta2';
% a3 = sigmoid(z3);
% output=a3;
% 
% ym = ones(size(output));
% for i=1:m
%   target = y(i);
%   yi = zeros(1, num_labels);
%   yi(target) = 1;
%   ym(i, :) = yi;
% end;
% pos = log(output);
% neg = log(1 - output);
% cost = sum(sum(pos .* ym + neg .* (1 - ym)));
% 
% % cost = 0;
% % for i=1:m
% %   target = y(i);
% %   yi = zeros(num_labels, 1);
% %   yi(target) = 1;
% %   hx = output(i, :);
% %   pos = log(hx);
% %   neg = log(1 - hx);
% %   cost = cost + pos * yi + neg * (1 - yi);
% % end
% 
% 
% J = -1 / m * cost;
% bias1 = Theta1(:, 1);
% bias2 = Theta2(:, 1);
% reg = nn_params' * nn_params - bias1' * bias1 - bias2' * bias2;
% J = J + lambda / (2 * m) * reg;
% 
% 
% Delta2 = zeros(size(Theta2)); % 10 * 26
% Delta1 = zeros(size(Theta1)); % 25 * 401
% 
% % for t=1:m
% %   a1 = X(t, :)';
% %   z2 = Theta1 * a1;
% %   a2 = sigmoid(z2);
% %   a2 = [1; a2];
% %   z3 = Theta2 * a2;
% %   a3 = sigmoid(z3);
% %   yt = zeros(num_labels, 1);
% %   yt(y(t)) = 1;
% %   delta3 = a3 - yt;
% %   delta2 = Theta2' * delta3 .* [1; sigmoidGradient(z2)];
% %   % delta2 = Theta2' * delta3 .* a2 .* (1 - a2);
% %   delta2 = delta2(2:end);
% %   Delta2 = Delta2 + delta3 * a2';
% %   Delta1 = Delta1 + delta2 * a1';
% % end
% 
% 
% a1 = X;
% delta3 = a3 - ym; % 5000 * 10
% % Theta2 10 * 26
% delta2 = delta3 * Theta2 .* a2 .* (1 - a2); %    5000 * 26, a2: 5000 * 26
% % 10 * 5000, 5000 * 26
% Delta2 = delta3' * a2;
% % 26 * 5000, 5000 * 401
% Delta1 = (delta2(:, 2:end))' * a1;
% 
% 
% Theta1_grad = 1 / m * Delta1;
% lambdaPlus = Theta1 * lambda / m;
% lambdaPlus(:, 1) = 0;
% Theta1_grad = Theta1_grad + lambdaPlus;
% 
% Theta2_grad = 1 / m * Delta2;
% lambdaPlus = Theta2 * lambda / m;
% lambdaPlus(:, 1) = 0;
% Theta2_grad = Theta2_grad + lambdaPlus;















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
